import { Tokenizer } from '@huggingface/tokenizers';

export function validate(request) {
  const { task, instruction, state, system = '', layout = 'instruction_state', candidates } = request;
  if (!['choice', 'noul', 'score'].includes(task)) throw Error('task must be choice, noul or score');
  if (typeof instruction !== 'string' || !instruction.trim() || typeof state !== 'string' || typeof system !== 'string') throw Error('Provide instruction and state strings');
  if (!['instruction_state', 'state_instruction'].includes(layout)) throw Error('Invalid layout');
  if (!Array.isArray(candidates) || candidates.length < 2 || candidates.length > 64) throw Error('Provide 2–64 candidates');
  if (candidates.some(c => !c || typeof c.id !== 'string' || !c.id.trim() || typeof c.text !== 'string') || new Set(candidates.map(c => c.id)).size !== candidates.length) throw Error('Candidates need unique nonempty string IDs and text');
  if (task === 'noul' && (candidates.length !== 2 || ![['true', 'false'], ['yes', 'no']].some(pair => pair.every(id => candidates.some(c => c.id === id))))) throw Error('Noul requires true/false or yes/no IDs');
  if (task === 'score' && (candidates.some(c => !Number.isFinite(c.value)) || new Set(candidates.map(c => c.value)).size !== candidates.length)) throw Error('Score requires distinct finite numeric values');
}

export function createTokenizer(tokenizerJson, config) {
  return new Tokenizer(tokenizerJson, config);
}

export function tokenize(request, tokenizer, manifest) {
  validate(request);
  const encode = text => tokenizer.encode(text, { add_special_tokens: false }).ids;
  const max = manifest.query_length;
  const system = encode(request.system?.trim() ? `${request.system}\n\n` : '').slice(0, max);
  const instruction = encode(request.instruction).slice(0, max);
  const state = encode(request.state).slice(0, max);
  const im = encode('Instruction: '), sm = encode('State: '), sep = encode('\n');
  const budget = max - 2 - system.length - im.length - sm.length - sep.length;
  if (budget < 2) throw Error('System prompt leaves fewer than two content tokens');
  let ni = Math.min(instruction.length, Math.ceil(budget / 2));
  let ns = Math.min(state.length, Math.floor(budget / 2));
  ni += Math.min(instruction.length - ni, budget - ni - ns);
  ns += Math.min(state.length - ns, budget - ni - ns);
  const ins = [...im, ...instruction.slice(0, ni)], ctx = [...sm, ...state.slice(0, ns)];
  const body = request.layout === 'state_instruction' ? [...ctx, ...sep, ...ins] : [...ins, ...sep, ...ctx];
  return {
    prefix_ids: [[manifest.cls_token_id, ...system, ...body, manifest.sep_token_id]],
    doc_ids: request.candidates.map(c => [...encode(c.text).slice(0, manifest.document_length - 1), manifest.sep_token_id]),
  };
}

export function feedsFor(tokens, ort, manifest) {
  function padded(rows) {
    const width = Math.max(...rows.map(row => row.length));
    const ids = new BigInt64Array(rows.length * width).fill(BigInt(manifest.pad_token_id));
    const mask = new Uint8Array(ids.length);
    rows.forEach((row, i) => row.forEach((id, j) => { ids[i * width + j] = BigInt(id); mask[i * width + j] = 1; }));
    return [new ort.Tensor('int64', ids, [rows.length, width]), new ort.Tensor('bool', mask, [rows.length, width])];
  }
  const [prefix_ids, prefix_mask] = padded(tokens.prefix_ids);
  const [doc_ids, doc_mask] = padded(tokens.doc_ids);
  return { prefix_ids, prefix_mask, doc_ids, doc_mask, owners: new ort.Tensor('int64', new BigInt64Array(tokens.doc_ids.length), [tokens.doc_ids.length]) };
}

export function interpret(request, logits) {
  validate(request);
  if (logits.length !== request.candidates.length || !logits.every(Number.isFinite)) throw Error('Invalid model logits');
  const max = Math.max(...logits), exp = logits.map(x => Math.exp(x - max));
  const sum = exp.reduce((a, b) => a + b, 0), p = exp.map(x => x / sum);
  const probabilities = Object.fromEntries(request.candidates.map((c, i) => [c.id, p[i]]));
  if (request.task === 'choice') return { selected_id: request.candidates[p.indexOf(Math.max(...p))].id, probabilities };
  if (request.task === 'noul') return { probability_yes: probabilities.true ?? probabilities.yes, probabilities };
  const values = request.candidates.map(c => c.value);
  const score = p.reduce((total, prob, i) => total + prob * values[i], 0);
  return { score, normalized_score: (score - Math.min(...values)) / (Math.max(...values) - Math.min(...values)), probabilities, values: Object.fromEntries(request.candidates.map(c => [c.id, c.value])) };
}

export async function predict(request, { tokenizer, manifest, session, ort }, { onInferenceTime } = {}) {
  const tokens = tokenize(request, tokenizer, manifest);
  const column = manifest.tasks.indexOf(request.task);
  if (column < 0) throw Error('Task is absent from this model');
  // Share prefix K/V across candidates in each run. Bound candidate attention
  // growth for long inputs; this is a batching heuristic, not a total RAM limit.
  const documentLength = Math.max(...tokens.doc_ids.map(doc => doc.length));
  const attentionPairs = documentLength * (tokens.prefix_ids[0].length + documentLength);
  const batchSize = Math.max(1, Math.floor(1_048_576 / attentionPairs));
  const logits = [];
  let inferenceMilliseconds = 0;
  for (let offset = 0; offset < tokens.doc_ids.length; offset += batchSize) {
    const documents = tokens.doc_ids.slice(offset, offset + batchSize);
    const feeds = feedsFor({ prefix_ids: tokens.prefix_ids, doc_ids: documents }, ort, manifest);
    const start = performance.now();
    const result = await session.run(feeds);
    inferenceMilliseconds += performance.now() - start;
    const output = result.logits;
    if (output.dims.length !== 2 || output.dims[0] !== documents.length || output.dims[1] !== manifest.tasks.length) {
      throw Error('Invalid model logits shape');
    }
    for (let row = 0; row < documents.length; row++) {
      logits.push(Number(output.data[row * manifest.tasks.length + column]));
    }
  }
  onInferenceTime?.(inferenceMilliseconds);
  return interpret(request, logits);
}
