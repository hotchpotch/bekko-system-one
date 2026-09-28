import { validate } from './core.js';

export const DEFAULT_YES = 'Yes, the condition in the question holds.';
export const DEFAULT_NO = 'No, the condition in the question does not hold.';

export function parseCriteria(task, lines, yes = DEFAULT_YES, no = DEFAULT_NO) {
  if (task === 'noul') {
    if (!yes.trim() || !no.trim()) throw Error('Give both Yes and No a meaning.');
    return [{ id: 'true', description: yes.trim() }, { id: 'false', description: no.trim() }];
  }
  const rows = lines.split('\n').map(line => line.trim()).filter(Boolean);
  if (rows.length < 2 || rows.length > 64) throw Error('Enter between 2 and 64 options, one per line.');
  const criteria = rows.map((line, index) => {
    const divider = line.indexOf('|');
    if (task === 'score') {
      const valueText = divider < 0 ? '' : line.slice(0, divider).trim();
      const value = Number(valueText), description = line.slice(divider + 1).trim();
      if (!valueText || !Number.isFinite(value) || !description) throw Error(`Line ${index + 1}: use a number followed by | and a description.`);
      return { id: String(value), description, value };
    }
    const id = divider < 0 ? line : line.slice(0, divider).trim();
    const description = divider < 0 ? line : line.slice(divider + 1).trim();
    if (!id || !description) throw Error(`Line ${index + 1}: add an option name and description.`);
    return { id, description };
  });
  if (new Set(criteria.map(c => c.id)).size !== criteria.length) throw Error('Use a different name or numeric value for each option.');
  return criteria;
}

export function renderDecision(decision) {
  const { task, instruction, state, criteria } = decision;
  if (state === undefined || !Array.isArray(criteria)) throw Error('Provide context and criteria.');
  if (criteria.some(c => typeof c.description !== 'string' || !c.description.trim())) throw Error('Each criterion needs a description.');
  let modelState = state;
  if (task === 'noul') {
    const yes = criteria.find(c => ['true', 'yes'].includes(c.id));
    const no = criteria.find(c => ['false', 'no'].includes(c.id));
    if (!yes || !no) throw Error('Yes / No requires two criteria.');
    modelState = { noul: { yes: yes.description, no: no.description }, state };
  }
  const request = {
    task, instruction, state: JSON.stringify(modelState), layout: 'instruction_state',
    candidates: criteria.map(c => ({ id: c.id, text: `Candidate: ${c.id}: ${c.description}`, ...(task === 'score' ? { value: c.value } : {}) })),
  };
  validate(request);
  return request;
}
