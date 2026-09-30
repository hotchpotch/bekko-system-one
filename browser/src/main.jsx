import React, { useEffect, useRef, useState } from "react";
import { Info, ChevronDown, HardDriveDownload, ArrowUpRight, BookOpen, Download, LoaderCircle, CircleCheck, Cpu, CircuitBoard, Play, RotateCcw, Plus, Timer, Braces, ChartNoAxesColumnIncreasing, SlidersHorizontal } from "lucide-react";
import { createRoot } from "react-dom/client";
import "./style.css";
import examples from "./examples.json";
import { RESOURCE_LINKS } from "./links.js";
import { MODELS, DEFAULT_MODEL_ID } from "./models.js";
import {
  DEFAULT_YES,
  DEFAULT_NO,
  parseCriteria,
  renderDecision,
} from "./decision.js";
import { Button, Textarea, ModelLoadProgress, Picker, HelpTooltip, DecisionActivity } from "./ui.jsx";

import { contextKeyError, contextObject } from "./context.js";

import { detectWebGPU, deviceLabel } from "./runtime.js";

const pretty = (value) => JSON.stringify(value, null, 2);
const percent = (value) => `${(value * 100).toFixed(1)}%`;
const defaults = {
  noul: "email-project-update",
  choice: "support-refund",
  score: "return-partial",
};
const tasks = { noul: "Noul (Yes/No)", choice: "Choice", score: "Score" };
function initialForm(task, id = defaults[task]) {
  const item = examples.find((e) => e.id === id);
  const criteria = item?.criteria || [];
  return {
    task,
    id,
    instruction: item?.instruction || "",
    context: Object.entries(item?.state || { text: "" }),
    yes: criteria.find((c) => c.id === "true")?.description || DEFAULT_YES,
    no: criteria.find((c) => c.id === "false")?.description || DEFAULT_NO,
    options:
      criteria.length && task !== "noul"
        ? criteria
            .map(
              (c) => `${task === "score" ? c.value : c.id} | ${c.description}`,
            )
            .join("\n")
        : task === "score"
          ? "0 | Not relevant\n1 | Partly relevant\n2 | Directly relevant"
          : "",
  };
}
function Result({ output }) {
  const { decision, result, inferenceMilliseconds, device = "cpu" } = output;
  const candidates = decision.criteria;
  const highlighted = candidates.reduce((best, candidate) => {
    if (decision.task === "score") {
      return Math.abs(candidate.value - result.score) < Math.abs(best.value - result.score)
        ? candidate : best;
    }
    return result.probabilities[candidate.id] > result.probabilities[best.id]
      ? candidate : best;
  });
  const min = Math.min(...candidates.map((c) => c.value)),
    max = Math.max(...candidates.map((c) => c.value));
  const choice = candidates.find((c) => c.id === result.selected_id);
  const title =
    decision.task === "noul"
      ? result.probability_yes >= 0.5
        ? "Yes"
        : "No"
      : decision.task === "choice"
        ? choice.id.replaceAll("_", " ")
        : result.score.toFixed(2);
  const note =
    decision.task === "noul"
      ? `${percent(result.probability_yes)} probability of Yes.`
      : decision.task === "choice"
        ? `${percent(result.probabilities[choice.id])} probability · ${choice.description}`
        : `On a ${min}–${max} scale. A probability-weighted average of the levels below.`;
  return (
    <>
      <div className="result-kicker">
        {decision.task === "score" ? "Estimated score" : decision.task === "choice" ? "Most likely option" : "Predicted answer"}
        <HelpTooltip id="result-help" label="Help with this result">
          {decision.task === "score"
            ? "The score averages your numeric levels using their probabilities. The highlighted level is closest to that score; ties use the first level in your input order. The bars show the probability for each level, so the longest bar may belong to a different level."
            : decision.task === "choice"
              ? "The heading shows the most likely option. Bars show probabilities across all options in your input order. Similar probabilities mean the model has no clear preference."
              : "Yes is shown when its probability is at least 50%; otherwise No is shown. Values near 50% indicate an uncertain decision."}
        </HelpTooltip>
      </div>
      <div className="result-value">{title}</div>
      <p className="result-note">{note}</p>
      {decision.task === "score" && (
        <>
          <div className="track score-track">
            <div
              className="fill"
              style={{ width: percent(result.normalized_score) }}
            />
          </div>
          <div className="scale-ends">
            <span>{min}</span>
            <span>{max}</span>
          </div>
        </>
      )}
      {candidates.map((c) => (
        <div
          key={c.id}
          className={`probability ${c.id === highlighted.id ? "winner" : ""}`}
        >
          <div className="probability-title">
            <span className="probability-name">
              {decision.task === "noul"
                ? ["true", "yes"].includes(c.id)
                  ? "Yes"
                  : "No"
                : decision.task === "score"
                  ? `${c.value} · ${c.description}`
                  : c.id.replaceAll("_", " ")}
            </span>
            <span className="probability-value">
              {percent(result.probabilities[c.id])}
            </span>
          </div>
          <div className="track">
            <div
              className="fill"
              style={{ width: percent(result.probabilities[c.id]) }}
            />
          </div>
        </div>
      ))}
      <p className="result-time">
        <Timer size={15} aria-hidden="true" />
        Inference: {inferenceMilliseconds.toFixed(1)} ms · {deviceLabel(device)}
        <HelpTooltip id="timing-help" label="Help with inference time">
          Time spent running the model across all candidates. Downloads, tokenization,
          and result formatting are excluded. First runs may take longer. WebGPU may use
          CPU assistance for unsupported operations.
        </HelpTooltip>
      </p>
    </>
  );
}
function App() {
  const resultPanel = useRef(null);
  const [editorOpen, setEditorOpen] = useState(false);
  const [modelId, setModelId] = useState(DEFAULT_MODEL_ID);
  const model = MODELS.find((item) => item.value === modelId);
  function switchModel(next) {
    worker.current?.terminate();
    worker.current = null;
    setModelId(next);
    setLoadedBytes(null);
    setFiles({});
    clear();
  }
  const [device, setDevice] = useState("cpu");
  const deviceChosen = useRef(false);
  const [gpu, setGpu] = useState({
    available: false,
    reason: "Checking WebGPU availability…",
  });
  useEffect(() => {
    let active = true;
    detectWebGPU().then((support) => {
      if (active) {
        setGpu(support);
        if (!deviceChosen.current) setDevice(support.available ? "webgpu" : "cpu");
      }
    });
    return () => {
      active = false;
    };
  }, []);
  function switchDevice(next) {
    deviceChosen.current = true;
    worker.current?.terminate();
    worker.current = null;
    setDevice(next);
    setLoadedBytes(null);
    setFiles({});
    clear();
  }
  const [form, setForm] = useState(() => initialForm("noul"));
  const [busy, setBusy] = useState(false),
    [status, setStatus] = useState(
      "Ready when you are. Your input stays in this browser.",
    );
  const [loadedBytes, setLoadedBytes] = useState(null),
    [files, setFiles] = useState({});
  const [output, setOutput] = useState(null),
    [error, setError] = useState(""),
    [request, setRequest] = useState(null);
  const worker = useRef(null),
    pending = useRef(null);
  useEffect(() => () => worker.current?.terminate(), []);
  function clear() {
    setOutput(null);
    setError("");
    setRequest(null);
    setStatus("Ready when you are. Your input stays in this browser.");
  }
  function change(key, value) {
    setForm((f) => ({ ...f, [key]: value }));
    clear();
  }
  function choose(task, id) {
    if (id === "custom") setEditorOpen(true);
    setForm(initialForm(task, id));
    clear();
  }
  function fail(message) {
    setBusy(false);
    setError(message);
    setStatus("Could not complete this request. Please try again.");
  }
  function getWorker() {
    if (worker.current) return worker.current;
    const current = new Worker(new URL("./worker.js", import.meta.url), {
      type: "module",
      credentials: "include",
    });
    current.onmessage = ({ data }) => {
      if (data.progress) {
        setFiles((previous) => ({
          ...previous,
          [data.progress.file]: data.progress,
        }));
        return;
      }
      if (data.loadedBytes !== undefined) {
        setLoadedBytes(data.loadedBytes);
        return;
      }
      if (data.status) {
        setStatus(data.status);
        return;
      }
      if (data.error) {
        // ORT keeps failed initialization state; retry with a fresh worker.
        current.terminate();
        worker.current = null;
        setLoadedBytes(null);
        setFiles({});
        fail(
          `${data.error}${device === "webgpu" ? " Try CPU if WebGPU cannot run this model on your device." : ""}`,
        );
        return;
      }
      setBusy(false);
      if (data.ready) {
        setStatus("Model loaded. Choose an example and click Run model.");
        return;
      }
      setOutput({ decision: pending.current, ...data });
      setStatus("Done. Change the context or try another example.");
    };
    current.onerror = (event) => {
      fail(event.message || "The inference worker stopped. Please try again.");
      current.terminate();
      worker.current = null;
      setLoadedBytes(null);
      setFiles({});
    };
    worker.current = current;
    return current;
  }
  function start(loadOnly = false) {
    deviceChosen.current = true;
    try {
      setError("");
      let decision;
      if (!loadOnly) {
        setOutput(null);
        decision = {
          task: form.task,
          instruction: form.instruction.trim(),
          state: contextObject(form.context),
          criteria: parseCriteria(form.task, form.options, form.yes, form.no),
        };
        renderDecision(decision);
        pending.current = decision;
        setRequest(decision);
      }
      if (loadedBytes === null) setFiles({});
      setBusy(true);
      setStatus(
        loadedBytes === null
          ? "Loading model…"
          : `Running on ${deviceLabel(device)}…`,
      );
      getWorker().postMessage({
        ...(loadOnly ? { type: "load" } : { decision }),
        device,
        base: model.base,
      });
    } catch (e) {
      fail(e.message);
    }
    if (!loadOnly && window.matchMedia("(max-width: 600px)").matches) {
      requestAnimationFrame(() => {
        resultPanel.current?.scrollIntoView({
          behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "instant" : "smooth",
          block: "start",
        });
      });
    }
  }
  const groups = [
    ...new Set(
      examples.filter((e) => e.task === form.task).map((e) => e.category),
    ),
  ];
  return (
    <main>
      <header className="hero">
        <div className="hero-copy">
          <div className="brand-wordmark">bekko-system-one</div>
          <h1>Ultra-small System One Model.<br /><span>Decisions in your browser.</span></h1>
          <p className="intro">
            Check a condition, choose an option, or score an answer. All on your device.
          </p>
        </div>
        <div className="hero-resources">
          <nav className="resource-links model-links" aria-label="Hugging Face models">
            {RESOURCE_LINKS.filter(link => link.kind === "model").map(({ label, href }) => (
              <a key={label} className="resource-badge" href={href} target="_blank" rel="noopener noreferrer">
                <span>{label}</span><ArrowUpRight size={14} aria-hidden="true" />
              </a>
            ))}
          </nav>
          <nav className="project-links" aria-label="Project resources">
            {RESOURCE_LINKS.filter(link => link.kind !== "model").map(({ label, href, placeholder }) => {
              const Link = placeholder ? "span" : "a";
              return (
              <Link className="project-link" key={label} href={placeholder ? undefined : href} target={placeholder ? undefined : "_blank"} rel={placeholder ? undefined : "noopener noreferrer"}>
                {label === "Technical article" ? <BookOpen size={15} aria-hidden="true" /> : <svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true" className="github-icon"><path d="M12 .75a11.25 11.25 0 0 0-3.558 21.923c.563.104.768-.244.768-.542 0-.267-.01-.975-.015-1.914-3.13.68-3.79-1.508-3.79-1.508-.512-1.3-1.25-1.646-1.25-1.646-1.022-.698.078-.684.078-.684 1.13.08 1.725 1.16 1.725 1.16 1.005 1.724 2.636 1.226 3.278.938.102-.729.393-1.226.715-1.508-2.498-.284-5.124-1.249-5.124-5.562 0-1.229.439-2.233 1.159-3.02-.116-.285-.503-1.43.11-2.98 0 0 .945-.303 3.094 1.154a10.79 10.79 0 0 1 5.63 0c2.148-1.457 3.09-1.153 3.09-1.153.615 1.549.228 2.694.112 2.979.722.787 1.158 1.791 1.158 3.02 0 4.324-2.63 5.275-5.136 5.553.404.348.766 1.034.766 2.084 0 1.504-.014 2.718-.014 3.087 0 .3.203.65.774.54A11.251 11.251 0 0 0 12 .75Z" /></svg>}
                <span>{label}</span>{placeholder && <span className="coming-soon">Coming soon</span>}
              </Link>
              );
            })}
          </nav>
        </div>
      </header>
      <details className="model-info">
        <summary>
          <Info size={17} aria-hidden="true" />
          <span>Small models, limited generalization</span>
          <ChevronDown size={16} className="model-info-chevron" aria-hidden="true" />
        </summary>
        <div className="model-info-content">
          <section className="model-info-section">
            <h3>What works well</h3>
            <p>bekko-system-one-v0 was trained on 100+ task-specific dataset subsets covering classification, selection, and scoring. It often works best when your question and context resemble those tasks.</p>
            <p>The examples here were deliberately selected to show cases it handles well; they are not a measure of general-purpose accuracy.</p>
          </section>
          <section className="model-info-section">
            <h3>Where it falls short</h3>
            <p>Training across many datasets does not automatically teach a model to handle unfamiliar instructions or domains. bekko-system-one-v0 still struggles with that transfer, and even simple questions can produce confidently wrong answers.</p>
            <p>These models are intended for English input.</p>
          </section>
          <section className="model-info-outlook">
            <h3>Why explore it?</h3>
            <p>bekko-system-one-v0 is not a replacement for Jev’s broad generalization. It demonstrates how small models can make useful, structured decisions directly in your browser—a starting point for exploring more capable on-device AI.</p>
          </section>
        </div>
      </details>
      <section className="model-panel" aria-label="Model">
        <div className="model-selection">
          <Picker
            id="model"
            label="Model"
            help={`${model.name} runs on your device. Larger models use more memory and take longer to load; they are not always more accurate. Switching models starts a new session.`}
            value={modelId}
            searchable={false}
            disabled={busy}
            options={MODELS}
            onChange={switchModel}
          />

        </div>
        <div className="runtime-selection">
          <fieldset
            className="runtime-radios"
            aria-describedby="runtime-help"
          >
            <legend>
              <span className="runtime-heading">
                Inference device
                <HelpTooltip id="runtime-help" label="Help with inference devices">
                  {gpu.reason} WebGPU is selected automatically when available.
                  Unsupported operations may run on CPU. Switching devices reloads the model.
                </HelpTooltip>
              </span>
            </legend>
            <div className="runtime-options">
              {[
                ["cpu", "CPU"],
                ["webgpu", "WebGPU"],
              ].map(([value, label]) => (
                <label
                  key={value}
                  className={`runtime-option ${device === value ? "selected" : ""}`}
                >
                  <input
                    type="radio"
                    name="runtime"
                    value={value}
                    checked={device === value}
                    onClick={() => { deviceChosen.current = true; }}
                    disabled={busy || (value === "webgpu" && !gpu.available)}
                    onChange={() => switchDevice(value)}
                  />
                  <span className="device-label">{value === "cpu" ? <Cpu size={16} aria-hidden="true" /> : <CircuitBoard size={16} aria-hidden="true" />}{label}</span>
                </label>
              ))}
            </div>
          </fieldset>

        </div>
        {loadedBytes === null ? (
          <Button
            id="load-model"
            secondary
            disabled={busy}
            onClick={() => start(true)}
          >
            {busy ? <LoaderCircle size={17} className="spin" aria-hidden="true" /> : <Download size={17} aria-hidden="true" />}
            {busy
              ? "Loading…"
              : error
                ? "Retry loading"
                : "Load model"}
          </Button>
        ) : (
          <span className="loaded-badge"><CircleCheck size={17} aria-hidden="true" /> Model ready</span>
        )}
          <p
            id="model-status"
            role="status"
            className={loadedBytes !== null ? "loaded" : ""}
          >
            {loadedBytes !== null
              ? `Model loaded · ${(loadedBytes / 1e6).toFixed(1)} MB · ${deviceLabel(device)}`
              : busy
                ? status
                : "Not loaded yet. Your first run loads the model automatically."}
          </p>
          {model.bytes >= 1e9 && (
            <div className="large-model-notice">
              <HardDriveDownload size={18} aria-hidden="true" />
              <div>
                <strong>Large model · 1.4 GB download to your browser</strong>
                <p>{loadedBytes === null
                  ? "About 1.4 GB will be downloaded to your browser and run on your device. This model also needs substantial memory. Choose 17m or 68m for a smaller download."
                  : "This model runs in your browser and uses substantial memory. Switching models or reloading the page may download another 1.4 GB to your browser."}</p>
              </div>
            </div>
          )}
        {busy && loadedBytes === null && (
          <ModelLoadProgress files={files} status={status} expectedBytes={model.bytes} />
        )}
      </section>
      <div className="workspace">
        <form
          id="form"
          className={`panel ${editorOpen ? "editor-open" : "editor-closed"}`}
          onInvalidCapture={(event) => {
            if (!editorOpen && window.matchMedia("(max-width: 600px)").matches) {
              event.preventDefault();
              setEditorOpen(true);
              const input = event.target;
              requestAnimationFrame(() => { input.focus(); input.reportValidity(); });
            }
          }}
          onSubmit={(event) => {
            event.preventDefault();
            start();
          }}
        >
          <fieldset id="inputs" disabled={busy}>
            <div className="section-title">
              <span className="step"><SlidersHorizontal size={18} aria-hidden="true" /></span>
              <h2>Try a decision</h2>
            </div>
            <div className="decision-pickers">
              <fieldset id="task" className="task-radios" aria-describedby="task-help">
                <legend>
                  <span className="runtime-heading">
                    1. Decision type
                    <HelpTooltip id="task-help" label="Help with decision types">
                      {{
                  noul: "Noul checks a condition and returns Yes/No probabilities. Use the Yes and No meanings to define what each answer represents.",
                  choice: "Choice selects one of your options. Write distinct alternatives and describe them if their names alone are unclear.",
                  score: "Score rates the context against numeric levels. The result is a probability-weighted average, so it can fall between levels.",
                      }[form.task]}
                    </HelpTooltip>
                  </span>
                </legend>
                <div className="task-options">
                  {Object.entries(tasks).map(([value, label]) => (
                    <label key={value} className={`task-option ${form.task === value ? "selected" : ""}`}>
                      <input type="radio" name="task" value={value} checked={form.task === value}
                        disabled={busy} onChange={() => choose(value)} />
                      <span>{label}</span>
                    </label>
                  ))}
                </div>
              </fieldset>
              <Picker
                id="example"
                label="2. Choose an example"
                value={form.id}
                disabled={busy}
                options={[
                  ...groups.map((group) => ({
                    label: group,
                    options: examples
                      .filter(
                        (e) => e.task === form.task && e.category === group,
                      )
                      .map((e) => ({ value: e.id, label: e.title })),
                  })),
                  { value: "custom", label: "Write your own…" },
                ]}
                onChange={(id) => choose(form.task, id)}
              />
            </div>
            <Button id="run-example" type="submit" disabled={busy}>
              {busy ? <LoaderCircle size={17} className="spin" aria-hidden="true" /> : <Play size={17} aria-hidden="true" />}
              {busy ? "Working…" : "Run model"}
            </Button>
            <button
              type="button"
              className="editor-toggle"
              aria-expanded={editorOpen}
              aria-controls="decision-editor"
              onClick={() => setEditorOpen(open => !open)}
            >
              <SlidersHorizontal size={16} aria-hidden="true" />
              <span>{editorOpen ? "Hide input" : "View or edit input"}</span>
              <ChevronDown size={16} aria-hidden="true" />
            </button>
            <div id="decision-editor">
              <div className="form-toolbar">
                <p className="hint">Edit any field to try your own input.</p>
                <Button
                  secondary
                  type="button"
                  onClick={() => choose(form.task, form.id)}
                >
                  <RotateCcw size={14} aria-hidden="true" /> Reset example
                </Button>
              </div>
              <Textarea
                id="instruction"
                label={form.task === "noul" ? "Question to answer" : form.task === "choice" ? "Selection instructions" : "Scoring instructions"}
                rows={2}
                required
                value={form.instruction}
                onChange={(e) => change("instruction", e.target.value)}
              />
              <div className="label-row">
                <div className="context-heading">
                  <h3>Context</h3>
                  <HelpTooltip id="context-help" label="Help with context fields">
                    Each field name describes the text below it. Both are sent to the model.
                    Give each field a unique, non-empty name.
                  </HelpTooltip>
                </div>
                <span className="hint">Information for the model</span>
              </div>
              <div id="state-fields">
                {form.context.map(([key, value], index) => {
                  const keyError = contextKeyError(form.context, index);
                  return (
                    <div className="context-field" key={index}>
                      <div className="context-key-row">
                        <label htmlFor={`state-key-${index}`}>Field name</label>
                        <input
                          id={`state-key-${index}`}
                          aria-label={`Context key ${index + 1}`}
                          aria-invalid={!!keyError}
                          aria-describedby={keyError ? `state-key-error-${index}` : "context-help"}
                          ref={(input) => input?.setCustomValidity(keyError)}
                          value={key}
                          required
                          spellCheck={false}
                          autoComplete="off"
                          onChange={(e) => change("context", form.context.map((entry, i) =>
                            i === index ? [e.target.value, entry[1]] : entry
                          ))}
                        />
                      </div>
                      {keyError && <p className="context-error" id={`state-key-error-${index}`} role="alert">{keyError}</p>}
                      <Textarea
                        id={`state-${index}`}
                        data-key={key}
                        aria-label={`Context value ${index + 1}${key ? ` (${key})` : ""}`}
                        maxRows={10}
                        value={value}
                        onChange={(e) => change("context", form.context.map((entry, i) =>
                          i === index ? [entry[0], e.target.value] : entry
                        ))}
                      />
                    </div>
                  );
                })}
              </div>
              <Button
                id="add-context"
                secondary
                type="button"
                onClick={() => {
                  let number = form.context.length + 1;
                  while (form.context.some(([key]) => key === `field_${number}`)) number++;
                  const index = form.context.length;
                  change("context", [...form.context, [`field_${number}`, ""]]);
                  requestAnimationFrame(() => {
                    const input = document.getElementById(`state-key-${index}`);
                    input?.focus();
                    input?.select();
                  });
                }}
              >
                <Plus size={15} aria-hidden="true" /> Add context field
              </Button>
              {form.task === "noul" ? (
                <div id="binary-fields">
                  <div className="binary-meanings">
                    <div className="context-field meaning-field">
                      <Textarea
                        id="yes"
                        label="Meaning of “Yes”"
                        help="Describe the condition that counts as Yes. Define the opposite under Meaning of “No”; these descriptions are included in the model input."
                        rows={2}
                        required
                        value={form.yes}
                        onChange={(e) => change("yes", e.target.value)}
                      />
                    </div>
                    <div className="context-field meaning-field">
                      <Textarea
                        id="no"
                        label="Meaning of “No”"
                        rows={2}
                        required
                        value={form.no}
                        onChange={(e) => change("no", e.target.value)}
                      />
                    </div>
                  </div>
                </div>
              ) : (
                <div id="option-fields">
                  <Textarea
                    id="options"
                    label={form.task === "score" ? "Scoring scale" : "Options to choose from"}
                    rows={7}
                    required
                    spellCheck={false}
                    value={form.options}
                    onChange={(e) => change("options", e.target.value)}
                    help={
                      form.task === "score"
                        ? "One level per line: number | meaning, such as 0 | Not relevant. Use distinct numbers and describe each level. Results keep your input order; scores can fall between levels."
                        : "One option per line: a unique name, or name | description. For example, refund | The customer wants their money back. Results keep your input order."
                    }
                  />
                </div>
              )}
            </div>
          </fieldset>
          <Button id="run" type="submit" disabled={busy}>
            {busy ? (
              <>
                <LoaderCircle size={17} className="spin" aria-hidden="true" /> Working…
              </>
            ) : (
              <>
                <Play size={17} aria-hidden="true" /> Run model
              </>
            )}
          </Button>
          <p id="status" className="hint" role="status">
            {status}
          </p>
        </form>
        <aside ref={resultPanel} className="panel result-panel">
          <div className="section-title">
            <span className="step"><ChartNoAxesColumnIncreasing size={18} aria-hidden="true" /></span>
            <h2>Your result</h2>
          </div>
          <div id="result" aria-live="polite" aria-busy={busy}>
            {error ? (
              <p role="alert" className="error">
                {error}
              </p>
            ) : output ? (
              <Result output={output} />
            ) : busy ? (
              <DecisionActivity loading={loadedBytes === null} device={deviceLabel(device)} files={files} status={status} expectedBytes={model.bytes} />
            ) : (
              <div className="empty-result">
                <span className="empty-icon"><ChartNoAxesColumnIncreasing size={27} aria-hidden="true" /></span>
                <h3>Your result will appear here</h3>
                <p>Choose an example, then select Run model to see its prediction and probabilities.</p>
              </div>
            )}
          </div>
          <details className="developer">
            <summary><Braces size={15} aria-hidden="true" /> Input JSON</summary>
            <pre id="request-json">
              {request
                ? pretty(request)
                : "Run a decision to inspect its input."}
            </pre>
            <details>
              <summary>Rendered model input</summary>
              <pre id="model-input-json">
                {request ? pretty(renderDecision(request)) : ""}
              </pre>
            </details>
          </details>
          <details className="developer">
            <summary><Braces size={15} aria-hidden="true" /> Output JSON</summary>
            <pre id="response-json">
              {error
                ? pretty({ error })
                : output
                  ? pretty(output.result)
                  : "Run a decision to inspect its output."}
            </pre>
          </details>
        </aside>
      </div>
      <footer>
        <p>Inference runs in your browser. No inference server.</p>

      </footer>
    </main>
  );
}
createRoot(document.getElementById("root")).render(<App />);
