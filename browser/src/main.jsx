import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import "./style.css";
import examples from "./examples.json";
import { MODELS, DEFAULT_MODEL_ID } from "./models.js";
import {
  DEFAULT_YES,
  DEFAULT_NO,
  parseCriteria,
  renderDecision,
} from "./decision.js";
import { Button, Textarea, ModelLoadProgress, Picker, HelpTooltip } from "./ui.jsx";

import { contextKeyError, contextObject } from "./context.js";

import { detectWebGPU, deviceLabel } from "./runtime.js";

const pretty = (value) => JSON.stringify(value, null, 2);
const percent = (value) => `${(value * 100).toFixed(1)}%`;
const defaults = {
  noul: "email-bulk-offer",
  choice: "support-password",
  score: "return-help-complete",
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
  const sorted = [...candidates].sort(
    (a, b) => result.probabilities[b.id] - result.probabilities[a.id],
  );
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
  const display =
    decision.task === "choice"
      ? sorted
      : decision.task === "score"
        ? [...candidates].sort((a, b) => a.value - b.value)
        : candidates;
  return (
    <>
      <div className="result-kicker">
        {decision.task === "score" ? "Model score" : "Model prediction"}
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
      {display.map((c) => (
        <div
          key={c.id}
          className={`probability ${c.id === sorted[0].id ? "winner" : ""}`}
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
        Inference: {inferenceMilliseconds.toFixed(1)} ms · {deviceLabel(device)}
        {device === "webgpu"
          ? " (CPU fallback for unsupported operations)"
          : " on your device"}
      </p>
    </>
  );
}
function App() {
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
  const [gpu, setGpu] = useState({
    available: false,
    reason: "Checking WebGPU availability…",
  });
  useEffect(() => {
    let active = true;
    detectWebGPU().then((support) => {
      if (active) setGpu(support);
    });
    return () => {
      active = false;
    };
  }, []);
  function switchDevice(next) {
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
        fail(
          `${data.error}${device === "webgpu" ? " Try CPU if WebGPU cannot run this model on your device." : ""}`,
        );
        return;
      }
      setBusy(false);
      if (data.ready) {
        setStatus("Model loaded. Choose an example and click Run decision.");
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
  }
  const groups = [
    ...new Set(
      examples.filter((e) => e.task === form.task).map((e) => e.category),
    ),
  ];
  return (
    <main>
      <header>
        <div>
          <div className="eyebrow">
            <span className="dot" /> BEKKO · ON-DEVICE AI
          </div>
          <h1>Run an ultra-small System One model in your browser</h1>
          <p className="intro">
            Load a model that runs directly in your browser. Check a condition,
            choose an option, or score an answer.
          </p>
        </div>
      </header>
      <section className="model-panel" aria-label="Model">
        <div className="model-selection">
          <Picker
            id="model"
            label="Model"
            value={modelId}
            searchable={false}
            disabled={busy}
            options={MODELS}
            onChange={switchModel}
          />
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
        </div>
        <div className="runtime-selection">
          <fieldset
            className="runtime-radios"
            disabled={busy}
            aria-describedby="runtime-help"
          >
            <legend>Inference device</legend>
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
                    disabled={value === "webgpu" && !gpu.available}
                    onChange={() => switchDevice(value)}
                  />
                  <span>{label}</span>
                </label>
              ))}
            </div>
          </fieldset>
          <p id="runtime-help" className="hint">
            {gpu.reason}
          </p>
          {device === "webgpu" && (
            <p className="hint">
              Unsupported operations run on CPU. Switching devices reloads the
              model.
            </p>
          )}
        </div>
        {loadedBytes === null ? (
          <Button
            id="load-model"
            secondary
            disabled={busy}
            onClick={() => start(true)}
          >
            {busy
              ? "Loading…"
              : error
                ? "Retry loading"
                : "Load model"}
          </Button>
        ) : (
          <span className="loaded-badge">✓ Loaded · Ready to run</span>
        )}
        {busy && loadedBytes === null && (
          <ModelLoadProgress files={files} status={status} />
        )}
      </section>
      <div className="workspace">
        <form
          id="form"
          className="panel"
          onSubmit={(event) => {
            event.preventDefault();
            start();
          }}
        >
          <fieldset id="inputs" disabled={busy}>
            <div className="section-title">
              <span className="step">1</span>
              <h2>Set up a decision</h2>
            </div>
            <div className="decision-pickers">
              <Picker
                id="task"
                label="1. Decision type"
                value={form.task}
                disabled={busy}
                searchable={false}
                options={Object.entries(tasks).map(([value, label]) => ({
                  value,
                  label,
                }))}
                onChange={(task) => choose(task)}
              />
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
              {busy ? "Working…" : "Run decision →"}
            </Button>
            <div className="form-toolbar">
              <p className="hint">Edit any field to try your own input.</p>
              <Button
                secondary
                type="button"
                onClick={() => choose(form.task, form.id)}
              >
                Reset example
              </Button>
            </div>
            <Textarea
              id="instruction"
              label="Question or instruction"
              rows={2}
              required
              value={form.instruction}
              onChange={(e) => change("instruction", e.target.value)}
            />
            <div className="label-row">
              <div className="context-heading">
                <h3>Context</h3>
                <HelpTooltip id="context-help" label="Help with context fields">
                  Each key names the value below it. Both are sent to the model.
                  Use a unique, non-empty key.
                </HelpTooltip>
              </div>
              <span className="hint">The information to judge</span>
            </div>
            <div id="state-fields">
              {form.context.map(([key, value], index) => {
                const keyError = contextKeyError(form.context, index);
                return (
                  <div className="context-field" key={index}>
                    <div className="context-key-row">
                      <label htmlFor={`state-key-${index}`}>Key</label>
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
                      rows={String(value).length > 180 ? 4 : 2}
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
              + Add context field
            </Button>
            {form.task === "noul" ? (
              <div id="binary-fields">
                <div className="binary-meanings">
                  <div className="context-field meaning-field">
                    <Textarea
                      id="yes"
                      label="Yes means"
                      rows={2}
                      required
                      value={form.yes}
                      onChange={(e) => change("yes", e.target.value)}
                    />
                  </div>
                  <div className="context-field meaning-field">
                    <Textarea
                      id="no"
                      label="No means"
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
                  label={form.task === "score" ? "Score levels" : "Options"}
                  rows={7}
                  required
                  spellCheck={false}
                  value={form.options}
                  onChange={(e) => change("options", e.target.value)}
                  help={
                    form.task === "score"
                      ? "One level per line: number | meaning. Scores may fall between levels."
                      : "One option per line: a name, or name | description."
                  }
                />
              </div>
            )}
          </fieldset>
          <Button id="run" type="submit" disabled={busy}>
            {busy ? (
              <>
                <span className="spinner" aria-hidden="true" /> Working…
              </>
            ) : (
              <>
                Run decision <span aria-hidden="true">→</span>
              </>
            )}
          </Button>
          <p id="status" className="hint" role="status">
            {status}
          </p>
        </form>
        <aside className="panel result-panel">
          <div className="section-title">
            <span className="step">2</span>
            <h2>Your result</h2>
          </div>
          <div id="result" aria-live="polite" aria-busy={busy}>
            {error ? (
              <p role="alert" className="error">
                {error}
              </p>
            ) : output ? (
              <Result output={output} />
            ) : (
              <div className="empty-result">
                <h3>
                  {busy
                    ? "Working on your decision…"
                    : "Ready for your first decision"}
                </h3>
                <p>
                  {busy
                    ? "Inference runs locally on your device."
                    : "Choose an example and run it to see the answer and probabilities."}
                </p>
              </div>
            )}
          </div>
          <details className="developer">
            <summary>Request JSON</summary>
            <pre id="request-json">
              {request
                ? pretty(request)
                : "Run a decision to inspect its request."}
            </pre>
            <details>
              <summary>Rendered model input</summary>
              <pre id="model-input-json">
                {request ? pretty(renderDecision(request)) : ""}
              </pre>
            </details>
          </details>
          <details className="developer">
            <summary>Response JSON</summary>
            <pre id="response-json">
              {error
                ? pretty({ error })
                : output
                  ? pretty(output.result)
                  : "Run a decision to inspect its response."}
            </pre>
          </details>
        </aside>
      </div>
      <footer>
        {model.name} · INT8 embeddings · {deviceLabel(device)} inference · No
        inference server
      </footer>
    </main>
  );
}
createRoot(document.getElementById("root")).render(<App />);
