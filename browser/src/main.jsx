import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import "./style.css";
import examples from "./examples.json";
import { DEFAULT_MODEL_PATH } from "./default-model.js";
import {
  DEFAULT_YES,
  DEFAULT_NO,
  parseCriteria,
  renderDecision,
} from "./decision.js";
import { Button, Textarea, ModelLoadProgress } from "./ui.jsx";

const pretty = (value) => JSON.stringify(value, null, 2);
const percent = (value) => `${(value * 100).toFixed(1)}%`;
const defaults = {
  noul: "train-route-same",
  choice: "news-tennis-championship",
  score: "photosynthesis-complete",
};
const tasks = { noul: "Yes / No", choice: "Choice", score: "Score" };
function initialForm(task, id = defaults[task]) {
  const item = examples.find((e) => e.id === id);
  const criteria = item?.criteria || [];
  return {
    task,
    id,
    instruction: item?.instruction || "",
    state: item?.state || { text: "" },
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
  const { decision, result, milliseconds } = output;
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
        Completed in {milliseconds.toFixed(0)} ms · On-device CPU
      </p>
    </>
  );
}
function App() {
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
        fail(data.error);
        return;
      }
      setBusy(false);
      if (data.ready) {
        setStatus("Model ready. Run any example to get started.");
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
          state: form.state,
          criteria: parseCriteria(form.task, form.options, form.yes, form.no),
        };
        renderDecision(decision);
        pending.current = decision;
        setRequest(decision);
      }
      if (loadedBytes === null) setFiles({});
      setBusy(true);
      setStatus(loadedBytes === null ? "Loading model…" : "Running on CPU…");
      getWorker().postMessage({
        ...(loadOnly ? { type: "load" } : { decision }),
        base: new URL(
          `${import.meta.env.BASE_URL}${DEFAULT_MODEL_PATH}`,
          document.baseURI,
        ).href,
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
          <h1>Local decisions</h1>
          <p className="intro">
            Check a condition, choose an option, or score an answer.
          </p>
        </div>
        <span className="badge">17M · CPU · Private inputs</span>
      </header>
      <section className="model-panel" aria-label="Model">
        <div>
          <strong>Bekko 17M</strong>
          <p id="model-status" role="status">
            {loadedBytes !== null
              ? `Model loaded · ${(loadedBytes / 1e6).toFixed(1)} MB · Ready on your device`
              : "Model not loaded · 29 MB model · loads on your first run"}
          </p>
        </div>
        <Button
          id="load-model"
          secondary
          disabled={busy || loadedBytes !== null}
          onClick={() => start(true)}
        >
          {loadedBytes !== null
            ? "Model ready"
            : error
              ? "Retry loading"
              : "Load model"}
        </Button>
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
            <label htmlFor="task">Decision type</label>
            <select
              id="task"
              value={form.task}
              onChange={(event) => choose(event.target.value)}
            >
              {Object.entries(tasks).map(([id, name]) => (
                <option key={id} value={id}>
                  {name}
                </option>
              ))}
            </select>
            <label htmlFor="example">Try an example</label>
            <select
              id="example"
              value={form.id}
              onChange={(event) => choose(form.task, event.target.value)}
            >
              {groups.map((group) => (
                <optgroup key={group} label={group}>
                  {examples
                    .filter((e) => e.task === form.task && e.category === group)
                    .map((e) => (
                      <option key={e.id} value={e.id}>
                        {e.title}
                      </option>
                    ))}
                </optgroup>
              ))}
              <option value="custom">Write your own…</option>
            </select>
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
              rows={3}
              required
              value={form.instruction}
              onChange={(e) => change("instruction", e.target.value)}
            />
            <div className="label-row">
              <h3>Context</h3>
              <span className="hint">The information to judge</span>
            </div>
            <div id="state-fields">
              {Object.entries(form.state).map(([key, value], index) => (
                <Textarea
                  key={key}
                  id={`state-${index}`}
                  data-key={key}
                  label={key
                    .replaceAll("_", " ")
                    .replace(/^./, (c) => c.toUpperCase())}
                  rows={String(value).length > 180 ? 5 : 3}
                  value={value}
                  onChange={(e) =>
                    change("state", { ...form.state, [key]: e.target.value })
                  }
                />
              ))}
            </div>
            {form.task === "noul" ? (
              <div id="binary-fields">
                <details id="binary-details" key={form.id}>
                  <summary>Customize what Yes and No mean</summary>
                  <Textarea
                    id="yes"
                    label="Yes means"
                    rows={2}
                    required
                    value={form.yes}
                    onChange={(e) => change("yes", e.target.value)}
                  />
                  <Textarea
                    id="no"
                    label="No means"
                    rows={2}
                    required
                    value={form.no}
                    onChange={(e) => change("no", e.target.value)}
                  />
                </details>
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
        Bekko 17M · INT8 embeddings · CPU inference · No inference server
      </footer>
    </main>
  );
}
createRoot(document.getElementById("root")).render(<App />);
