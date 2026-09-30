import React, { useEffect, useLayoutEffect, useRef, useState } from "react";
import Select from "react-select";
import { Info, Download, LoaderCircle, Cpu, Timer } from "lucide-react";

export function Button({ secondary = false, className = "", ...props }) {
  return (
    <button
      className={`${secondary ? "secondary" : "primary"} ${className}`}
      {...props}
    />
  );
}
export function Textarea({ id, label, help, maxRows, ...props }) {
  const textarea = useRef(null);
  useLayoutEffect(() => {
    if (!maxRows) return;
    const element = textarea.current;
    const resize = () => {
      const style = getComputedStyle(element);
      const line = parseFloat(style.lineHeight);
      const padding = parseFloat(style.paddingTop) + parseFloat(style.paddingBottom);
      const border = parseFloat(style.borderTopWidth) + parseFloat(style.borderBottomWidth);
      const maximum = line * maxRows + padding + border;
      element.style.height = "auto";
      const content = element.scrollHeight + border;
      element.style.height = `${Math.min(maximum, Math.max(line + padding + border, content))}px`;
      element.style.overflowY = content > maximum ? "auto" : "hidden";
    };
    resize();
    let width = element.getBoundingClientRect().width;
    const observer = new ResizeObserver(() => {
      const nextWidth = element.getBoundingClientRect().width;
      if (nextWidth !== width) {
        width = nextWidth;
        resize();
      }
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, [maxRows, props.value]);
  return (
    <div className="field">
      {label && <FieldLabel id={id} label={label} help={help} />}
      <textarea
        ref={textarea}
        rows={maxRows ? 1 : undefined}
        style={maxRows ? { resize: "none" } : undefined}
        id={id}
        aria-describedby={help ? `${id}-help` : undefined}
        {...props}
      />

    </div>
  );
}
export function Progress({ value, label }) {
  return (
    <progress
      aria-label={label}
      max="100"
      {...(value == null ? {} : { value })}
    />
  );
}
export function ModelLoadProgress({ files, status, expectedBytes }) {
  const model = Object.entries(files).find(([name]) => name.endsWith(".onnx"))?.[1];
  const preparing = status.startsWith("Preparing");
  const total = model?.total || expectedBytes;
  // Show model bytes only: tokenizer downloads are separate from this total.
  const loaded = model?.loaded || 0;
  const value = !preparing && total ? Math.min(100, loaded / total * 100) : null;
  const mb = bytes => (bytes / 1e6).toLocaleString("en-US", { minimumFractionDigits: 1, maximumFractionDigits: 1 });
  return (
    <div className="load-progress" aria-label="Model loading progress">
      <div className="progress-heading">
        <span className="progress-stage">
          {preparing ? <LoaderCircle className="spin" size={16} aria-hidden="true" /> : <Download size={16} aria-hidden="true" />}
          {preparing ? "Preparing model…" : "Downloading model…"}
        </span>
        <span className="progress-amount">
          {mb(loaded)} MB{total ? ` / ${mb(total)} MB` : " received"}
        </span>
      </div>
      <Progress label={preparing ? "Preparing inference runtime" : "Model download"} value={value} />
      <div className="progress-caption">
        <span>{preparing ? "Setting up inference on your device" : "Loading into your browser"}</span>
        <span>{preparing ? "Initializing" : value == null ? "Receiving data" : `${Math.floor(value)}%`}</span>
      </div>
    </div>
  );
}

export function Picker({
  help,
  id,
  label,
  options,
  value,
  onChange,
  disabled,
  searchable = true,
}) {
  const flat = options.flatMap((option) => option.options || [option]);
  return (
    <div className="picker">
      <FieldLabel id={id} label={label} help={help} />
      <Select
        inputId={id}
        instanceId={id}
        classNamePrefix="picker"
        options={options}
        value={flat.find((option) => option.value === value)}
        onChange={(option) => option && onChange(option.value)}
        isDisabled={disabled}
        isSearchable={searchable}
        isOptionDisabled={(option) => !!option.disabled}
        placeholder="Search examples…"
        noOptionsMessage={() => "No matching examples"}
        theme={(theme) => ({
          ...theme,
          colors: {
            ...theme.colors,
            primary: "#235d40",
            primary25: "#eaf0e4",
            primary50: "#d5e4d3",
          },
        })}
      />
    </div>
  );
}

export function FieldLabel({ id, label, help }) {
  return (
    <div className="field-label">
      <label htmlFor={id}>{label}</label>
      {help && <HelpTooltip id={`${id}-help`} label={`Help: ${label}`}>{help}</HelpTooltip>}
    </div>
  );
}

export function HelpTooltip({ id, label, children, trigger, className = "" }) {
  const [open, setOpen] = useState(false);
  const anchor = useRef(null);
  const bubble = useRef(null);
  const [position, setPosition] = useState({});
  useLayoutEffect(() => {
    if (!open) return;
    const place = () => {
      const rect = anchor.current.getBoundingClientRect();
      const box = bubble.current.getBoundingClientRect();
      setPosition({
        left: Math.max(12, Math.min(rect.left, innerWidth - box.width - 12)),
        top: rect.bottom + box.height + 8 < innerHeight
          ? rect.bottom + 6
          : Math.max(12, rect.top - box.height - 6),
      });
    };
    place();
    window.addEventListener("resize", place);
    window.addEventListener("scroll", place, true);
    return () => {
      window.removeEventListener("resize", place);
      window.removeEventListener("scroll", place, true);
    };
  }, [open, children]);
  return (
    <span
      className={`help-tooltip ${className}`}
      ref={anchor}
      onPointerEnter={(event) => {
        if (event.pointerType === "mouse") setOpen(true);
      }}
      onPointerLeave={() => setOpen(false)}
      onBlur={() => setOpen(false)}
      onKeyDown={(event) => {
        if (event.key === "Escape") {
          setOpen(false);
          event.stopPropagation();
        }
      }}
    >
      <button
        type="button"
        className={trigger ? "score-help-trigger" : "help-trigger"}
        aria-label={label}
        aria-describedby={id}
        onFocus={() => setOpen(true)}
        onClick={() => setOpen(true)}
      >
        {trigger ?? <Info size={16} strokeWidth={1.8} aria-hidden="true" />}
      </button>
      <span ref={bubble} id={id} role="tooltip" className="help-content" style={position} hidden={!open}>
        {children}
      </span>
    </span>
  );
}

export function DecisionActivity({ loading, device, files, status, expectedBytes }) {
  const [elapsed, setElapsed] = useState(0);
  useEffect(() => {
    const started = performance.now();
    setElapsed(0);
    const timer = setInterval(() => setElapsed((performance.now() - started) / 1000), 250);
    return () => clearInterval(timer);
  }, [loading]);
  return (
    <div className="empty-result decision-activity">
      <div className="activity-visual" aria-hidden="true">
        <span className="activity-halo" />
        <span className="activity-core">{loading ? <Download size={24} /> : <Cpu size={24} />}</span>
        <div className="activity-wave">
          {[18, 28, 22, 38, 30, 42, 24, 34, 18].map((height, i) => (
            <span key={i} style={{ height, animationDelay: `${i * -0.16}s` }} />
          ))}
        </div>
      </div>
      <h3>{loading ? "Getting your model ready" : "Analyzing your input"}<span className="activity-dots" aria-hidden="true"><span>.</span><span>.</span><span>.</span></span></h3>
      <p>{loading
        ? "Downloading the model to your browser to run on your device."
        : "Comparing your options with the information you provided."}</p>
      <div className="activity-meta">
        <span>{loading ? "Loading model" : `Running locally · ${device}`}</span>
        <span className="activity-elapsed" aria-live="off"><Timer size={13} aria-hidden="true" />{elapsed.toFixed(1)} s elapsed</span>
      </div>
      {loading
        ? <ModelLoadProgress files={files} status={status} expectedBytes={expectedBytes} />
        : <div className="activity-sweep" aria-hidden="true"><span /></div>}
      <p className="activity-reassurance">{!loading && elapsed >= 10
        ? "Larger models and longer inputs can take more time. You can keep this tab open."
        : "Your input stays on this device."}</p>
    </div>
  );
}
