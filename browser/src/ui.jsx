import React, { useLayoutEffect, useRef, useState } from "react";
import Select from "react-select";

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
export function ModelLoadProgress({ files, status }) {
  const list = Object.entries(files);
  const loaded = list.reduce((sum, [, file]) => sum + file.loaded, 0);
  const model = list.find(([name]) => name.endsWith(".onnx"))?.[1];
  const preparing = status.startsWith("Preparing");
  const value = !preparing && model?.total ? Math.min(100, model.loaded / model.total * 100) : null;
  return (
    <div className="load-progress" aria-label="Model loading progress">
      <div className="progress-heading">
        <span>{preparing ? "Preparing model…" : "Downloading model…"}</span>
        <span className="progress-amount">
          {preparing ? "Almost ready" : `${(loaded / 1e6).toFixed(1)} MB received`}
          {value != null && ` · ${Math.floor(value)}%`}
        </span>
      </div>
      <Progress label={preparing ? "Preparing inference runtime" : "Model download"} value={value} />
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

export function HelpTooltip({ id, label, children }) {
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
      className="help-tooltip"
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
        className="help-trigger"
        aria-label={label}
        aria-describedby={id}
        onFocus={() => setOpen(true)}
        onClick={() => setOpen(true)}
      >
        <svg viewBox="0 0 24 24" width="17" height="17" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" aria-hidden="true">
          <circle cx="12" cy="12" r="9" />
          <path d="M12 11v6" />
          <circle cx="12" cy="7.5" r=".9" fill="currentColor" stroke="none" />
        </svg>
      </button>
      <span ref={bubble} id={id} role="tooltip" className="help-content" style={position} hidden={!open}>
        {children}
      </span>
    </span>
  );
}
