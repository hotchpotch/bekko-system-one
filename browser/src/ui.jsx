import React, { useState } from "react";
import Select from "react-select";

export function Button({ secondary = false, className = "", ...props }) {
  return (
    <button
      className={`${secondary ? "secondary" : "primary"} ${className}`}
      {...props}
    />
  );
}
export function Textarea({ id, label, help, ...props }) {
  return (
    <div className="field">
      {label && <label htmlFor={id}>{label}</label>}
      <textarea
        id={id}
        aria-describedby={help ? `${id}-help` : undefined}
        {...props}
      />
      {help && (
        <p id={`${id}-help`} className="hint">
          {help}
        </p>
      )}
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
  return (
    <div className="load-progress" aria-label="Model loading progress">
      <div className="progress-heading">
        <span className="spinner" aria-hidden="true" />
        <strong>{status}</strong>
      </div>
      <p className="hint">
        {(loaded / 1e6).toFixed(1)} MB received · model and tokenizer
      </p>
      {list.map(([name, file]) => (
        <div className="file-progress" key={name}>
          <div>
            <span>{name}</span>
            <span>
              {file.done
                ? "Done"
                : file.total
                  ? `${Math.floor((file.loaded / file.total) * 100)}%`
                  : `${(file.loaded / 1e6).toFixed(1)} MB`}
            </span>
          </div>
          <Progress
            label={`Download ${name}`}
            value={
              file.done
                ? 100
                : file.total
                  ? (file.loaded / file.total) * 100
                  : null
            }
          />
        </div>
      ))}
      <p className="hint">
        First use also loads the CPU runtime. Downloads can finish before the
        model is ready.
      </p>
    </div>
  );
}

export function Picker({
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
      <label htmlFor={id}>{label}</label>
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
            primary: "var(--accent)",
            primary25: "var(--amber-soft)",
            primary50: "#f2cf91",
            primary75: "#c48b39",
            neutral0: "var(--surface)",
            neutral5: "var(--surface-soft)",
            neutral10: "var(--amber-soft)",
            neutral20: "var(--input-border)",
            neutral30: "var(--focus)",
            neutral40: "var(--muted)",
            neutral50: "var(--muted)",
            neutral60: "var(--muted)",
            neutral70: "var(--ink)",
            neutral80: "var(--ink)",
            neutral90: "var(--ink)",
          },
        })}
      />
    </div>
  );
}

export function HelpTooltip({ id, label, children }) {
  const [open, setOpen] = useState(false);
  return (
    <span
      className="help-tooltip"
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
        <span aria-hidden="true">ⓘ</span>
      </button>
      <span id={id} role="tooltip" className="help-content" hidden={!open}>
        {children}
      </span>
    </span>
  );
}
