import React from "react";

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
      <label htmlFor={id}>{label}</label>
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
