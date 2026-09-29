// Keep editable keys in rows until submission so duplicates cannot overwrite values.
export function contextKeyError(entries, index) {
  const key = entries[index][0];
  if (!key.trim()) return 'Enter a context key.';
  if (entries.some(([other], i) => i !== index && other === key)) {
    return 'This key is already used. Choose a different key.';
  }
  return '';
}

export function contextObject(entries) {
  entries.forEach((_, index) => {
    const error = contextKeyError(entries, index);
    if (error) throw Error(error);
  });
  return Object.fromEntries(entries);
}
