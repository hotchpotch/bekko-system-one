/** Stream an asset and report received bytes; unknown sizes stay indeterminate. */
export async function download(url, onProgress = () => {}, fetcher = fetch) {
  const response = await fetcher(url);
  if (!response.ok)
    throw Error(
      `Could not download ${new URL(url).pathname.split("/").pop()} (${response.status}). Please try again.`,
    );
  const length = Number(response.headers.get("content-length"));
  const total =
    !response.headers.get("content-encoding") &&
    Number.isFinite(length) &&
    length > 0
      ? length
      : null;
  let loaded = 0;
  onProgress({ loaded, total, done: false });
  const chunks = [];
  if (response.body) {
    const reader = response.body.getReader();
    try {
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        chunks.push(value);
        loaded += value.length;
        onProgress({
          loaded,
          total: total && total >= loaded ? total : null,
          done: false,
        });
      }
    } finally {
      reader.releaseLock();
    }
  } else {
    const bytes = new Uint8Array(await response.arrayBuffer());
    chunks.push(bytes);
    loaded = bytes.length;
  }
  const bytes = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.length;
  }
  onProgress({ loaded, total: loaded, done: true });
  return bytes;
}
