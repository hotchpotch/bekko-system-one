import { spawnSync } from 'node:child_process';
import { cp, mkdir, readdir, rm, stat } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';

export const browserRoot = fileURLToPath(new URL('../', import.meta.url));
export const exportDirectory = fileURLToPath(new URL('../export/space/', import.meta.url));

export function run(command, args, capture = false) {
  const result = spawnSync(command, args, {
    cwd: browserRoot, encoding: 'utf8', stdio: capture ? ['ignore', 'pipe', 'inherit'] : 'inherit',
  });
  if (result.error) throw result.error;
  if (result.status !== 0) throw Error(`${command} failed (exit ${result.status})`);
  return result.stdout;
}

export async function exportSpace() {
  run('npm', ['run', 'build']);
  await rm(exportDirectory, { recursive: true, force: true });
  await mkdir(exportDirectory, { recursive: true });
  // Only Vite's output and the Space card are deployed, never local model files or credentials.
  await cp(new URL('../dist/', import.meta.url), exportDirectory, { recursive: true });
  await cp(new URL('../SPACE_README.md', import.meta.url), `${exportDirectory}/README.md`);
  let bytes = 0;
  const files = await readdir(exportDirectory, { recursive: true });
  for (const file of files) {
    const info = await stat(`${exportDirectory}/${file}`);
    if (info.isFile()) bytes += info.size;
  }
  console.log(`Exported ${(bytes / 1e6).toFixed(1)} MB to ${exportDirectory}`);
}

if (process.argv[1] === fileURLToPath(import.meta.url)) await exportSpace();
