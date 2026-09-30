import { exportSpace, exportDirectory, run } from './export-space.js';

const repo = process.argv[2] || 'hotchpotch/bekko-system-one-in-browser';
if (process.argv.length > 3 || !/^[\w-]+\/[\w.-]+$/.test(repo)) {
  throw Error('Usage: npm run deploy:space -- owner/space-name');
}

await exportSpace();
// Authentication stays with the HF CLI (hf auth login / HF_TOKEN), never in the export.
run('hf', ['repos', 'create', repo, '--type', 'space', '--sdk', 'static', '--private', '--exist-ok']);
const info = JSON.parse(run('hf', ['spaces', 'info', repo, '--expand', 'private,sdk', '--json'], true));
if (info.sdk !== 'static') {
  throw Error('Deployment requires a Static Space. Existing visibility is never changed automatically.');
}
run('hf', ['upload', repo, exportDirectory, '.', '--type', 'space',
  '--commit-message', 'Deploy Bekko browser demo']);
console.log(`Deployed to https://huggingface.co/spaces/${repo} (${info.private ? "private" : "public"})`);
console.log(`Check startup with: hf spaces wait ${repo} --timeout 5m`);
