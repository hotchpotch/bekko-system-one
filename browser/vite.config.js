import { defineConfig, loadEnv } from 'vite';
import { fileURLToPath } from 'node:url';
import { modelProxy } from './scripts/hf-model-proxy.js';

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, fileURLToPath(new URL('.', import.meta.url)), 'HF_');
  return {
    base: './',
    publicDir: false,
    build: { target: 'es2022' },
    plugins: [{
      name: 'local-hf-model-downloads',
      apply: 'serve',
      configureServer(server) {
        server.middlewares.use(modelProxy(process.env.HF_TOKEN || env.HF_TOKEN));
      },
    }],
  };
});
