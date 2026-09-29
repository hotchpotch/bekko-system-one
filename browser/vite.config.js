import { defineConfig } from "vite";

export default defineConfig({
  base: "./",
  // Models are downloaded from the pinned public Hub repository at runtime.
  publicDir: false,
  build: { target: "es2022" },
});
