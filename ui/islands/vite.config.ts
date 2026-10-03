import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Built output is committed to ../../static/islands (served by Streamlit at
// <base>/app/static/islands/). Do not name the dir build/ or dist/ (.gitignore).
export default defineConfig({
  plugins: [react()],
  base: "./",
  build: {
    outDir: "../../static/islands",
    emptyOutDir: true,
    manifest: "manifest.json",
    target: "es2022",
    cssCodeSplit: false,
    rollupOptions: {
      input: { hero: "src/hero/main.tsx" },
      output: {
        entryFileNames: "[name].js",
        chunkFileNames: "assets/[name]-[hash].js",
        assetFileNames: "assets/[name]-[hash][extname]",
      },
    },
  },
});
