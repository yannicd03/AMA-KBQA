import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev loop: the FastAPI backend (ama_kbqa/api/, `ama-kbqa-api`) listens on
// 127.0.0.1:8506. Proxying /api keeps the app same-origin, exactly like nginx
// does in the container, so no CORS setup is needed anywhere.
const API_TARGET = process.env.AMA_KBQA_API ?? "http://127.0.0.1:8506";

// `VITE_MOCK=1 npm run dev` swaps the API client for local fixtures with a
// fake streamed run (src/mock/). It is a literal boolean at build time, so a
// normal build drops the mock module and its fixtures completely.
const MOCK = process.env.VITE_MOCK === "1";

export default defineConfig({
  plugins: [react()],
  define: {
    __MOCK__: JSON.stringify(MOCK),
  },
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: API_TARGET,
        changeOrigin: false,
        // Server-Sent Events: never time out the stream. The backend sends a
        // keepalive comment every 15 s, and a run can take several minutes.
        timeout: 0,
        proxyTimeout: 0,
      },
    },
  },
  preview: {
    port: 4173,
  },
  build: {
    target: "es2022",
    sourcemap: false,
    // vis-network's standalone build (~650 kB) is one lazily loaded chunk,
    // fetched only when the graph panel first opens.
    chunkSizeWarningLimit: 700,
  },
});
