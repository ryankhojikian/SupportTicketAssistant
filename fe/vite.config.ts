import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const apiTarget = process.env.VITE_DEV_API_PROXY ?? "http://127.0.0.1:8000";

const proxy = {
  "/api": {
    target: apiTarget,
    changeOrigin: true,
    rewrite: (path: string) => path.replace(/^\/api/, ""),
  },
};

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy,
  },
  preview: {
    host: "0.0.0.0",
    port: 4173,
    proxy,
  },
});
