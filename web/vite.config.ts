import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// `npm run dev` proxies the API and media to a running `adbreak label` (default 127.0.0.1:8585).
const server = process.env.ADBREAK_LABEL_URL ?? "http://127.0.0.1:8585";

export default defineConfig({
  plugins: [react()],
  base: "./",
  server: { proxy: { "/api": server, "/files": server } },
});
