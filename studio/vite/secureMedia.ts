import { readFileSync } from "node:fs";
import type { Plugin } from "vite";

/** Keep the service worker at the app root so its scope is restricted to Studio. */
export function secureMediaPlugin(): Plugin {
  const source = readFileSync(
    new URL("../api/secureMediaWorker.js", import.meta.url),
    "utf8",
  );
  return {
    name: "mold-secure-media",
    generateBundle() {
      this.emitFile({ type: "asset", fileName: "secure-media-sw.js", source });
    },
    configureServer(server) {
      server.middlewares.use((request, response, next) => {
        if (!request.url?.split("?")[0]?.endsWith("/secure-media-sw.js")) {
          next();
          return;
        }
        response.setHeader("Content-Type", "application/javascript");
        response.setHeader("Service-Worker-Allowed", server.config.base);
        response.end(source);
      });
    },
  };
}
