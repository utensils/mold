import type { Plugin } from "vite";

export function formatDevServerUrl(
  label: string,
  host: string | boolean,
  port: number,
  pathname = "/",
): string {
  const usableHost =
    host === true || host === false || host === "0.0.0.0" || host === "::"
      ? "localhost"
      : host;
  const path = pathname.startsWith("/") ? pathname : `/${pathname}`;
  return `${label}: http://${usableHost}:${port}${path}`;
}

/** Print the exact browser URL after Vite has successfully bound its port. */
export function devServerUrlPlugin(label: string, pathname = "/"): Plugin {
  return {
    name: `mold-dev-server-url-${label.toLowerCase().replaceAll(/\W+/g, "-")}`,
    configureServer(server) {
      server.httpServer?.once("listening", () => {
        const address = server.httpServer?.address();
        if (!address || typeof address === "string") return;
        const configuredHost = server.config.server.host ?? "localhost";
        server.config.logger.info(
          `\n  ${formatDevServerUrl(label, configuredHost, address.port, pathname)}`,
        );
      });
    },
  };
}
