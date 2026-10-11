/** Module reachability complements Knip's test-inclusive export/dependency check.
 * Tests are never roots. Resolve the workspace's exact aliases, including imports
 * inside Vue SFCs, which Knip's per-package graph does not reliably connect. */
import ts from "typescript";
import { existsSync, readFileSync, readdirSync, statSync } from "node:fs";
import { dirname, extname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const repository = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const config = {
  sourceDirs: ["studio", "ui", "web/src", "desktop/src"],
  entries: [
    "web/src/main.ts",
    "desktop/src/main.ts",
    "desktop/src/mobile/main.ts",
    // The reviewed @mold/studio package facade is an intentional public API.
    "studio/index.ts",
    // Build-time code is reachable through Vite, not application entry modules.
    "studio/vite/serverUrl.ts",
  ],
  // Exact paths and reasons only. Never add ordinary production modules merely
  // because a test imports them. These are retained test contracts/fixtures.
  exceptions: {
    "studio/vitest.config.ts": "Studio Vitest runner configuration",
    "studio/test-setup.ts": "Studio Vitest environment setup",
    "desktop/src/test-setup.ts": "Desktop Vitest environment setup",
    "web/src/__tests__/setup.ts": "Web Vitest environment setup",
    "desktop/src/lib/testSupport/memoryLocalStorage.ts":
      "In-memory storage test double",
    "studio/lib/generationProfile.testFixtures.ts":
      "Generation contract fixtures",
    "studio/lib/minimaxH3Inventory.testFixtures.ts": "H3 inventory fixtures",
    "studio/lib/minimaxH3MediaProbe.testFixtures.ts": "H3 media fixtures",
    "studio/lib/glbFixture.ts": "Tiny deterministic GLB fixture builder",
    "studio/lib/galleryPerfBudget.ts":
      "Operation-count budgets enforced by performance tests",
    "studio/lib/lexicon.ts":
      "Design vocabulary contract enforced across surfaces",
    "web/src/styles/tokenRatchet.ts":
      "CSS token migration ratchet, run by scripts/web-token-ratchet.ts",
  },
};

function sourceFile(file) {
  return (
    /\.(?:[cm]?[jt]sx?|vue)$/.test(file) &&
    !/\.(?:test|spec)\.[cm]?[jt]sx?$/.test(file) &&
    !file.endsWith(".d.ts")
  );
}

function imports(source, file) {
  const body = file.endsWith(".vue")
    ? [...source.matchAll(/<script\b[^>]*>([\s\S]*?)<\/script>/g)]
        .map((match) => match[1])
        .join("\n")
    : source;
  const ast = ts.createSourceFile(file, body, ts.ScriptTarget.Latest, true);
  const found = [];
  const add = (node) => {
    if (node && ts.isStringLiteralLike(node)) found.push(node.text);
  };
  const visit = (node) => {
    if (ts.isImportDeclaration(node) || ts.isExportDeclaration(node))
      add(node.moduleSpecifier);
    if (
      ts.isCallExpression(node) &&
      (node.expression.kind === ts.SyntaxKind.ImportKeyword ||
        node.expression.getText(ast) === "require")
    )
      add(node.arguments[0]);
    if (ts.isImportTypeNode(node) && ts.isLiteralTypeNode(node.argument))
      add(node.argument.literal);
    ts.forEachChild(node, visit);
  };
  visit(ast);
  return found;
}

export function analyzeFrontend(root, options = config) {
  const files = new Set();
  function walk(directory) {
    if (!existsSync(directory)) return;
    for (const entry of readdirSync(directory, { withFileTypes: true })) {
      if (["node_modules", "dist", ".vite"].includes(entry.name)) continue;
      const file = join(directory, entry.name);
      if (entry.isDirectory()) walk(file);
      else if (sourceFile(file)) files.add(file);
    }
  }
  for (const directory of options.sourceDirs) walk(resolve(root, directory));
  const reached = new Set();
  const unresolved = [];
  const queue = options.entries.map((file) => resolve(root, file));
  function resolveImport(specifier, importer) {
    const clean = specifier.split("?")[0];
    let base;
    if (clean.startsWith("@studio/"))
      base = resolve(root, "studio", clean.slice(8));
    else if (clean.startsWith("@ui/"))
      base = resolve(root, "ui", clean.slice(4));
    else if (clean.startsWith("@/"))
      base = resolve(root, "desktop/src", clean.slice(2));
    else if (clean.startsWith(".")) base = resolve(dirname(importer), clean);
    else return null; // external package or platform import
    if (extname(base) && !/\.(?:[cm]?[jt]sx?|vue)$/.test(base)) return null;
    const candidates = [
      base,
      ...[".ts", ".tsx", ".js", ".mjs", ".vue", "/index.ts", "/index.js"].map(
        (suffix) => base + suffix,
      ),
    ];
    const target = candidates.find(
      (file) => existsSync(file) && statSync(file).isFile(),
    );
    if (!target) unresolved.push(`${relative(root, importer)}: ${specifier}`);
    return target ?? null;
  }
  while (queue.length) {
    const file = queue.pop();
    if (reached.has(file)) continue;
    if (!existsSync(file)) {
      unresolved.push(`entry: ${relative(root, file)}`);
      continue;
    }
    reached.add(file);
    for (const specifier of imports(readFileSync(file, "utf8"), file)) {
      const target = resolveImport(specifier, file);
      if (target && sourceFile(target)) queue.push(target);
    }
  }
  const unreachable = [...files]
    .filter(
      (file) =>
        !reached.has(file) && !(relative(root, file) in options.exceptions),
    )
    .map((file) => relative(root, file))
    .sort();
  return {
    unreachable,
    unresolved,
    scanned: files.size,
    reached: reached.size,
  };
}

if (
  process.argv[1] &&
  resolve(process.argv[1]) === fileURLToPath(import.meta.url)
) {
  const result = analyzeFrontend(repository);
  for (const file of result.unreachable)
    console.error(`Production-unreachable module: ${file}`);
  for (const item of result.unresolved)
    console.error(`Unresolved production import: ${item}`);
  if (result.unreachable.length || result.unresolved.length)
    process.exitCode = 1;
  else
    console.log(
      `frontend-production-graph: ${result.scanned} modules checked; ${Object.keys(config.exceptions).length} documented test/tooling exceptions`,
    );
}
