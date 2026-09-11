/*
 * Prints the three frozen tables for web/src/styles/tokens.legacy.test.ts
 * from the tree as it is, using that test's own measurers, so a migration
 * pass regenerates the numbers instead of anyone editing them by hand.
 * Run from web/: `bun run ratchet:web`.
 */
import { measure } from "../web/src/styles/tokenRatchet";

function table(name: string, counts: Record<string, number>): string {
  const rows = Object.entries(counts)
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([file, count]) => `  "${file}": ${count},`)
    .join("\n");
  return `const ${name}: Record<string, number> = {\n${rows}\n};`;
}

const m = measure();
process.stdout.write(
  [table("LEGACY_FROZEN", m.legacy), table("LITERAL_FROZEN", m.literal), table("HEX_FROZEN", m.hex)].join(
    "\n\n",
  ) + "\n",
);
