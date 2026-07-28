/**
 * Minimal TOML syntax highlighter for the chain script editor. Emits
 * HTML-escaped text with `tk-*` classed spans — deliberately dependency-free
 * and line-based: `mold.chain.v1` scripts are flat tables with single-line
 * values, so a full TOML grammar buys nothing here. Unknown constructs fall
 * through as escaped plain text.
 */

const ESCAPES: Record<string, string> = {
  "&": "&amp;",
  "<": "&lt;",
  ">": "&gt;",
  '"': "&quot;",
};

function esc(text: string): string {
  return text.replace(/[&<>"]/g, (c) => ESCAPES[c]!);
}

function span(cls: string, text: string): string {
  return `<span class="${cls}">${esc(text)}</span>`;
}

/** Highlight a value segment: strings, numbers, booleans, trailing comments. */
function highlightValue(value: string): string {
  let out = "";
  let i = 0;
  while (i < value.length) {
    const rest = value.slice(i);
    let m = rest.match(/^"(?:\\.|[^"\\])*"?|^'[^']*'?/);
    if (m) {
      out += span("tk-str", m[0]);
      i += m[0].length;
      continue;
    }
    m = rest.match(/^#.*$/);
    if (m) {
      out += span("tk-comment", m[0]);
      i += m[0].length;
      continue;
    }
    m = rest.match(/^(?:true|false)\b/);
    if (m) {
      out += span("tk-bool", m[0]);
      i += m[0].length;
      continue;
    }
    m = rest.match(/^[+-]?\d[\d_]*(?:\.\d+)?(?:[eE][+-]?\d+)?/);
    if (m) {
      out += span("tk-num", m[0]);
      i += m[0].length;
      continue;
    }
    out += esc(value[i]!);
    i += 1;
  }
  return out;
}

function highlightLine(line: string): string {
  const table = line.match(/^(\s*)(\[\[?[^\]]*\]\]?)(\s*)(#.*)?$/);
  if (table) {
    return (
      table[1]! +
      span("tk-table", table[2]!) +
      table[3]! +
      (table[4] ? span("tk-comment", table[4]) : "")
    );
  }
  const comment = line.match(/^(\s*)(#.*)$/);
  if (comment) return comment[1]! + span("tk-comment", comment[2]!);
  const kv = line.match(/^(\s*)([A-Za-z0-9_.-]+|"[^"]*")(\s*=\s*)(.*)$/);
  if (kv) return kv[1]! + span("tk-key", kv[2]!) + esc(kv[3]!) + highlightValue(kv[4]!);
  return esc(line);
}

export function highlightToml(source: string): string {
  return source.split("\n").map(highlightLine).join("\n");
}
