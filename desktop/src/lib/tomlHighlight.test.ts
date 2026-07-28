import { describe, expect, it } from "vitest";
import { highlightToml } from "./tomlHighlight";

describe("highlightToml", () => {
  it("marks table headers, keys, strings, numbers, and booleans", () => {
    const html = highlightToml(
      ["[chain]", 'model = "ltx-2"', "fps = 24", "audio = true", "[[stage]]"].join("\n"),
    );
    expect(html).toContain('<span class="tk-table">[chain]</span>');
    expect(html).toContain('<span class="tk-key">model</span>');
    expect(html).toContain('<span class="tk-str">&quot;ltx-2&quot;</span>');
    expect(html).toContain('<span class="tk-num">24</span>');
    expect(html).toContain('<span class="tk-bool">true</span>');
    expect(html).toContain('<span class="tk-table">[[stage]]</span>');
  });

  it("marks comments, full-line and trailing", () => {
    const html = highlightToml("# top\nfps = 24 # frames per second");
    expect(html).toContain('<span class="tk-comment"># top</span>');
    expect(html).toContain('<span class="tk-comment"># frames per second</span>');
  });

  it("escapes HTML everywhere, including inside strings", () => {
    const html = highlightToml('prompt = "<script>alert(1)</script> & more"');
    expect(html).not.toContain("<script>");
    expect(html).toContain("&lt;script&gt;");
    expect(html).toContain("&amp; more");
  });

  it("does not treat a # inside a string as a comment", () => {
    const html = highlightToml('prompt = "shade #5 of grey"');
    expect(html).toContain('<span class="tk-str">&quot;shade #5 of grey&quot;</span>');
    expect(html).not.toContain("tk-comment");
  });

  it("passes unknown lines through escaped and keeps line count", () => {
    const source = "not toml <b>\n\nplain";
    const html = highlightToml(source);
    expect(html.split("\n")).toHaveLength(3);
    expect(html).toContain("not toml &lt;b&gt;");
  });
});
