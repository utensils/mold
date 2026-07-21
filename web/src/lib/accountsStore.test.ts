import { beforeEach, describe, expect, it } from "vitest";
import {
  catalogAuthHeaders,
  clearCivitaiToken,
  clearHfToken,
  getAccountTokens,
  maskToken,
  setCivitaiToken,
  setHfToken,
} from "./accountsStore";

describe("accountsStore", () => {
  beforeEach(() => localStorage.clear());

  it("round-trips HF and Civitai tokens through localStorage", () => {
    setHfToken("hf_secretvalue1234");
    setCivitaiToken("cv_abcdef7890");
    expect(getAccountTokens()).toEqual({
      hfToken: "hf_secretvalue1234",
      civitaiToken: "cv_abcdef7890",
    });
  });

  it("clearing a token removes only that one and drops the row when empty", () => {
    setHfToken("hf_x1234");
    setCivitaiToken("cv_y5678");
    clearHfToken();
    expect(getAccountTokens()).toEqual({ civitaiToken: "cv_y5678" });
    clearCivitaiToken();
    expect(getAccountTokens()).toEqual({});
    expect(localStorage.getItem("mold.web.accounts.v1")).toBeNull();
  });

  it("masks tokens to a recognizable prefix + last four", () => {
    expect(maskToken("hf_secretvalue1234")).toBe("hf_••••1234");
    expect(maskToken("cv_abcdef7890")).toBe("cv_••••7890");
    expect(maskToken("nopunctuationtoken")).toBe("••••oken");
    expect(maskToken("")).toBe("");
    expect(maskToken(undefined)).toBe("");
  });

  it("catalogAuthHeaders forwards only the tokens that are set", () => {
    expect(catalogAuthHeaders()).toEqual({});
    setHfToken("hf_x1234");
    expect(catalogAuthHeaders()).toEqual({ "x-mold-hf-token": "hf_x1234" });
    setCivitaiToken("cv_y5678");
    expect(catalogAuthHeaders()).toEqual({
      "x-mold-hf-token": "hf_x1234",
      "x-mold-civitai-token": "cv_y5678",
    });
  });

  it("trims whitespace and treats blank input as clearing", () => {
    setHfToken("  hf_trim1234  ");
    expect(getAccountTokens().hfToken).toBe("hf_trim1234");
    setHfToken("   ");
    expect(getAccountTokens().hfToken).toBeUndefined();
  });
});
