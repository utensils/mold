import { describe, expect, it } from "vitest";
import { newGenerateForm } from "./generateForm";
import { attachPickedImage, attachPickedVideo } from "./sourceAttachment";

describe("source attachment", () => {
  it("applies one image policy for uploads and gallery actions", () => {
    const form = newGenerateForm();
    form.sourceFit = { mode: "pad-repaint" };
    attachPickedImage(form, { filename: "source.png", base64: "QUJD" });
    expect(form.sourceImage).toBe("QUJD");
    expect(form.sourceImageName).toBe("source.png");
    expect(form.sourceFit).toEqual({ mode: "lanczos-resize" });
  });

  it("attaches a gallery video to the existing LTX source-video field", () => {
    const form = newGenerateForm();
    attachPickedVideo(form, { filename: "clip.mp4", base64: "QUJD" });
    expect(form.sourceVideo).toEqual({ filename: "clip.mp4", base64: "QUJD" });
  });
});
