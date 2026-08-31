import { afterEach, describe, expect, it } from "vitest";
import {
  beginRetainedSourceReuseIntent,
  clearRetainedSourceReuseIntent,
  retainedSourceReuseIntent,
  setRetainedSourceReuseIntent,
  setRetainedSourceReuseIntentIfCurrent,
} from "./retainedSourceReuse";

afterEach(clearRetainedSourceReuseIntent);

describe("retained source reuse intent", () => {
  it("carries only the exact output identity, target, and opaque inventory", () => {
    setRetainedSourceReuseIntent({
      filename: "print.png",
      origin: { baseUrl: "http://host:7680", apiKey: "secret" },
      inventory: {
        availability: "available",
        members: [
          {
            member_id: "opaque",
            role: "audio_file",
            display_name: "audio",
            size_bytes: 3,
          },
        ],
      },
    });
    expect(retainedSourceReuseIntent()?.filename).toBe("print.png");
    expect(JSON.stringify(retainedSourceReuseIntent())).not.toMatch(
      /queue-media|pin_id|server_path/,
    );
  });

  it("rejects stale async inventory after reset, new print, or source clear", () => {
    const version = beginRetainedSourceReuseIntent();
    clearRetainedSourceReuseIntent();
    expect(
      setRetainedSourceReuseIntentIfCurrent(version, {
        filename: "old.png",
        origin: { baseUrl: "http://old:7680", apiKey: "secret" },
        inventory: { availability: "available", members: [] },
      }),
    ).toBe(false);
    expect(retainedSourceReuseIntent()).toBeNull();
  });
});
