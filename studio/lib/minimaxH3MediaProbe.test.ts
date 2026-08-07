import { describe, expect, it } from "vitest";
import { probeMinimaxH3Mp4, probeMinimaxH3Wav } from "./minimaxH3MediaProbe";
import {
  h264AacMp4Fixture,
  pcmWavFixture,
} from "./minimaxH3MediaProbe.testFixtures";

describe("MiniMax H3 media probing", () => {
  it("keeps exact 44.1 kHz WAV samples and uses server-compatible ceiling duration", () => {
    const facts = probeMinimaxH3Wav(
      pcmWavFixture({
        sampleRate: 44_100,
        channels: 2,
        sampleCount: 44_101,
      }),
    );

    expect(facts).toEqual({
      mimeType: "audio/wav",
      durationMs: 1_001,
      sampleRate: 44_100,
      channels: 2,
      sampleCount: 44_101,
    });
  });

  it("rejects WAV declarations the server RIFF probe would reject", () => {
    const wav = pcmWavFixture({
      sampleRate: 44_100,
      channels: 2,
      sampleCount: 2_000,
    });
    new DataView(wav.buffer).setUint32(28, 44_100, true);

    expect(() => probeMinimaxH3Wav(wav)).toThrow(
      "WAVE byte rate is inconsistent",
    );
  });

  it("derives bounded H.264/AAC container hints for upload canonicalization", () => {
    expect(probeMinimaxH3Mp4(h264AacMp4Fixture())).toEqual({
      mimeType: "video/mp4",
      width: 1_280,
      height: 720,
      frameCount: 48,
      durationMs: 2_000,
      fps: 24,
      hasAudio: true,
      audioDurationMs: 2_021,
      audioSampleCount: 89_088,
      audioSampleRate: 44_100,
      audioChannels: 2,
    });
  });
});
