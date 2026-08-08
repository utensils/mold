function join(...parts: readonly Uint8Array[]): Uint8Array {
  const result = new Uint8Array(
    parts.reduce((total, part) => total + part.length, 0),
  );
  let offset = 0;
  for (const part of parts) {
    result.set(part, offset);
    offset += part.length;
  }
  return result;
}

function ascii(value: string): Uint8Array {
  return Uint8Array.from(
    [...value].map((character) => character.charCodeAt(0)),
  );
}

function be32(value: number): Uint8Array {
  const result = new Uint8Array(4);
  new DataView(result.buffer).setUint32(0, value, false);
  return result;
}

function box(type: string, ...payload: readonly Uint8Array[]): Uint8Array {
  const body = join(...payload);
  return join(be32(body.length + 8), ascii(type), body);
}

function descriptor(tag: number, payload: Uint8Array): Uint8Array {
  if (payload.length > 0x7f) throw new Error("fixture descriptor is too large");
  return join(Uint8Array.of(tag, payload.length), payload);
}

function fullBoxPayload(...payload: readonly Uint8Array[]): Uint8Array {
  return join(new Uint8Array(4), ...payload);
}

function sampleTable(sampleEntry: Uint8Array, sampleCount: number): Uint8Array {
  return box(
    "stbl",
    box("stsd", fullBoxPayload(be32(1), sampleEntry)),
    box("stts", fullBoxPayload(be32(1), be32(sampleCount), be32(1))),
    box("stsc", fullBoxPayload(be32(0))),
    box("stsz", fullBoxPayload(be32(1), be32(sampleCount))),
    box("stco", fullBoxPayload(be32(0))),
  );
}

function track(options: {
  id: number;
  handler: "vide" | "soun";
  timescale: number;
  duration: number;
  sampleCount: number;
  sampleEntry: Uint8Array;
}): Uint8Array {
  const tkhd = new Uint8Array(20);
  new DataView(tkhd.buffer).setUint32(12, options.id, false);
  const mdhd = new Uint8Array(24);
  const mdhdView = new DataView(mdhd.buffer);
  mdhdView.setUint32(12, options.timescale, false);
  mdhdView.setUint32(16, options.duration, false);
  const hdlr = new Uint8Array(24);
  hdlr.set(ascii(options.handler), 8);
  return box(
    "trak",
    box("tkhd", tkhd),
    box(
      "mdia",
      box("mdhd", mdhd),
      box("hdlr", hdlr),
      box(
        "minf",
        box("dinf"),
        sampleTable(options.sampleEntry, options.sampleCount),
      ),
    ),
  );
}

function avcSampleEntry(width: number, height: number): Uint8Array {
  const visual = new Uint8Array(78);
  const view = new DataView(visual.buffer);
  view.setUint16(24, width, false);
  view.setUint16(26, height, false);
  const avcConfig = Uint8Array.of(
    1,
    100,
    0,
    31,
    0xff,
    0xe1,
    0,
    1,
    0x67,
    1,
    0,
    1,
    0x68,
  );
  return box("avc1", visual, box("avcC", avcConfig));
}

function aacSampleEntry(sampleRate: number, channels: number): Uint8Array {
  const audio = new Uint8Array(28);
  const view = new DataView(audio.buffer);
  view.setUint16(16, channels, false);
  view.setUint16(18, 16, false);
  view.setUint32(24, sampleRate * 65_536, false);
  // AAC-LC, 44.1 kHz, stereo, 1024-sample frames.
  const specific = descriptor(0x05, Uint8Array.of(0x12, 0x10));
  const decoderConfig = descriptor(
    0x04,
    join(Uint8Array.of(0x40, 0x15, 0, 0, 0), new Uint8Array(8), specific),
  );
  const es = descriptor(0x03, join(Uint8Array.of(0, 1, 0), decoderConfig));
  return box("mp4a", audio, box("esds", fullBoxPayload(es)));
}

export function h264AacMp4Fixture(): Uint8Array {
  const video = track({
    id: 1,
    handler: "vide",
    timescale: 24_000,
    duration: 48_000,
    sampleCount: 48,
    sampleEntry: avcSampleEntry(1_280, 720),
  });
  const audio = track({
    id: 2,
    handler: "soun",
    timescale: 44_100,
    duration: 89_088,
    sampleCount: 87,
    sampleEntry: aacSampleEntry(44_100, 2),
  });
  return join(
    box("ftyp", ascii("isom"), be32(0), ascii("isom")),
    box("moov", box("mvhd", new Uint8Array(4)), video, audio),
  );
}

export function pcmWavFixture(options: {
  sampleRate: number;
  channels: number;
  sampleCount: number;
  bitsPerSample?: number;
}): Uint8Array {
  const bitsPerSample = options.bitsPerSample ?? 16;
  const blockAlign = options.channels * (bitsPerSample / 8);
  const byteRate = options.sampleRate * blockAlign;
  const dataBytes = options.sampleCount * blockAlign;
  const result = new Uint8Array(44 + dataBytes);
  result.set(ascii("RIFF"), 0);
  result.set(ascii("WAVE"), 8);
  result.set(ascii("fmt "), 12);
  result.set(ascii("data"), 36);
  const view = new DataView(result.buffer);
  view.setUint32(4, result.length - 8, true);
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, options.channels, true);
  view.setUint32(24, options.sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bitsPerSample, true);
  view.setUint32(40, dataBytes, true);
  return result;
}
