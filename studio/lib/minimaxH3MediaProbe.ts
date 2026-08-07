/**
 * Browser-side MiniMax H3 media probing.
 *
 * These parsers provide bounded provisional descriptors for the authenticated
 * upload session. They read container metadata directly instead of asking Web
 * Audio to decode (and potentially resample) the source. The host always
 * replaces uploaded descriptors with content-probed canonical metadata before
 * rebinding the one-use generation scope; MP4 packet arithmetic is never queue
 * authority.
 */

export interface MinimaxH3WavFacts {
  mimeType: "audio/wav";
  durationMs: number;
  sampleRate: number;
  channels: number;
  sampleCount: number;
}

export interface MinimaxH3Mp4Facts {
  mimeType: "video/mp4";
  width: number;
  height: number;
  frameCount: number;
  durationMs: number;
  fps: number;
  hasAudio: boolean;
  audioDurationMs: number | null;
  /** Provisional AAC packet arithmetic, replaced by decoded host metadata. */
  audioSampleCount: number | null;
  audioSampleRate: number | null;
  audioChannels: number | null;
}

interface IsoBox {
  type: string;
  start: number;
  dataStart: number;
  end: number;
}

interface IsoTrack {
  id: number;
  handler: string;
  timescale: number;
  duration: bigint;
  sampleEntry: IsoBox;
  sampleCount: number;
}

interface AacFacts {
  sampleRate: number;
  channels: number;
  samplesPerPacket: number;
}

const AAC_SAMPLE_RATES = [
  96_000, 88_200, 64_000, 48_000, 44_100, 32_000, 24_000, 22_050, 16_000,
  12_000, 11_025, 8_000, 7_350,
] as const;

function bytesOf(input: ArrayBuffer | Uint8Array): Uint8Array {
  return input instanceof Uint8Array ? input : new Uint8Array(input);
}

function text(bytes: Uint8Array, offset: number, length: number): string {
  let value = "";
  for (let index = 0; index < length; index += 1) {
    value += String.fromCharCode(bytes[offset + index] ?? 0);
  }
  return value;
}

function requireRange(
  bytes: Uint8Array,
  offset: number,
  length: number,
  label: string,
): void {
  if (
    !Number.isSafeInteger(offset) ||
    !Number.isSafeInteger(length) ||
    offset < 0 ||
    length < 0 ||
    offset + length > bytes.length
  ) {
    throw new Error(`${label} is truncated.`);
  }
}

function u16be(bytes: Uint8Array, offset: number, label: string): number {
  requireRange(bytes, offset, 2, label);
  return new DataView(
    bytes.buffer,
    bytes.byteOffset,
    bytes.byteLength,
  ).getUint16(offset, false);
}

function u16le(bytes: Uint8Array, offset: number, label: string): number {
  requireRange(bytes, offset, 2, label);
  return new DataView(
    bytes.buffer,
    bytes.byteOffset,
    bytes.byteLength,
  ).getUint16(offset, true);
}

function u32be(bytes: Uint8Array, offset: number, label: string): number {
  requireRange(bytes, offset, 4, label);
  return new DataView(
    bytes.buffer,
    bytes.byteOffset,
    bytes.byteLength,
  ).getUint32(offset, false);
}

function u32le(bytes: Uint8Array, offset: number, label: string): number {
  requireRange(bytes, offset, 4, label);
  return new DataView(
    bytes.buffer,
    bytes.byteOffset,
    bytes.byteLength,
  ).getUint32(offset, true);
}

function u64be(bytes: Uint8Array, offset: number, label: string): bigint {
  requireRange(bytes, offset, 8, label);
  const high = BigInt(u32be(bytes, offset, label));
  const low = BigInt(u32be(bytes, offset + 4, label));
  return (high << 32n) | low;
}

function safeNumber(value: bigint, label: string): number {
  if (value < 0n || value > BigInt(Number.MAX_SAFE_INTEGER)) {
    throw new Error(`${label} exceeds the browser's safe integer range.`);
  }
  return Number(value);
}

function parseBox(
  bytes: Uint8Array,
  start: number,
  end: number,
  label: string,
): IsoBox {
  requireRange(bytes, start, 8, `${label} box header`);
  const shortSize = u32be(bytes, start, `${label} box size`);
  const type = text(bytes, start + 4, 4);
  let headerSize = 8;
  let size: number;
  if (shortSize === 1) {
    const largeSize = u64be(bytes, start + 8, `${type} box size`);
    size = safeNumber(largeSize, `${type} box size`);
    headerSize = 16;
  } else if (shortSize === 0) {
    size = end - start;
  } else {
    size = shortSize;
  }
  if (size < headerSize || start + size > end) {
    throw new Error(`${type || label} box is truncated.`);
  }
  return {
    type,
    start,
    dataStart: start + headerSize,
    end: start + size,
  };
}

function parseBoxes(
  bytes: Uint8Array,
  start: number,
  end: number,
  label: string,
): IsoBox[] {
  requireRange(bytes, start, end - start, label);
  const boxes: IsoBox[] = [];
  let cursor = start;
  while (cursor < end) {
    const box = parseBox(bytes, cursor, end, label);
    boxes.push(box);
    cursor = box.end;
  }
  return boxes;
}

function requiredBox(boxes: readonly IsoBox[], type: string): IsoBox {
  const match = boxes.find((box) => box.type === type);
  if (!match) throw new Error(`MP4 is missing its ${type} box.`);
  return match;
}

function parseSingleBox(
  bytes: Uint8Array,
  start: number,
  end: number,
  label: string,
): IsoBox {
  if (start >= end) throw new Error(`${label} is missing.`);
  return parseBox(bytes, start, end, label);
}

function fullBoxFlags(bytes: Uint8Array, box: IsoBox): number {
  requireRange(bytes, box.dataStart, 4, `${box.type} full-box header`);
  return (
    ((bytes[box.dataStart + 1] ?? 0) << 16) |
    ((bytes[box.dataStart + 2] ?? 0) << 8) |
    (bytes[box.dataStart + 3] ?? 0)
  );
}

function parseWavFacts(bytes: Uint8Array): MinimaxH3WavFacts {
  requireRange(bytes, 0, 12, "WAVE header");
  if (text(bytes, 0, 4) !== "RIFF" || text(bytes, 8, 4) !== "WAVE") {
    throw new Error("Audio is not a RIFF/WAVE file.");
  }
  const riffEnd = 8 + u32le(bytes, 4, "WAVE container length");
  if (riffEnd < 12 || riffEnd > bytes.length) {
    throw new Error("WAVE container is truncated.");
  }

  let cursor = 12;
  let sampleRate: number | null = null;
  let channels: number | null = null;
  let byteRate: number | null = null;
  let blockAlign: number | null = null;
  let dataBytes: number | null = null;

  while (cursor + 8 <= riffEnd) {
    const kind = text(bytes, cursor, 4);
    const length = u32le(bytes, cursor + 4, "WAVE chunk length");
    const dataStart = cursor + 8;
    const paddedLength = length + (length % 2);
    const nextChunk = dataStart + paddedLength;
    if (nextChunk > riffEnd || nextChunk > bytes.length) {
      throw new Error("WAVE chunk is truncated.");
    }

    if (kind === "fmt ") {
      if (length < 16) throw new Error("WAVE fmt chunk is truncated.");
      const encoding = u16le(bytes, dataStart, "WAVE sample encoding");
      if (encoding !== 1 && encoding !== 3) {
        throw new Error("WAVE sample encoding is unsupported.");
      }
      const observedChannels = u16le(
        bytes,
        dataStart + 2,
        "WAVE channel count",
      );
      const observedSampleRate = u32le(
        bytes,
        dataStart + 4,
        "WAVE sample rate",
      );
      const observedByteRate = u32le(bytes, dataStart + 8, "WAVE byte rate");
      const observedBlockAlign = u16le(
        bytes,
        dataStart + 12,
        "WAVE block alignment",
      );
      const bitsPerSample = u16le(
        bytes,
        dataStart + 14,
        "WAVE bits per sample",
      );
      if (
        observedChannels === 0 ||
        observedSampleRate === 0 ||
        observedBlockAlign === 0 ||
        bitsPerSample === 0 ||
        bitsPerSample % 8 !== 0
      ) {
        throw new Error("WAVE format fields are invalid.");
      }
      const expectedAlign = observedChannels * (bitsPerSample / 8);
      if (expectedAlign > 0xffff || observedBlockAlign !== expectedAlign) {
        throw new Error("WAVE block alignment is inconsistent.");
      }
      if (observedByteRate !== observedSampleRate * observedBlockAlign) {
        throw new Error("WAVE byte rate is inconsistent.");
      }
      channels = observedChannels;
      sampleRate = observedSampleRate;
      byteRate = observedByteRate;
      blockAlign = observedBlockAlign;
    } else if (kind === "data") {
      dataBytes = length;
    }

    cursor = nextChunk;
    if (sampleRate !== null && dataBytes !== null) break;
  }

  if (!sampleRate) throw new Error("WAVE sample rate is missing.");
  if (!channels) throw new Error("WAVE channel count is missing.");
  if (!byteRate) throw new Error("WAVE byte rate is missing.");
  if (!blockAlign) throw new Error("WAVE block alignment is missing.");
  if (!dataBytes) throw new Error("WAVE data chunk is missing or empty.");
  if (dataBytes % blockAlign !== 0) {
    throw new Error("WAVE data is not frame-aligned.");
  }

  return {
    mimeType: "audio/wav",
    durationMs: Math.ceil((dataBytes * 1_000) / byteRate),
    sampleRate,
    channels,
    sampleCount: dataBytes / blockAlign,
  };
}

function parseTrackId(bytes: Uint8Array, tkhd: IsoBox): number {
  requireRange(bytes, tkhd.dataStart, 4, "tkhd full-box header");
  const version = bytes[tkhd.dataStart];
  const offset = version === 1 ? 20 : version === 0 ? 12 : -1;
  if (offset < 0) throw new Error("MP4 tkhd version is unsupported.");
  const id = u32be(bytes, tkhd.dataStart + offset, "MP4 track id");
  if (id === 0) throw new Error("MP4 track id cannot be zero.");
  return id;
}

function parseMediaTime(
  bytes: Uint8Array,
  mdhd: IsoBox,
): { timescale: number; duration: bigint } {
  requireRange(bytes, mdhd.dataStart, 4, "mdhd full-box header");
  const version = bytes[mdhd.dataStart];
  if (version === 0) {
    const timescale = u32be(bytes, mdhd.dataStart + 12, "MP4 timescale");
    const duration = BigInt(
      u32be(bytes, mdhd.dataStart + 16, "MP4 track duration"),
    );
    if (timescale === 0) throw new Error("MP4 track timescale is zero.");
    return { timescale, duration };
  }
  if (version === 1) {
    const timescale = u32be(bytes, mdhd.dataStart + 20, "MP4 timescale");
    const duration = u64be(bytes, mdhd.dataStart + 24, "MP4 track duration");
    if (timescale === 0) throw new Error("MP4 track timescale is zero.");
    return { timescale, duration };
  }
  throw new Error("MP4 mdhd version is unsupported.");
}

function parseSampleCount(bytes: Uint8Array, stsz: IsoBox): number {
  const sampleSize = u32be(bytes, stsz.dataStart + 4, "MP4 sample size");
  const sampleCount = u32be(bytes, stsz.dataStart + 8, "MP4 sample count");
  if (sampleSize === 0) {
    requireRange(
      bytes,
      stsz.dataStart + 12,
      sampleCount * 4,
      "MP4 sample-size table",
    );
    if (stsz.dataStart + 12 + sampleCount * 4 > stsz.end) {
      throw new Error("MP4 sample-size table is truncated.");
    }
  }
  return sampleCount;
}

function validateTimeToSample(bytes: Uint8Array, stts: IsoBox): void {
  const entryCount = u32be(
    bytes,
    stts.dataStart + 4,
    "MP4 time-to-sample entry count",
  );
  requireRange(
    bytes,
    stts.dataStart + 8,
    entryCount * 8,
    "MP4 time-to-sample table",
  );
  if (stts.dataStart + 8 + entryCount * 8 > stts.end) {
    throw new Error("MP4 time-to-sample table is truncated.");
  }
}

function parseTrack(bytes: Uint8Array, trak: IsoBox): IsoTrack {
  const trackBoxes = parseBoxes(bytes, trak.dataStart, trak.end, "trak");
  const id = parseTrackId(bytes, requiredBox(trackBoxes, "tkhd"));
  const mdia = requiredBox(trackBoxes, "mdia");
  const mediaBoxes = parseBoxes(bytes, mdia.dataStart, mdia.end, "mdia");
  const { timescale, duration } = parseMediaTime(
    bytes,
    requiredBox(mediaBoxes, "mdhd"),
  );
  const hdlr = requiredBox(mediaBoxes, "hdlr");
  requireRange(bytes, hdlr.dataStart + 8, 4, "MP4 media handler");
  const handler = text(bytes, hdlr.dataStart + 8, 4);
  const minf = requiredBox(mediaBoxes, "minf");
  const mediaInfoBoxes = parseBoxes(bytes, minf.dataStart, minf.end, "minf");
  requiredBox(mediaInfoBoxes, "dinf");
  const stbl = requiredBox(mediaInfoBoxes, "stbl");
  const sampleBoxes = parseBoxes(bytes, stbl.dataStart, stbl.end, "stbl");
  requiredBox(sampleBoxes, "stsc");
  if (!sampleBoxes.some((box) => box.type === "stco" || box.type === "co64")) {
    throw new Error("MP4 is missing its stco or co64 box.");
  }
  validateTimeToSample(bytes, requiredBox(sampleBoxes, "stts"));
  const sampleCount = parseSampleCount(bytes, requiredBox(sampleBoxes, "stsz"));
  const stsd = requiredBox(sampleBoxes, "stsd");
  const sampleEntry = parseSingleBox(
    bytes,
    stsd.dataStart + 8,
    stsd.end,
    "MP4 sample description",
  );
  return { id, handler, timescale, duration, sampleEntry, sampleCount };
}

function validateAvcConfiguration(bytes: Uint8Array, avc1: IsoBox): void {
  const avcC = parseSingleBox(
    bytes,
    avc1.dataStart + 78,
    avc1.end,
    "AVC configuration",
  );
  if (avcC.type !== "avcC") {
    throw new Error("H3 video references require H.264/AVC video.");
  }
  requireRange(bytes, avcC.dataStart, 7, "AVC configuration");
  let cursor = avcC.dataStart + 5;
  const sequenceCount = (bytes[cursor] ?? 0) & 0x1f;
  cursor += 1;
  if (sequenceCount === 0) {
    throw new Error("H.264 video is missing a sequence parameter set.");
  }
  for (let index = 0; index < sequenceCount; index += 1) {
    const length = u16be(bytes, cursor, "H.264 sequence parameter set");
    cursor += 2;
    if (length === 0) {
      throw new Error("H.264 sequence parameter set is empty.");
    }
    requireRange(bytes, cursor, length, "H.264 sequence parameter set");
    cursor += length;
  }
  requireRange(bytes, cursor, 1, "H.264 picture parameter set count");
  const pictureCount = bytes[cursor] ?? 0;
  cursor += 1;
  if (pictureCount === 0) {
    throw new Error("H.264 video is missing a picture parameter set.");
  }
  for (let index = 0; index < pictureCount; index += 1) {
    const length = u16be(bytes, cursor, "H.264 picture parameter set");
    cursor += 2;
    if (length === 0) {
      throw new Error("H.264 picture parameter set is empty.");
    }
    requireRange(bytes, cursor, length, "H.264 picture parameter set");
    cursor += length;
  }
  if (cursor > avcC.end) throw new Error("AVC configuration is truncated.");
}

interface Descriptor {
  tag: number;
  dataStart: number;
  end: number;
  next: number;
}

function descriptor(
  bytes: Uint8Array,
  start: number,
  end: number,
  label: string,
): Descriptor {
  requireRange(bytes, start, 2, `${label} descriptor`);
  const tag = bytes[start] ?? 0;
  let cursor = start + 1;
  let length = 0;
  let complete = false;
  for (let index = 0; index < 4; index += 1) {
    requireRange(bytes, cursor, 1, `${label} descriptor length`);
    const value = bytes[cursor] ?? 0;
    cursor += 1;
    length = length * 128 + (value & 0x7f);
    if ((value & 0x80) === 0) {
      complete = true;
      break;
    }
  }
  if (!complete || cursor + length > end) {
    throw new Error(`${label} descriptor is truncated.`);
  }
  return {
    tag,
    dataStart: cursor,
    end: cursor + length,
    next: cursor + length,
  };
}

class BitReader {
  private bit = 0;

  constructor(private readonly bytes: Uint8Array) {}

  read(length: number): number {
    let value = 0;
    for (let index = 0; index < length; index += 1) {
      if (this.bit >= this.bytes.length * 8) {
        throw new Error("AAC AudioSpecificConfig is truncated.");
      }
      const byte = this.bytes[Math.floor(this.bit / 8)] ?? 0;
      value = value * 2 + ((byte >> (7 - (this.bit % 8))) & 1);
      this.bit += 1;
    }
    return value;
  }
}

function parseAacSpecificConfig(config: Uint8Array): AacFacts {
  const bits = new BitReader(config);
  let objectType = bits.read(5);
  if (objectType === 31) objectType = 32 + bits.read(6);
  const frequencyIndex = bits.read(4);
  const sampleRate =
    frequencyIndex === 15
      ? bits.read(24)
      : (AAC_SAMPLE_RATES[frequencyIndex] ?? 0);
  const channelConfiguration = bits.read(4);
  if (objectType !== 2) {
    throw new Error("H3 video soundtracks require AAC-LC audio.");
  }
  if (sampleRate === 0) {
    throw new Error("AAC soundtrack sample rate is unsupported.");
  }
  if (channelConfiguration !== 1 && channelConfiguration !== 2) {
    throw new Error("H3 accepts mono or stereo AAC soundtracks.");
  }
  const shortFrame = bits.read(1);
  if (shortFrame !== 0) {
    throw new Error(
      "960-sample AAC frames are unsupported by the host decoder.",
    );
  }
  return {
    sampleRate,
    channels: channelConfiguration,
    samplesPerPacket: 1_024,
  };
}

function parseAacFacts(bytes: Uint8Array, mp4a: IsoBox): AacFacts {
  const esds = parseSingleBox(
    bytes,
    mp4a.dataStart + 28,
    mp4a.end,
    "AAC elementary stream descriptor",
  );
  if (esds.type !== "esds") {
    throw new Error("AAC soundtrack is missing its esds descriptor.");
  }
  let cursor = esds.dataStart + 4;
  const es = descriptor(bytes, cursor, esds.end, "AAC ES");
  if (es.tag !== 0x03) throw new Error("AAC ES descriptor is missing.");
  requireRange(bytes, es.dataStart, 3, "AAC ES descriptor");
  if ((bytes[es.dataStart + 2] ?? 0) !== 0) {
    throw new Error("AAC ES descriptor flags are unsupported.");
  }
  cursor = es.dataStart + 3;
  let decoderConfig: Descriptor | null = null;
  while (cursor < es.end) {
    const value = descriptor(bytes, cursor, es.end, "AAC decoder");
    if (value.tag === 0x04) {
      decoderConfig = value;
      break;
    }
    cursor = value.next;
  }
  if (!decoderConfig) throw new Error("AAC decoder descriptor is missing.");
  requireRange(bytes, decoderConfig.dataStart, 13, "AAC decoder descriptor");
  if ((bytes[decoderConfig.dataStart] ?? 0) !== 0x40) {
    throw new Error("H3 video soundtracks require MPEG-4 AAC audio.");
  }
  cursor = decoderConfig.dataStart + 13;
  let specific: Descriptor | null = null;
  while (cursor < decoderConfig.end) {
    const value = descriptor(bytes, cursor, decoderConfig.end, "AAC config");
    if (value.tag === 0x05) {
      specific = value;
      break;
    }
    cursor = value.next;
  }
  if (!specific) throw new Error("AAC AudioSpecificConfig is missing.");
  return parseAacSpecificConfig(
    bytes.subarray(specific.dataStart, specific.end),
  );
}

function fragmentSampleCounts(
  bytes: Uint8Array,
  topLevel: readonly IsoBox[],
  trackIds: ReadonlySet<number>,
): Map<number, number> {
  const counts = new Map<number, number>();
  for (const moof of topLevel.filter((box) => box.type === "moof")) {
    const moofBoxes = parseBoxes(bytes, moof.dataStart, moof.end, "moof");
    requiredBox(moofBoxes, "mfhd");
    for (const traf of moofBoxes.filter((box) => box.type === "traf")) {
      const trafBoxes = parseBoxes(bytes, traf.dataStart, traf.end, "traf");
      const tfhd = requiredBox(trafBoxes, "tfhd");
      const trackId = u32be(bytes, tfhd.dataStart + 4, "fragment track id");
      if (!trackIds.has(trackId)) {
        throw new Error(`MP4 fragment refers to unknown track ${trackId}.`);
      }
      const truns = trafBoxes.filter((box) => box.type === "trun");
      const trun = truns.at(-1);
      const sampleCount = trun
        ? u32be(bytes, trun.dataStart + 4, "fragment sample count")
        : 0;
      if (trun) {
        const flags = fullBoxFlags(bytes, trun);
        const fixedFields = 4 + (flags & 0x01 ? 4 : 0) + (flags & 0x04 ? 4 : 0);
        const fieldsPerSample =
          (flags & 0x100 ? 4 : 0) +
          (flags & 0x200 ? 4 : 0) +
          (flags & 0x400 ? 4 : 0) +
          (flags & 0x800 ? 4 : 0);
        if (
          trun.dataStart + 4 + fixedFields + sampleCount * fieldsPerSample >
          trun.end
        ) {
          throw new Error("MP4 fragment sample table is truncated.");
        }
      }
      const next = (counts.get(trackId) ?? 0) + sampleCount;
      if (!Number.isSafeInteger(next)) {
        throw new Error("MP4 fragment sample count is too large.");
      }
      counts.set(trackId, next);
    }
  }
  return counts;
}

function trackDurationMs(track: IsoTrack): number {
  return safeNumber(
    (track.duration * 1_000n) / BigInt(track.timescale),
    "MP4 track duration",
  );
}

function parseMp4Facts(bytes: Uint8Array): MinimaxH3Mp4Facts {
  const topLevel = parseBoxes(bytes, 0, bytes.length, "MP4");
  requiredBox(topLevel, "ftyp");
  const moov = requiredBox(topLevel, "moov");
  const movieBoxes = parseBoxes(bytes, moov.dataStart, moov.end, "moov");
  requiredBox(movieBoxes, "mvhd");
  const tracks = movieBoxes
    .filter((box) => box.type === "trak")
    .map((track) => parseTrack(bytes, track));
  const trackIds = new Set<number>();
  for (const track of tracks) {
    if (trackIds.has(track.id)) {
      throw new Error(`MP4 contains duplicate track id ${track.id}.`);
    }
    trackIds.add(track.id);
  }
  const fragmentCounts = fragmentSampleCounts(bytes, topLevel, trackIds);
  const countFor = (track: IsoTrack): number =>
    fragmentCounts.has(track.id)
      ? (fragmentCounts.get(track.id) ?? 0)
      : track.sampleCount;

  const videoTracks = tracks.filter((track) => track.handler === "vide");
  if (videoTracks.length !== 1) {
    throw new Error("H3 references require exactly one MP4 video track.");
  }
  const video = videoTracks[0];
  if (!video) throw new Error("MP4 video track is missing.");
  if (video.sampleEntry.type !== "avc1") {
    throw new Error("H3 video references require H.264/AVC video.");
  }
  validateAvcConfiguration(bytes, video.sampleEntry);
  const width = u16be(
    bytes,
    video.sampleEntry.dataStart + 24,
    "MP4 video width",
  );
  const height = u16be(
    bytes,
    video.sampleEntry.dataStart + 26,
    "MP4 video height",
  );
  if (width === 0 || height === 0) {
    throw new Error("MP4 video dimensions are invalid.");
  }
  const frameCount = countFor(video);
  if (frameCount === 0) throw new Error("MP4 video contains no frames.");
  const durationMs = trackDurationMs(video);
  if (durationMs === 0) throw new Error("MP4 video duration is invalid.");
  const fps = Math.floor((frameCount * 1_000) / durationMs);
  if (fps === 0) throw new Error("MP4 video frame rate is invalid.");

  const audioTracks = tracks.filter((track) => track.handler === "soun");
  if (audioTracks.length > 1) {
    throw new Error("H3 references accept at most one MP4 soundtrack.");
  }
  const audio = audioTracks[0];
  if (!audio) {
    return {
      mimeType: "video/mp4",
      width,
      height,
      frameCount,
      durationMs,
      fps,
      hasAudio: false,
      audioDurationMs: null,
      audioSampleCount: null,
      audioSampleRate: null,
      audioChannels: null,
    };
  }
  if (audio.sampleEntry.type !== "mp4a") {
    throw new Error("H3 video soundtracks require AAC audio.");
  }
  const aac = parseAacFacts(bytes, audio.sampleEntry);
  const packetCount = countFor(audio);
  if (packetCount === 0) throw new Error("MP4 soundtrack contains no samples.");
  // This is only an authoring/session hint. A malformed packet can be skipped
  // by the host decoder, so packetCount * 1024 is deliberately never retained
  // in the final generation request without canonical upload replacement.
  const audioSampleCount = packetCount * aac.samplesPerPacket;
  if (!Number.isSafeInteger(audioSampleCount)) {
    throw new Error("MP4 soundtrack sample count is too large.");
  }
  return {
    mimeType: "video/mp4",
    width,
    height,
    frameCount,
    durationMs,
    fps,
    hasAudio: true,
    audioDurationMs: Math.ceil((audioSampleCount * 1_000) / aac.sampleRate),
    audioSampleCount,
    audioSampleRate: aac.sampleRate,
    audioChannels: aac.channels,
  };
}

export function probeMinimaxH3Wav(
  input: ArrayBuffer | Uint8Array,
): MinimaxH3WavFacts {
  return parseWavFacts(bytesOf(input));
}

export function probeMinimaxH3Mp4(
  input: ArrayBuffer | Uint8Array,
): MinimaxH3Mp4Facts {
  return parseMp4Facts(bytesOf(input));
}
