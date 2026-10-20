//! Allocation fences and sequential packet access for retained MP4 files.
use anyhow::{ensure, Context, Result};
use mp4_rs::{Mp4Reader, Mp4Sample, Mp4Track};
use std::{
    fs::File,
    io::{BufReader, Read, Seek, SeekFrom},
};

const MAX_METADATA_BYTES: u64 = 256 * 1024 * 1024;
const MAX_PACKET_BYTES: u32 = 64 * 1024 * 1024;

fn u32_at(file: &mut File, offset: u64) -> Result<u32> {
    file.seek(SeekFrom::Start(offset))?;
    let mut bytes = [0; 4];
    file.read_exact(&mut bytes)?;
    Ok(u32::from_be_bytes(bytes))
}

/// Inspect box extents/counts before mp4-rs allocates vectors. In particular,
/// sample table counts must fit their OWN box, not merely the enclosing file.
fn boxes(
    file: &mut File,
    mut position: u64,
    end: u64,
    depth: u32,
    metadata: &mut u64,
    checkpoint: &mut dyn FnMut() -> Result<()>,
) -> Result<()> {
    ensure!(depth <= 12, "MP4 box nesting exceeds the parser budget");
    while position < end {
        checkpoint()?;
        ensure!(end - position >= 8, "truncated MP4 box header");
        file.seek(SeekFrom::Start(position))?;
        let mut header = [0; 8];
        file.read_exact(&mut header)?;
        let short = u32::from_be_bytes(header[..4].try_into()?);
        let (size, header_size) = match short {
            0 => (end - position, 8),
            1 => {
                let mut large = [0; 8];
                file.read_exact(&mut large)?;
                (u64::from_be_bytes(large), 16)
            }
            size => (u64::from(size), 8),
        };
        ensure!(
            size >= header_size && size <= end - position,
            "MP4 box extends outside its container"
        );
        let kind = &header[4..8];
        let body = position + header_size;
        let payload = size - header_size;
        if depth == 0 && kind != b"mdat" {
            *metadata = metadata
                .checked_add(size)
                .context("MP4 metadata size overflowed")?;
            ensure!(
                *metadata <= MAX_METADATA_BYTES,
                "MP4 metadata exceeds its bounded parsing budget"
            );
        }
        match kind {
            b"emsg" | b"udta" | b"meta" | b"mvex" => {
                anyhow::bail!("source metadata requires normalization before bounded assembly")
            }
            b"hev1" | b"hvc1" | b"vp09" | b"tx3g" => {
                anyhow::bail!("source codec requires normalization before bounded assembly")
            }
            b"stsd" => {
                ensure!(
                    payload >= 8 && u32_at(file, body + 4)? == 1,
                    "assembly requires one decoder configuration per track"
                );
                boxes(
                    file,
                    body + 8,
                    position + size,
                    depth + 1,
                    metadata,
                    checkpoint,
                )?;
            }
            b"avc1" => {
                ensure!(
                    payload >= 78 && size <= 4 * 1024 * 1024,
                    "invalid or oversized AVC sample description"
                );
                boxes(
                    file,
                    body + 78,
                    position + size,
                    depth + 1,
                    metadata,
                    checkpoint,
                )?;
            }
            b"mp4a" => {
                ensure!(
                    payload >= 28 && size <= 64 * 1024,
                    "invalid or oversized AAC sample description"
                );
                boxes(
                    file,
                    body + 28,
                    position + size,
                    depth + 1,
                    metadata,
                    checkpoint,
                )?;
            }
            b"moov" | b"trak" | b"mdia" | b"minf" | b"stbl" | b"dinf" | b"edts" => {
                boxes(file, body, position + size, depth + 1, metadata, checkpoint)?
            }
            b"stts" | b"ctts" | b"stsc" | b"stco" | b"co64" | b"stss" | b"elst" => {
                ensure!(payload >= 8, "truncated MP4 sample table");
                let count = u64::from(u32_at(file, body + 4)?);
                let stride = match kind {
                    b"stsc" => 12,
                    b"stco" | b"stss" => 4,
                    b"elst" => {
                        if u32_at(file, body)? >> 24 == 1 {
                            20
                        } else {
                            12
                        }
                    }
                    _ => 8,
                };
                ensure!(
                    count <= (payload - 8) / stride,
                    "MP4 table count exceeds its box"
                );
            }
            b"stsz" => {
                ensure!(payload >= 12, "truncated MP4 packet-size table");
                let fixed = u32_at(file, body + 4)?;
                let count = u64::from(u32_at(file, body + 8)?);
                ensure!(
                    fixed <= MAX_PACKET_BYTES,
                    "MP4 packet exceeds its allocation budget"
                );
                if fixed == 0 {
                    ensure!(
                        count <= (payload - 12) / 4,
                        "MP4 packet count exceeds its box"
                    );
                    file.seek(SeekFrom::Start(body + 12))?;
                    let mut buffer = [0; 64 * 1024];
                    let mut remaining = count;
                    while remaining > 0 {
                        checkpoint()?;
                        let entries = remaining.min((buffer.len() / 4) as u64) as usize;
                        file.read_exact(&mut buffer[..entries * 4])?;
                        for value in buffer[..entries * 4].as_chunks::<4>().0 {
                            ensure!(
                                u32::from_be_bytes(*value) <= MAX_PACKET_BYTES,
                                "MP4 packet exceeds its allocation budget"
                            );
                        }
                        remaining -= entries as u64;
                    }
                }
            }
            b"moof" => anyhow::bail!("fragmented source requires decoded assembly"),
            _ => {}
        }
        position += size;
    }
    Ok(())
}

pub(crate) fn checked_reader(
    mut file: File,
    checkpoint: &mut dyn FnMut() -> Result<()>,
) -> Result<Mp4Reader<BufReader<File>>> {
    let metadata = file.metadata()?;
    ensure!(
        metadata.is_file(),
        "MP4 input must be a regular retained file"
    );
    boxes(&mut file, 0, metadata.len(), 0, &mut 0, checkpoint)?;
    file.seek(SeekFrom::Start(0))?;
    Mp4Reader::read_header(BufReader::new(file), metadata.len())
        .context("reading bounded MP4 metadata")
}

/// mp4-rs rescans sizes from a chunk's beginning for each sample. The original
/// encoder writes one large chunk, so using read_sample on a long prefix is
/// quadratic. This cursor visits each sample and chunk-table entry once.
pub(crate) struct PacketCursor<'a> {
    track: &'a Mp4Track,
    file: File,
    length: u64,
    sample: u32,
    chunk: u32,
    chunk_remaining: u32,
    chunk_entry: usize,
    offset: u64,
    time_entry: usize,
    time_remaining: u32,
    timestamp: u64,
}

impl<'a> PacketCursor<'a> {
    pub(crate) fn new(track: &'a Mp4Track, file: File) -> Result<Self> {
        ensure!(
            track.trafs.is_empty(),
            "fragmented source requires decoded assembly"
        );
        let length = file.metadata()?.len();
        let table = &track.trak.mdia.minf.stbl;
        ensure!(
            table
                .ctts
                .as_ref()
                .is_none_or(|table| table.entries.iter().all(|entry| entry.sample_offset == 0)),
            "reordered pictures require decoded assembly"
        );
        ensure!(
            table
                .stsc
                .entries
                .first()
                .is_some_and(|entry| entry.first_chunk == 1)
                && table
                    .stsc
                    .entries
                    .iter()
                    .all(|entry| entry.samples_per_chunk > 0 && entry.sample_description_index == 1)
                && table
                    .stsc
                    .entries
                    .windows(2)
                    .all(|pair| pair[0].first_chunk < pair[1].first_chunk),
            "invalid MP4 chunk mapping"
        );
        ensure!(
            !table.stts.entries.is_empty()
                && table
                    .stts
                    .entries
                    .iter()
                    .all(|entry| entry.sample_count > 0 && entry.sample_delta > 0),
            "invalid MP4 sample timeline"
        );
        Ok(Self {
            track,
            file,
            length,
            sample: 0,
            chunk: 0,
            chunk_remaining: 0,
            chunk_entry: 0,
            offset: 0,
            time_entry: 0,
            time_remaining: 0,
            timestamp: 0,
        })
    }

    pub(crate) fn next(&mut self) -> Result<Option<Mp4Sample>> {
        let Some((mut sample, offset, size)) = self.next_location(true)? else {
            return Ok(None);
        };
        self.file.seek(SeekFrom::Start(offset))?;
        let mut bytes = vec![0; size as usize];
        self.file.read_exact(&mut bytes)?;
        sample.bytes = bytes.into();
        Ok(Some(sample))
    }

    pub(crate) fn skip_to(
        &mut self,
        sample_index: u32,
        checkpoint: &mut dyn FnMut() -> Result<()>,
    ) -> Result<()> {
        ensure!(
            sample_index >= self.sample && sample_index <= self.track.sample_count(),
            "invalid sequential video seek"
        );
        while self.sample < sample_index {
            if self.sample % 4096 == 0 {
                checkpoint()?;
            }
            self.next_location(false)?
                .context("video seek exceeded sample table")?;
        }
        Ok(())
    }

    fn next_location(&mut self, include_sync: bool) -> Result<Option<(Mp4Sample, u64, u32)>> {
        if self.sample >= self.track.sample_count() {
            return Ok(None);
        }
        let table = &self.track.trak.mdia.minf.stbl;
        if self.chunk_remaining == 0 {
            self.chunk = self
                .chunk
                .checked_add(1)
                .context("MP4 chunk count overflowed")?;
            while table
                .stsc
                .entries
                .get(self.chunk_entry + 1)
                .is_some_and(|entry| entry.first_chunk <= self.chunk)
            {
                self.chunk_entry += 1;
            }
            let entry = &table.stsc.entries[self.chunk_entry];
            self.chunk_remaining = entry.samples_per_chunk;
            self.offset = if let Some(offsets) = &table.co64 {
                *offsets
                    .entries
                    .get((self.chunk - 1) as usize)
                    .context("missing MP4 chunk offset")?
            } else {
                u64::from(
                    *table
                        .stco
                        .as_ref()
                        .context("missing MP4 chunk offsets")?
                        .entries
                        .get((self.chunk - 1) as usize)
                        .context("missing MP4 chunk offset")?,
                )
            };
        }
        if self.time_remaining == 0 {
            let entry = table
                .stts
                .entries
                .get(self.time_entry)
                .context("missing MP4 sample timing")?;
            self.time_remaining = entry.sample_count;
            self.time_entry += 1;
        }
        let duration = table.stts.entries[self.time_entry - 1].sample_delta;
        let size = if table.stsz.sample_size > 0 {
            table.stsz.sample_size
        } else {
            *table
                .stsz
                .sample_sizes
                .get(self.sample as usize)
                .context("missing MP4 packet size")?
        };
        ensure!(
            size > 0 && size <= MAX_PACKET_BYTES,
            "MP4 packet exceeds its allocation budget"
        );
        let end = self
            .offset
            .checked_add(u64::from(size))
            .context("MP4 packet offset overflowed")?;
        ensure!(
            end <= self.length,
            "MP4 packet extends beyond its retained file"
        );
        let offset = self.offset;
        self.sample += 1;
        self.chunk_remaining -= 1;
        self.time_remaining -= 1;
        let sample = Mp4Sample {
            start_time: self.timestamp,
            duration,
            rendering_offset: 0,
            is_sync: include_sync
                && table
                    .stss
                    .as_ref()
                    .is_none_or(|sync| sync.entries.binary_search(&self.sample).is_ok()),
            bytes: Vec::new().into(),
        };
        self.timestamp = self
            .timestamp
            .checked_add(u64::from(duration))
            .context("MP4 timestamp overflowed")?;
        self.offset = end;
        Ok(Some((sample, offset, size)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn atom(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&u32::try_from(payload.len() + 8).unwrap().to_be_bytes());
        bytes.extend_from_slice(kind);
        bytes.extend_from_slice(payload);
        bytes
    }

    #[test]
    fn refuses_oversized_packets_and_false_table_counts_before_allocation() {
        for (fixed, count, tail, expected) in [
            (u32::MAX, 1, vec![], "allocation budget"),
            (0, u32::MAX, vec![], "packet count"),
            (0, 1, u32::MAX.to_be_bytes().to_vec(), "allocation budget"),
        ] {
            let mut table = vec![0; 4];
            table.extend_from_slice(&fixed.to_be_bytes());
            table.extend_from_slice(&count.to_be_bytes());
            table.extend_from_slice(&tail);
            let bytes = atom(
                b"moov",
                &atom(b"trak", &atom(b"stbl", &atom(b"stsz", &table))),
            );
            let mut file = tempfile::tempfile().unwrap();
            std::io::Write::write_all(&mut file, &bytes).unwrap();
            assert!(checked_reader(file, &mut || Ok(()))
                .unwrap_err()
                .to_string()
                .contains(expected));
        }
    }

    #[test]
    fn rejects_overflowing_extents_and_excessive_nesting() {
        let mut bytes = vec![0, 0, 0, 1, b'm', b'o', b'o', b'v'];
        bytes.extend_from_slice(&u64::MAX.to_be_bytes());
        let mut nested = Vec::new();
        for _ in 0..14 {
            nested = atom(b"moov", &nested);
        }
        for input in [bytes, nested] {
            let mut file = tempfile::tempfile().unwrap();
            std::io::Write::write_all(&mut file, &input).unwrap();
            assert!(checked_reader(file, &mut || Ok(())).is_err());
        }
    }

    #[test]
    fn unused_metadata_never_reaches_an_unbounded_metadata_parser() {
        for kind in [b"emsg", b"udta", b"meta", b"mvex"] {
            let mut file = tempfile::tempfile().unwrap();
            std::io::Write::write_all(&mut file, &atom(b"moov", &atom(kind, &[]))).unwrap();
            assert!(checked_reader(file, &mut || Ok(()))
                .unwrap_err()
                .to_string()
                .contains("normalization"));
        }
    }

    #[test]
    fn linear_cursor_preserves_every_packet_in_a_single_large_chunk() {
        use image::{Rgb, RgbImage};
        let frames = vec![RgbImage::from_pixel(32, 32, Rgb([17, 50, 90])); 100];
        let mut file = tempfile::tempfile().unwrap();
        std::io::Write::write_all(
            &mut file,
            &crate::ltx_video::video_enc::encode_mp4(&frames, 24).unwrap(),
        )
        .unwrap();
        let mut oracle = checked_reader(file.try_clone().unwrap(), &mut || Ok(())).unwrap();
        let parsed = checked_reader(file.try_clone().unwrap(), &mut || Ok(())).unwrap();
        let mut cursor = PacketCursor::new(&parsed.tracks()[&1], file).unwrap();
        for index in 1..=100 {
            let actual = cursor.next().unwrap().unwrap();
            let expected = oracle.read_sample(1, index).unwrap().unwrap();
            assert_eq!(actual.bytes, expected.bytes);
            assert_eq!(actual.start_time, expected.start_time);
            assert_eq!(actual.duration, expected.duration);
            assert_eq!(actual.is_sync, expected.is_sync);
        }
        assert!(cursor.next().unwrap().is_none());
    }
}
