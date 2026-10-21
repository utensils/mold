//! Small presentation-box updates without loading or cloning the movie index.

use anyhow::{ensure, Context, Result};
use std::{
    fs::File,
    io::{Read, Seek, SeekFrom, Write},
};

#[derive(Clone, Copy)]
struct BoxLocation {
    start: u64,
    size: u64,
    header: u64,
    kind: [u8; 4],
}
impl BoxLocation {
    fn body(self) -> u64 {
        self.start + self.header
    }
    fn end(self) -> u64 {
        self.start + self.size
    }
}

fn header(file: &mut File, start: u64, end: u64) -> Result<BoxLocation> {
    ensure!(
        end.checked_sub(start).is_some_and(|left| left >= 8),
        "truncated presentation box"
    );
    file.seek(SeekFrom::Start(start))?;
    let mut bytes = [0; 8];
    file.read_exact(&mut bytes)?;
    let short = u32::from_be_bytes(bytes[..4].try_into()?);
    let (size, header) = match short {
        0 => (end - start, 8),
        1 => {
            let mut large = [0; 8];
            file.read_exact(&mut large)?;
            (u64::from_be_bytes(large), 16)
        }
        value => (u64::from(value), 8),
    };
    ensure!(
        size >= header && size <= end - start,
        "presentation box exceeds its container"
    );
    Ok(BoxLocation {
        start,
        size,
        header,
        kind: bytes[4..].try_into()?,
    })
}

fn child(file: &mut File, parent: BoxLocation, kind: [u8; 4]) -> Result<Option<BoxLocation>> {
    let mut position = parent.body();
    let mut found = None;
    while position < parent.end() {
        let location = header(file, position, parent.end())?;
        if location.kind == kind {
            ensure!(found.is_none(), "duplicate presentation box");
            found = Some(location);
        }
        position = location.end();
    }
    Ok(found)
}

fn movie(file: &mut File) -> Result<BoxLocation> {
    let end = file.metadata()?.len();
    let mut position = 0;
    let mut found = None;
    while position < end {
        let location = header(file, position, end)?;
        if location.kind == *b"moov" {
            ensure!(found.is_none(), "duplicate movie box");
            found = Some(location);
        }
        position = location.end();
    }
    found.context("missing movie box")
}

fn track(file: &mut File, movie: BoxLocation, id: u32) -> Result<BoxLocation> {
    let mut position = movie.body();
    let mut found = None;
    while position < movie.end() {
        let location = header(file, position, movie.end())?;
        if location.kind == *b"trak" {
            let tkhd = child(file, location, *b"tkhd")?.context("missing track header")?;
            file.seek(SeekFrom::Start(tkhd.body()))?;
            let mut version = [0];
            file.read_exact(&mut version)?;
            ensure!(version[0] <= 1, "unsupported track-header version");
            let offset = if version[0] == 0 { 12 } else { 20 };
            ensure!(
                tkhd.size - tkhd.header >= offset + 4,
                "truncated track header"
            );
            file.seek(SeekFrom::Start(tkhd.body() + offset))?;
            let mut bytes = [0; 4];
            file.read_exact(&mut bytes)?;
            if u32::from_be_bytes(bytes) == id {
                ensure!(found.is_none(), "duplicate track identity");
                found = Some(location);
            }
        }
        position = location.end();
    }
    found.context("missing presentation track")
}

pub(crate) fn read_track_box(
    file: &mut File,
    id: u32,
    path: &[[u8; 4]],
    max_bytes: u64,
) -> Result<Vec<u8>> {
    let movie = movie(file)?;
    let mut location = track(file, movie, id)?;
    for kind in path {
        location = child(file, location, *kind)?.context("missing presentation box")?;
    }
    ensure!(
        location.size <= max_bytes,
        "presentation box exceeds its bounded budget"
    );
    file.seek(SeekFrom::Start(location.start))?;
    let mut bytes = vec![0; usize::try_from(location.size)?];
    file.read_exact(&mut bytes)?;
    Ok(bytes)
}

/// Replace a leaf, or append a missing leaf to its existing parent. Generated
/// files place moov last, so sample offsets never change. Only the metadata
/// following this leaf moves, through one fixed 64 KiB buffer.
pub(crate) fn replace_track_box(
    file: &mut File,
    id: u32,
    path: &[[u8; 4]],
    replacement: &[u8],
) -> Result<()> {
    ensure!(
        !path.is_empty() && replacement.len() >= 8,
        "invalid presentation replacement"
    );
    ensure!(
        replacement[4..8] == path[path.len() - 1]
            && usize::try_from(u32::from_be_bytes(replacement[..4].try_into()?))?
                == replacement.len(),
        "replacement must be one complete bounded box"
    );
    let movie = movie(file)?;
    let old_end = file.metadata()?.len();
    ensure!(
        movie.end() == old_end,
        "presentation updates require a final movie box"
    );
    let track = track(file, movie, id)?;
    let mut ancestors = vec![movie, track];
    for kind in &path[..path.len() - 1] {
        let parent = *ancestors.last().unwrap();
        ancestors.push(child(file, parent, *kind)?.context("missing presentation parent")?);
    }
    let parent = *ancestors.last().unwrap();
    let old = child(file, parent, path[path.len() - 1])?;
    replace_leaf(file, ancestors, old, replacement)
}

fn replace_leaf(
    file: &mut File,
    ancestors: Vec<BoxLocation>,
    old: Option<BoxLocation>,
    replacement: &[u8],
) -> Result<()> {
    let old_end = file.metadata()?.len();
    let parent = *ancestors.last().context("missing presentation parent")?;
    let start = old.map_or(parent.end(), |old| old.start);
    let size = old.map_or(0, |old| old.size);
    let tail = start + size;
    let delta = i128::try_from(replacement.len())? - i128::from(size);
    let new_end = u64::try_from(i128::from(old_end) + delta)?;
    for ancestor in &ancestors {
        let new_size = u64::try_from(i128::from(ancestor.size) + delta)?;
        ensure!(
            ancestor.header == 16 || new_size <= u64::from(u32::MAX),
            "presentation box requires a larger size representation"
        );
    }
    let mut buffer = [0; 64 * 1024];
    if delta > 0 {
        file.set_len(new_end)?;
        let mut cursor = old_end;
        while cursor > tail {
            let count = usize::try_from((cursor - tail).min(buffer.len() as u64))?;
            cursor -= count as u64;
            file.seek(SeekFrom::Start(cursor))?;
            file.read_exact(&mut buffer[..count])?;
            file.seek(SeekFrom::Start(u64::try_from(i128::from(cursor) + delta)?))?;
            file.write_all(&buffer[..count])?;
        }
    } else if delta < 0 {
        let mut cursor = tail;
        while cursor < old_end {
            let count = usize::try_from((old_end - cursor).min(buffer.len() as u64))?;
            file.seek(SeekFrom::Start(cursor))?;
            file.read_exact(&mut buffer[..count])?;
            file.seek(SeekFrom::Start(u64::try_from(i128::from(cursor) + delta)?))?;
            file.write_all(&buffer[..count])?;
            cursor += count as u64;
        }
        file.set_len(new_end)?;
    }
    file.seek(SeekFrom::Start(start))?;
    file.write_all(replacement)?;
    for ancestor in ancestors {
        let new_size = u64::try_from(i128::from(ancestor.size) + delta)?;
        file.seek(SeekFrom::Start(
            ancestor.start + if ancestor.header == 16 { 8 } else { 0 },
        ))?;
        if ancestor.header == 16 {
            file.write_all(&new_size.to_be_bytes())?;
        } else {
            file.write_all(&u32::try_from(new_size)?.to_be_bytes())?;
        }
    }
    Ok(())
}

#[cfg(any(feature = "mp4", test))]
pub(crate) fn set_durations(file: &mut File, id: u32, ticks: u64) -> Result<()> {
    for (kind, version0, version1) in [(*b"mvhd", 16, 24), (*b"tkhd", 20, 28)] {
        // The first upgrade may move every following box; resolve fresh each time.
        let movie = movie(file)?;
        ensure!(
            movie.end() == file.metadata()?.len(),
            "duration updates require a final movie box"
        );
        let mut ancestors = vec![movie];
        if kind == *b"tkhd" {
            ancestors.push(track(file, movie, id)?);
        }
        let location = child(file, *ancestors.last().unwrap(), kind)?
            .context("missing presentation header")?;
        file.seek(SeekFrom::Start(location.body()))?;
        let mut version = [0];
        file.read_exact(&mut version)?;
        ensure!(version[0] <= 1, "unsupported presentation duration version");
        let offset = if version[0] == 0 { version0 } else { version1 };
        let length = if version[0] == 0 { 4 } else { 8 };
        ensure!(
            location.size - location.header >= offset + length,
            "truncated presentation duration"
        );
        if version[0] == 0 && ticks > u64::from(u32::MAX) {
            ensure!(
                location.size <= 1024 && location.header == 8,
                "unexpected presentation header size"
            );
            file.seek(SeekFrom::Start(location.body()))?;
            let mut original = vec![0; usize::try_from(location.size - location.header)?];
            file.read_exact(&mut original)?;
            let mut upgraded = Vec::with_capacity(original.len() + 20);
            upgraded.extend_from_slice(&u32::try_from(location.size + 12)?.to_be_bytes());
            upgraded.extend_from_slice(&kind);
            original[0] = 1;
            upgraded.extend_from_slice(&original[..4]);
            for range in [4..8, 8..12] {
                upgraded.extend_from_slice(&[0; 4]);
                upgraded.extend_from_slice(&original[range]);
            }
            upgraded.extend_from_slice(&original[12..version0 as usize]);
            upgraded.extend_from_slice(&ticks.to_be_bytes());
            upgraded.extend_from_slice(&original[version0 as usize + 4..]);
            replace_leaf(file, ancestors, Some(location), &upgraded)?;
            continue;
        }
        file.seek(SeekFrom::Start(location.body() + offset))?;
        if version[0] == 0 {
            file.write_all(&u32::try_from(ticks)?.to_be_bytes())?;
        } else {
            file.write_all(&ticks.to_be_bytes())?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ltx_video::video_enc;
    use image::{Rgb, RgbImage};

    #[test]
    fn presentation_box_replacement_round_trips_without_changing_packets() {
        let frames = vec![RgbImage::from_pixel(32, 32, Rgb([10, 20, 30])); 3];
        let bytes = video_enc::encode_mp4(&frames, 24).unwrap();
        let mut input = tempfile::tempfile().unwrap();
        std::io::Write::write_all(&mut input, &bytes).unwrap();
        let mut file = tempfile::tempfile().unwrap();
        let mut assembly = crate::video_assembly::Mp4FileAssembly::new(file.try_clone().unwrap());
        assembly.append(input, 3, &mut || Ok(())).unwrap();
        assembly.finish().unwrap();
        let path = [*b"mdia", *b"minf", *b"stbl", *b"stsd"];
        let original = read_track_box(&mut file, 1, &path, 1024 * 1024).unwrap();
        replace_track_box(&mut file, 1, &path, &original).unwrap();
        let after = read_track_box(&mut file, 1, &path, 1024 * 1024).unwrap();
        assert_eq!(after, original);
        let checked = crate::video_assembly_input::checked_reader(file, &mut || Ok(())).unwrap();
        assert_eq!(checked.tracks()[&1].sample_count(), 3);
    }

    #[test]
    fn adding_and_shrinking_an_edit_box_updates_only_metadata_lengths() {
        let frames = vec![RgbImage::from_pixel(32, 32, Rgb([10, 20, 30])); 3];
        let mut input = tempfile::tempfile().unwrap();
        input
            .write_all(&video_enc::encode_mp4(&frames, 24).unwrap())
            .unwrap();
        let mut file = tempfile::tempfile().unwrap();
        let mut assembly = crate::video_assembly::Mp4FileAssembly::new(file.try_clone().unwrap());
        assembly.append(input, 3, &mut || Ok(())).unwrap();
        assembly.finish().unwrap();
        let original_size = file.metadata().unwrap().len();
        let mut edit = Vec::new();
        edit.extend_from_slice(&44u32.to_be_bytes());
        edit.extend_from_slice(b"edts");
        edit.extend_from_slice(&36u32.to_be_bytes());
        edit.extend_from_slice(b"elst");
        edit.extend_from_slice(&[1, 0, 0, 0]);
        edit.extend_from_slice(&1u32.to_be_bytes());
        edit.extend_from_slice(&3000u64.to_be_bytes());
        edit.extend_from_slice(&0u64.to_be_bytes());
        edit.extend_from_slice(&1u16.to_be_bytes());
        edit.extend_from_slice(&0u16.to_be_bytes());
        replace_track_box(&mut file, 1, &[*b"edts"], &edit).unwrap();
        assert_eq!(file.metadata().unwrap().len(), original_size + 44);
        assert_eq!(
            read_track_box(&mut file, 1, &[*b"edts"], 100).unwrap(),
            edit
        );
        set_durations(&mut file, 1, 3000).unwrap();
        let checked =
            crate::video_assembly_input::checked_reader(file.try_clone().unwrap(), &mut || Ok(()))
                .unwrap();
        assert_eq!(checked.tracks()[&1].sample_count(), 3);
        assert_eq!(checked.tracks()[&1].trak.tkhd.duration, 3000);
        let mut shorter = Vec::new();
        shorter.extend_from_slice(&36u32.to_be_bytes());
        shorter.extend_from_slice(b"edts");
        shorter.extend_from_slice(&28u32.to_be_bytes());
        shorter.extend_from_slice(b"elst");
        shorter.extend_from_slice(&0u32.to_be_bytes());
        shorter.extend_from_slice(&1u32.to_be_bytes());
        shorter.extend_from_slice(&3000u32.to_be_bytes());
        shorter.extend_from_slice(&0u32.to_be_bytes());
        shorter.extend_from_slice(&1u16.to_be_bytes());
        shorter.extend_from_slice(&0u16.to_be_bytes());
        replace_track_box(&mut file, 1, &[*b"edts"], &shorter).unwrap();
        assert_eq!(file.metadata().unwrap().len(), original_size + 36);
        assert_eq!(
            crate::video_assembly_input::checked_reader(file, &mut || Ok(()))
                .unwrap()
                .tracks()[&1]
                .sample_count(),
            3
        );
    }

    #[test]
    fn duration_headers_upgrade_to_64_bit_without_a_movie_length_cap() {
        let frames = vec![RgbImage::from_pixel(32, 32, Rgb([10, 20, 30])); 3];
        let mut input = tempfile::tempfile().unwrap();
        input
            .write_all(&video_enc::encode_mp4(&frames, 24).unwrap())
            .unwrap();
        let mut file = tempfile::tempfile().unwrap();
        let mut assembly = crate::video_assembly::Mp4FileAssembly::new(file.try_clone().unwrap());
        assembly.append(input, 3, &mut || Ok(())).unwrap();
        assembly.finish().unwrap();
        let ticks = u64::from(u32::MAX) + 100;
        set_durations(&mut file, 1, ticks).unwrap();
        let checked = crate::video_assembly_input::checked_reader(file, &mut || Ok(())).unwrap();
        assert_eq!(checked.moov.mvhd.version, 1);
        assert_eq!(checked.moov.mvhd.duration, ticks);
        assert_eq!(checked.tracks()[&1].trak.tkhd.version, 1);
        assert_eq!(checked.tracks()[&1].trak.tkhd.duration, ticks);
        assert_eq!(checked.tracks()[&1].sample_count(), 3);
    }
}
