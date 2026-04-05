//! Video encoding utilities for LTX Video output.

use anyhow::{Context, Result};
use image::RgbImage;

/// Generation metadata embedded as tEXt chunks in APNG output.
pub struct VideoMetadata {
    pub prompt: String,
    pub model: String,
    pub seed: u64,
    pub steps: u32,
    pub guidance: f64,
    pub width: u32,
    pub height: u32,
    pub frames: u32,
    pub fps: u32,
}

/// Encode a sequence of RGB frames into an animated GIF.
///
/// Uses per-frame NeuQuant palette quantization (256 colors).
pub fn encode_gif(frames: &[RgbImage], fps: u32) -> Result<Vec<u8>> {
    anyhow::ensure!(!frames.is_empty(), "no frames to encode");

    let (width, height) = (frames[0].width() as u16, frames[0].height() as u16);
    let delay_cs = (100.0 / fps as f64).round() as u16;

    let mut buf = Vec::new();
    {
        let mut encoder = gif::Encoder::new(&mut buf, width, height, &[])
            .context("failed to create GIF encoder")?;
        encoder
            .set_repeat(gif::Repeat::Infinite)
            .context("failed to set GIF repeat")?;

        for frame_img in frames {
            let rgba: image::RgbaImage =
                image::DynamicImage::ImageRgb8(frame_img.clone()).into_rgba8();
            let mut pixels = rgba.into_raw();

            let mut gif_frame = gif::Frame::from_rgba_speed(width, height, &mut pixels, 10);
            gif_frame.delay = delay_cs;
            gif_frame.dispose = gif::DisposalMethod::Any;

            encoder
                .write_frame(&gif_frame)
                .context("failed to write GIF frame")?;
        }
    }
    Ok(buf)
}

/// Extract the first frame as a PNG thumbnail.
pub fn first_frame_png(frames: &[RgbImage]) -> Result<Vec<u8>> {
    anyhow::ensure!(!frames.is_empty(), "no frames for thumbnail");

    let mut buf = std::io::Cursor::new(Vec::new());
    frames[0]
        .write_to(&mut buf, image::ImageFormat::Png)
        .context("failed to encode thumbnail PNG")?;
    Ok(buf.into_inner())
}

/// Encode a sequence of RGB frames into an animated PNG (APNG).
///
/// Loops infinitely. Optionally embeds generation metadata as tEXt/iTXt chunks.
pub fn encode_apng(
    frames: &[RgbImage],
    fps: u32,
    metadata: Option<&VideoMetadata>,
) -> Result<Vec<u8>> {
    anyhow::ensure!(!frames.is_empty(), "no frames to encode");

    let (width, height) = (frames[0].width(), frames[0].height());
    let num_frames = frames.len() as u32;

    let mut buf = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut buf, width, height);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        encoder.set_animated(num_frames, 0)?;
        encoder.set_frame_delay(1, fps as u16)?;

        if let Some(meta) = metadata {
            encoder.add_itxt_chunk("mold:prompt".to_string(), meta.prompt.clone())?;
            encoder.add_itxt_chunk("mold:model".to_string(), meta.model.clone())?;
            encoder.add_text_chunk("mold:seed".to_string(), meta.seed.to_string())?;
            encoder.add_text_chunk("mold:steps".to_string(), meta.steps.to_string())?;
            encoder.add_text_chunk("mold:guidance".to_string(), meta.guidance.to_string())?;
            encoder.add_text_chunk("mold:width".to_string(), meta.width.to_string())?;
            encoder.add_text_chunk("mold:height".to_string(), meta.height.to_string())?;
            encoder.add_text_chunk("mold:frames".to_string(), meta.frames.to_string())?;
            encoder.add_text_chunk("mold:fps".to_string(), meta.fps.to_string())?;
        }

        let mut writer = encoder
            .write_header()
            .context("failed to write APNG header")?;

        for (i, frame) in frames.iter().enumerate() {
            if i > 0 {
                writer.set_blend_op(png::BlendOp::Source)?;
                writer.set_dispose_op(png::DisposeOp::Background)?;
            }
            writer
                .write_image_data(frame.as_raw())
                .with_context(|| format!("failed to write APNG frame {i}"))?;
        }

        writer.finish().context("failed to finalize APNG")?;
    }
    Ok(buf)
}

/// Encode a sequence of RGB frames into an animated WebP.
///
/// Uses the `webp-animation` crate (statically linked libwebp).
#[cfg(feature = "webp")]
pub fn encode_webp(frames: &[RgbImage], fps: u32) -> Result<Vec<u8>> {
    anyhow::ensure!(!frames.is_empty(), "no frames to encode");

    let (width, height) = (frames[0].width(), frames[0].height());
    let frame_duration_ms = (1000.0 / fps as f64).round() as i32;

    let mut encoder = webp_animation::Encoder::new((width, height))
        .map_err(|e| anyhow::anyhow!("failed to create WebP encoder: {e}"))?;

    for (i, frame_img) in frames.iter().enumerate() {
        let rgba: image::RgbaImage = image::DynamicImage::ImageRgb8(frame_img.clone()).into_rgba8();
        let timestamp_ms = i as i32 * frame_duration_ms;
        encoder
            .add_frame(rgba.as_raw(), timestamp_ms)
            .map_err(|e| anyhow::anyhow!("failed to add WebP frame {i}: {e}"))?;
    }

    let final_timestamp_ms = frames.len() as i32 * frame_duration_ms;
    let webp_data = encoder
        .finalize(final_timestamp_ms)
        .map_err(|e| anyhow::anyhow!("failed to finalize WebP animation: {e}"))?;

    Ok(webp_data.to_vec())
}

/// Encode a sequence of RGB frames into an MP4 (H.264/AVC) video.
///
/// Uses OpenH264 for H.264 encoding with a minimal QuickTime-compatible MP4 muxer.
/// Produces `ftyp[isom,iso2,avc1,mp41] + moov + mdat` (faststart layout).
#[cfg(feature = "mp4")]
pub fn encode_mp4(frames: &[RgbImage], fps: u32) -> Result<Vec<u8>> {
    use openh264::encoder::{EncoderConfig, FrameRate, VuiConfig};
    use openh264::formats::{RgbSliceU8, YUVBuffer};

    anyhow::ensure!(!frames.is_empty(), "no frames to encode");

    let (width, height) = (frames[0].width(), frames[0].height());

    let config = EncoderConfig::new()
        .max_frame_rate(FrameRate::from_hz(fps as f32))
        .bitrate(openh264::encoder::BitRate::from_bps(10_000_000))
        .rate_control_mode(openh264::encoder::RateControlMode::Quality)
        .profile(openh264::encoder::Profile::High)
        .vui(VuiConfig::bt601()); // BT.601 limited range — matches YUVBuffer::from_rgb_source() conversion

    let api = openh264::OpenH264API::from_source();
    let mut h264 = openh264::encoder::Encoder::with_api_config(api, config)
        .context("failed to create H.264 encoder")?;

    // Encode all frames, collecting NAL units and keyframe flags
    let mut samples: Vec<(Vec<u8>, bool)> = Vec::with_capacity(frames.len());
    let mut sps: Option<Vec<u8>> = None;
    let mut pps: Option<Vec<u8>> = None;

    for frame in frames {
        let rgb = RgbSliceU8::new(frame.as_raw(), (width as usize, height as usize));
        let yuv = YUVBuffer::from_rgb_source(rgb);
        let bitstream = h264.encode(&yuv).context("failed to encode H.264 frame")?;
        let is_key = matches!(bitstream.frame_type(), openh264::encoder::FrameType::IDR);

        // Parse Annex B byte stream into NALs, extract SPS/PPS, convert rest to
        // length-prefixed format for MP4.
        let annex_b = bitstream.to_vec();
        let mut frame_nals = Vec::new();
        for nal in split_annex_b_nals(&annex_b) {
            if nal.is_empty() {
                continue;
            }
            let nal_type = nal[0] & 0x1F;
            match nal_type {
                7 => sps = Some(nal.to_vec()),
                8 => pps = Some(nal.to_vec()),
                _ => {
                    let len = nal.len() as u32;
                    frame_nals.extend_from_slice(&len.to_be_bytes());
                    frame_nals.extend_from_slice(nal);
                }
            }
        }
        if !frame_nals.is_empty() {
            samples.push((frame_nals, is_key));
        }
    }

    let sps = sps.context("H.264 encoder produced no SPS")?;
    let pps = pps.context("H.264 encoder produced no PPS")?;

    /// Split Annex B byte stream (00 00 00 01 or 00 00 01 delimited) into NAL units.
    fn split_annex_b_nals(data: &[u8]) -> Vec<&[u8]> {
        let mut nals = Vec::new();
        let mut i = 0;
        while i < data.len() {
            // Find start code (00 00 00 01 or 00 00 01)
            let sc_len = if i + 4 <= data.len() && data[i..i + 4] == [0, 0, 0, 1] {
                4
            } else if i + 3 <= data.len() && data[i..i + 3] == [0, 0, 1] {
                3
            } else {
                i += 1;
                continue;
            };
            let nal_start = i + sc_len;
            // Find next start code or end
            let mut nal_end = data.len();
            let mut j = nal_start;
            while j + 3 <= data.len() {
                if data[j..j + 3] == [0, 0, 1]
                    || (j + 4 <= data.len() && data[j..j + 4] == [0, 0, 0, 1])
                {
                    // Trim trailing zeros that are part of the start code
                    nal_end = j;
                    while nal_end > nal_start && data[nal_end - 1] == 0 {
                        nal_end -= 1;
                    }
                    break;
                }
                j += 1;
            }
            if nal_start < nal_end {
                nals.push(&data[nal_start..nal_end]);
            }
            i = if j < data.len() { j } else { data.len() };
        }
        nals
    }

    // Build QuickTime-compatible MP4: ftyp + moov + mdat (faststart)
    mp4_mux::write_mp4(&samples, &sps, &pps, width, height, fps)
}

/// Minimal MP4 muxer producing QuickTime/macOS-compatible output.
///
/// Writes ftyp(isom,iso2,avc1,mp41) + moov(mvhd,trak(tkhd,edts,mdia(mdhd,hdlr,minf(vmhd,dinf,stbl)))) + mdat.
#[cfg(feature = "mp4")]
mod mp4_mux {
    use anyhow::Result;

    fn write_u16(buf: &mut Vec<u8>, v: u16) {
        buf.extend_from_slice(&v.to_be_bytes());
    }
    fn write_u32(buf: &mut Vec<u8>, v: u32) {
        buf.extend_from_slice(&v.to_be_bytes());
    }

    /// Write an MP4 box: size(4) + type(4) + content
    fn write_box(buf: &mut Vec<u8>, box_type: &[u8; 4], content: &[u8]) {
        write_u32(buf, (8 + content.len()) as u32);
        buf.extend_from_slice(box_type);
        buf.extend_from_slice(content);
    }

    fn build_ftyp() -> Vec<u8> {
        let mut b = Vec::new();
        let mut content = Vec::new();
        content.extend_from_slice(b"isom"); // major_brand
        write_u32(&mut content, 0x200); // minor_version
        content.extend_from_slice(b"isom"); // compatible brands
        content.extend_from_slice(b"iso2");
        content.extend_from_slice(b"avc1");
        content.extend_from_slice(b"mp41");
        write_box(&mut b, b"ftyp", &content);
        b
    }

    fn build_mvhd(duration_ticks: u32, timescale: u32) -> Vec<u8> {
        let mut c = Vec::new();
        write_u32(&mut c, 0); // version + flags
        write_u32(&mut c, 0); // creation_time
        write_u32(&mut c, 0); // modification_time
        write_u32(&mut c, timescale); // timescale
        write_u32(&mut c, duration_ticks); // duration
        write_u32(&mut c, 0x0001_0000); // rate (1.0 fixed point)
        write_u16(&mut c, 0x0100); // volume (1.0 fixed point)
        c.extend_from_slice(&[0u8; 10]); // reserved
                                         // Unity matrix (identity 3x3)
        for &v in &[0x0001_0000u32, 0, 0, 0, 0x0001_0000, 0, 0, 0, 0x4000_0000] {
            write_u32(&mut c, v);
        }
        c.extend_from_slice(&[0u8; 24]); // pre_defined
        write_u32(&mut c, 2); // next_track_ID
        let mut b = Vec::new();
        write_box(&mut b, b"mvhd", &c);
        b
    }

    fn build_avc_c(sps: &[u8], pps: &[u8]) -> Vec<u8> {
        let profile_idc = if sps.len() > 1 { sps[1] } else { 0x42 };
        let compat = if sps.len() > 2 { sps[2] } else { 0xC0 };
        let level_idc = if sps.len() > 3 { sps[3] } else { 0x1E };

        let mut c = Vec::new();
        c.push(1); // configurationVersion
        c.push(profile_idc);
        c.push(compat);
        c.push(level_idc);
        c.push(0xFF); // lengthSizeMinusOne = 3 (4-byte NAL lengths)
        c.push(0xE1); // numOfSequenceParameterSets = 1
        write_u16(&mut c, sps.len() as u16);
        c.extend_from_slice(sps);
        c.push(1); // numOfPictureParameterSets = 1
        write_u16(&mut c, pps.len() as u16);
        c.extend_from_slice(pps);
        let mut b = Vec::new();
        write_box(&mut b, b"avcC", &c);
        b
    }

    pub fn write_mp4(
        samples: &[(Vec<u8>, bool)],
        sps: &[u8],
        pps: &[u8],
        width: u32,
        height: u32,
        fps: u32,
    ) -> Result<Vec<u8>> {
        let timescale = fps * 1000;
        let sample_duration = 1000u32; // each frame = 1000 ticks at timescale = fps*1000
        let duration_ticks = samples.len() as u32 * sample_duration;

        // Compute sample sizes and total mdat size
        let sample_sizes: Vec<u32> = samples.iter().map(|(d, _)| d.len() as u32).collect();
        let mdat_payload: usize = sample_sizes.iter().map(|&s| s as usize).sum();

        // Build stbl children
        let mut stsd_content = Vec::new();
        write_u32(&mut stsd_content, 0); // version + flags
        write_u32(&mut stsd_content, 1); // entry_count
        {
            // avc1 visual sample entry
            let mut avc1 = Vec::new();
            avc1.extend_from_slice(&[0u8; 6]); // reserved
            write_u16(&mut avc1, 1); // data_reference_index
            avc1.extend_from_slice(&[0u8; 16]); // pre_defined + reserved
            write_u16(&mut avc1, width as u16);
            write_u16(&mut avc1, height as u16);
            write_u32(&mut avc1, 0x0048_0000); // horizresolution (72 dpi)
            write_u32(&mut avc1, 0x0048_0000); // vertresolution (72 dpi)
            write_u32(&mut avc1, 0); // reserved
            write_u16(&mut avc1, 1); // frame_count
            avc1.extend_from_slice(&[0u8; 32]); // compressorname
            write_u16(&mut avc1, 0x0018); // depth (24-bit)
            write_u16(&mut avc1, 0xFFFF); // pre_defined (-1)
                                          // avcC box
            avc1.extend_from_slice(&build_avc_c(sps, pps));
            // colr box (BT.601/SMPTE 170M — matches YUVBuffer::from_rgb_source() conversion)
            let mut colr = Vec::new();
            colr.extend_from_slice(b"nclx");
            write_u16(&mut colr, 6); // colour_primaries (SMPTE 170M / BT.601)
            write_u16(&mut colr, 6); // transfer_characteristics (SMPTE 170M)
            write_u16(&mut colr, 6); // matrix_coefficients (SMPTE 170M / BT.601)
            colr.push(0x00); // full_range_flag=0 (limited range, matches SPS VUI)
            write_box(&mut avc1, b"colr", &colr);
            // pasp box (square pixels)
            let mut pasp = Vec::new();
            write_u32(&mut pasp, 1); // hSpacing
            write_u32(&mut pasp, 1); // vSpacing
            write_box(&mut avc1, b"pasp", &pasp);

            write_box(&mut stsd_content, b"avc1", &avc1);
        }

        // stts: sample-to-time (all frames same duration)
        let mut stts = Vec::new();
        write_u32(&mut stts, 0); // version + flags
        write_u32(&mut stts, 1); // entry_count
        write_u32(&mut stts, samples.len() as u32);
        write_u32(&mut stts, sample_duration);

        // stsc: sample-to-chunk (1 chunk with all samples)
        let mut stsc = Vec::new();
        write_u32(&mut stsc, 0); // version + flags
        write_u32(&mut stsc, 1); // entry_count
        write_u32(&mut stsc, 1); // first_chunk
        write_u32(&mut stsc, samples.len() as u32); // samples_per_chunk
        write_u32(&mut stsc, 1); // sample_description_index

        // stsz: per-sample sizes
        let mut stsz = Vec::new();
        write_u32(&mut stsz, 0); // version + flags
        write_u32(&mut stsz, 0); // sample_size (0 = variable)
        write_u32(&mut stsz, sample_sizes.len() as u32);
        for &sz in &sample_sizes {
            write_u32(&mut stsz, sz);
        }

        // stss: sync sample table (keyframes)
        let keyframes: Vec<u32> = samples
            .iter()
            .enumerate()
            .filter(|(_, (_, is_key))| *is_key)
            .map(|(i, _)| (i + 1) as u32) // 1-indexed
            .collect();
        let mut stss = Vec::new();
        write_u32(&mut stss, 0); // version + flags
        write_u32(&mut stss, keyframes.len() as u32);
        for &kf in &keyframes {
            write_u32(&mut stss, kf);
        }

        // Build moov with stco pointing at correct mdat offset.
        // Two-pass: first to measure moov size, second with correct offset.
        let build_moov = |mdat_offset: u32| -> Vec<u8> {
            let mut stco = Vec::new();
            write_u32(&mut stco, 0); // version + flags
            write_u32(&mut stco, 1); // entry_count
            write_u32(&mut stco, mdat_offset + 8); // offset to mdat payload

            let mut stbl = Vec::new();
            write_box(&mut stbl, b"stsd", &stsd_content);
            write_box(&mut stbl, b"stts", &stts);
            write_box(&mut stbl, b"stsc", &stsc);
            write_box(&mut stbl, b"stsz", &stsz);
            write_box(&mut stbl, b"stco", &stco);
            write_box(&mut stbl, b"stss", &stss);

            let mut dinf = Vec::new();
            {
                // ISO 14496-12 §8.7.2: dinf → dref → url entries
                let mut dref_payload = Vec::new();
                write_u32(&mut dref_payload, 0); // version + flags
                write_u32(&mut dref_payload, 1); // entry_count
                write_box(&mut dref_payload, b"url ", &[0, 0, 0, 1]); // self-contained flag
                let mut dinf_content = Vec::new();
                write_box(&mut dinf_content, b"dref", &dref_payload);
                write_box(&mut dinf, b"dinf", &dinf_content);
            }

            let mut vmhd = Vec::new();
            write_u32(&mut vmhd, 1); // version=0 + flags=1
            vmhd.extend_from_slice(&[0u8; 8]); // graphicsmode + opcolor

            let mut minf = Vec::new();
            write_box(&mut minf, b"vmhd", &vmhd);
            minf.extend_from_slice(&dinf);
            write_box(&mut minf, b"stbl", &stbl);

            let mut hdlr = Vec::new();
            write_u32(&mut hdlr, 0); // version + flags
            write_u32(&mut hdlr, 0); // pre_defined
            hdlr.extend_from_slice(b"vide"); // handler_type
            hdlr.extend_from_slice(&[0u8; 12]); // reserved
            hdlr.extend_from_slice(b"VideoHandler\0");

            let mut mdhd = Vec::new();
            write_u32(&mut mdhd, 0); // version + flags
            write_u32(&mut mdhd, 0); // creation_time
            write_u32(&mut mdhd, 0); // modification_time
            write_u32(&mut mdhd, timescale); // timescale
            write_u32(&mut mdhd, duration_ticks); // duration
            write_u32(&mut mdhd, 0x55C40000); // language (und) + pre_defined

            let mut mdia = Vec::new();
            write_box(&mut mdia, b"mdhd", &mdhd);
            write_box(&mut mdia, b"hdlr", &hdlr);
            write_box(&mut mdia, b"minf", &minf);

            let mut tkhd = Vec::new();
            write_u32(&mut tkhd, 3); // version=0 + flags=3 (enabled+in_movie)
            write_u32(&mut tkhd, 0); // creation_time
            write_u32(&mut tkhd, 0); // modification_time
            write_u32(&mut tkhd, 1); // track_ID
            write_u32(&mut tkhd, 0); // reserved
            write_u32(&mut tkhd, duration_ticks); // duration (in mvhd timescale)
            tkhd.extend_from_slice(&[0u8; 8]); // reserved
            write_u16(&mut tkhd, 0); // layer
            write_u16(&mut tkhd, 0); // alternate_group
            write_u16(&mut tkhd, 0); // volume
            write_u16(&mut tkhd, 0); // reserved
            for &v in &[0x0001_0000u32, 0, 0, 0, 0x0001_0000, 0, 0, 0, 0x4000_0000] {
                write_u32(&mut tkhd, v);
            }
            write_u32(&mut tkhd, width << 16); // width (fixed point)
            write_u32(&mut tkhd, height << 16); // height (fixed point)

            // edts/elst: identity edit list (playback starts at media time 0)
            let mut elst = Vec::new();
            write_u32(&mut elst, 0); // version + flags
            write_u32(&mut elst, 1); // entry_count
            write_u32(&mut elst, duration_ticks); // segment_duration
            write_u32(&mut elst, 0); // media_time (start at 0)
            write_u32(&mut elst, 0x0001_0000); // media_rate (1.0 fixed point)
            let mut edts = Vec::new();
            write_box(&mut edts, b"elst", &elst);

            let mut trak = Vec::new();
            write_box(&mut trak, b"tkhd", &tkhd);
            write_box(&mut trak, b"edts", &edts);
            write_box(&mut trak, b"mdia", &mdia);

            let mut moov = Vec::new();
            moov.extend_from_slice(&build_mvhd(duration_ticks, timescale));
            write_box(&mut moov, b"trak", &trak);

            let mut out = Vec::new();
            write_box(&mut out, b"moov", &moov);
            out
        };

        // First pass: build moov with placeholder to measure its size
        let ftyp = build_ftyp();
        let moov_pass1 = build_moov(0);
        let mdat_offset = (ftyp.len() + moov_pass1.len()) as u32;

        // Second pass: build moov with correct mdat offset
        let moov = build_moov(mdat_offset);
        anyhow::ensure!(
            moov.len() == moov_pass1.len(),
            "moov size changed between passes ({} vs {})",
            moov.len(),
            moov_pass1.len()
        );

        // Build mdat (reject if payload exceeds u32 box size limit)
        let mdat_total = 8 + mdat_payload;
        anyhow::ensure!(
            mdat_total <= u32::MAX as usize,
            "MP4 mdat exceeds 4 GB limit ({} bytes) — reduce frames or resolution",
            mdat_total
        );
        let mut mdat = Vec::with_capacity(mdat_total);
        write_u32(&mut mdat, mdat_total as u32);
        mdat.extend_from_slice(b"mdat");
        for (data, _) in samples {
            mdat.extend_from_slice(data);
        }

        // Assemble: ftyp + moov + mdat
        let mut out = Vec::with_capacity(ftyp.len() + moov.len() + mdat.len());
        out.extend_from_slice(&ftyp);
        out.extend_from_slice(&moov);
        out.extend_from_slice(&mdat);

        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a simple test frame with a gradient pattern.
    fn test_frame(width: u32, height: u32, seed: u8) -> RgbImage {
        let mut img = RgbImage::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let r = ((x as f32 / width as f32) * 255.0) as u8;
                let g = ((y as f32 / height as f32) * 255.0) as u8;
                let b = seed.wrapping_mul(37).wrapping_add((x ^ y) as u8);
                img.put_pixel(x, y, image::Rgb([r, g, b]));
            }
        }
        img
    }

    fn test_frames(width: u32, height: u32, count: usize) -> Vec<RgbImage> {
        (0..count)
            .map(|i| test_frame(width, height, i as u8))
            .collect()
    }

    #[test]
    fn gif_encodes_valid_output() {
        let frames = test_frames(64, 64, 3);
        let data = encode_gif(&frames, 10).unwrap();
        assert!(
            data.len() > 100,
            "GIF output too small: {} bytes",
            data.len()
        );
        assert_eq!(&data[..6], b"GIF89a"); // GIF89a magic
    }

    #[test]
    fn gif_empty_frames_rejected() {
        assert!(encode_gif(&[], 24).is_err());
    }

    #[test]
    fn apng_encodes_valid_output() {
        let frames = test_frames(64, 64, 3);
        let data = encode_apng(&frames, 10, None).unwrap();
        assert!(
            data.len() > 100,
            "APNG output too small: {} bytes",
            data.len()
        );
        assert_eq!(&data[..8], &[137, 80, 78, 71, 13, 10, 26, 10]); // PNG magic
    }

    #[test]
    fn apng_with_metadata() {
        let frames = test_frames(64, 64, 2);
        let meta = VideoMetadata {
            prompt: "a test prompt".to_string(),
            model: "test-model".to_string(),
            seed: 42,
            steps: 30,
            guidance: 3.0,
            width: 64,
            height: 64,
            frames: 2,
            fps: 10,
        };
        let data = encode_apng(&frames, 10, Some(&meta)).unwrap();
        assert!(data.len() > 100);
        // Metadata should be embedded — the file should be larger than without
        let data_no_meta = encode_apng(&frames, 10, None).unwrap();
        assert!(
            data.len() > data_no_meta.len(),
            "metadata should increase file size"
        );
    }

    #[test]
    fn apng_empty_frames_rejected() {
        assert!(encode_apng(&[], 24, None).is_err());
    }

    #[test]
    fn first_frame_png_works() {
        let frames = test_frames(32, 32, 3);
        let data = first_frame_png(&frames).unwrap();
        assert_eq!(&data[..8], &[137, 80, 78, 71, 13, 10, 26, 10]); // PNG magic
    }

    #[cfg(feature = "webp")]
    #[test]
    fn webp_encodes_valid_output() {
        let frames = test_frames(64, 64, 3);
        let data = encode_webp(&frames, 10).unwrap();
        assert!(
            data.len() > 100,
            "WebP output too small: {} bytes",
            data.len()
        );
        assert_eq!(&data[..4], b"RIFF"); // WebP RIFF magic
    }

    #[cfg(feature = "mp4")]
    #[test]
    fn mp4_encodes_valid_output() {
        let frames = test_frames(64, 64, 3);
        let data = encode_mp4(&frames, 10).unwrap();
        // MP4 should have ftyp box
        assert!(
            data.len() > 1000,
            "MP4 output too small: {} bytes",
            data.len()
        );
        let ftyp = &data[4..8];
        assert_eq!(ftyp, b"ftyp", "MP4 should start with ftyp box");
    }

    #[cfg(feature = "mp4")]
    #[test]
    fn mp4_768x512_reasonable_size() {
        // Regression: 768x512 at 2 Mbps produced only ~10KB/frame
        let frames = test_frames(768, 512, 3);
        let data = encode_mp4(&frames, 24).unwrap();
        // At quality mode with 10 Mbps, 3 frames should be at least 50KB
        assert!(
            data.len() > 50_000,
            "MP4 768x512 output too small: {} bytes (expected >50KB)",
            data.len()
        );
    }

    #[cfg(feature = "mp4")]
    mod mp4_quicktime_compat {
        use super::*;

        /// Find a named MP4 box in a byte slice, returning its content (after the 8-byte header).
        fn find_box(data: &[u8], name: &[u8; 4]) -> Option<(usize, Vec<u8>)> {
            let mut pos = 0;
            while pos + 8 <= data.len() {
                let size =
                    u32::from_be_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]])
                        as usize;
                if size < 8 || pos + size > data.len() {
                    break;
                }
                if &data[pos + 4..pos + 8] == name {
                    return Some((pos, data[pos + 8..pos + size].to_vec()));
                }
                pos += size;
            }
            None
        }

        #[test]
        fn colr_atom_declares_limited_range() {
            let frames = test_frames(64, 64, 3);
            let data = encode_mp4(&frames, 10).unwrap();

            // Navigate: ftyp -> moov -> trak -> ... find colr inside the stsd/avc1
            // The colr box is inside avc1, which is inside stsd, inside stbl, inside minf,
            // inside mdia, inside trak, inside moov. Search for "colr" in the raw bytes.
            let colr_tag = b"colr";
            let colr_pos = data
                .windows(4)
                .position(|w| w == colr_tag)
                .expect("colr box not found in MP4 output");

            // colr box layout: size(4) + "colr"(4) + "nclx"(4) + primaries(2) + transfer(2) + matrix(2) + full_range(1)
            // The "colr" we found is at offset colr_pos, which is the type field (bytes 4-7 of the box).
            // So the box starts at colr_pos - 4.
            let nclx_start = colr_pos + 4; // after "colr" type
            assert_eq!(
                &data[nclx_start..nclx_start + 4],
                b"nclx",
                "colr should use nclx type"
            );

            // full_range_flag is the last byte of the colr nclx content
            let full_range_offset = nclx_start + 4 + 2 + 2 + 2; // nclx + primaries + transfer + matrix
            assert_eq!(
                data[full_range_offset], 0x00,
                "colr full_range_flag should be 0 (limited range) for QuickTime compatibility"
            );

            // Verify BT.601/SMPTE 170M primaries/transfer/matrix (value 6)
            let primaries = u16::from_be_bytes([data[nclx_start + 4], data[nclx_start + 5]]);
            let transfer = u16::from_be_bytes([data[nclx_start + 6], data[nclx_start + 7]]);
            let matrix = u16::from_be_bytes([data[nclx_start + 8], data[nclx_start + 9]]);
            assert_eq!(primaries, 6, "colour_primaries should be SMPTE 170M");
            assert_eq!(transfer, 6, "transfer_characteristics should be SMPTE 170M");
            assert_eq!(matrix, 6, "matrix_coefficients should be SMPTE 170M");
        }

        #[test]
        fn edts_elst_box_present() {
            let frames = test_frames(64, 64, 3);
            let data = encode_mp4(&frames, 10).unwrap();

            // Find moov box
            let moov = find_box(data.as_slice(), b"moov")
                .expect("moov box not found")
                .1;

            // Find trak inside moov
            let trak = find_box(&moov, b"trak").expect("trak box not found").1;

            // Find edts inside trak
            let edts = find_box(&trak, b"edts")
                .expect("edts box not found in trak")
                .1;

            // Find elst inside edts
            let elst = find_box(&edts, b"elst")
                .expect("elst box not found in edts")
                .1;

            // elst content: version(4) + entry_count(4) + segment_duration(4) + media_time(4) + media_rate(4)
            assert!(
                elst.len() >= 20,
                "elst content too short: {} bytes",
                elst.len()
            );
            let entry_count = u32::from_be_bytes([elst[4], elst[5], elst[6], elst[7]]);
            assert_eq!(entry_count, 1, "elst should have exactly 1 entry");
            let media_time = u32::from_be_bytes([elst[12], elst[13], elst[14], elst[15]]);
            assert_eq!(media_time, 0, "media_time should be 0");
            let media_rate = u32::from_be_bytes([elst[16], elst[17], elst[18], elst[19]]);
            assert_eq!(
                media_rate, 0x0001_0000,
                "media_rate should be 1.0 (fixed point)"
            );
        }

        #[test]
        fn sps_contains_vui_parameters() {
            let frames = test_frames(64, 64, 3);
            let data = encode_mp4(&frames, 10).unwrap();

            // Find avcC box — search for "avcC" in the raw bytes
            let avcc_tag = b"avcC";
            let avcc_pos = data
                .windows(4)
                .position(|w| w == avcc_tag)
                .expect("avcC box not found");

            // avcC box: size(4) + "avcC"(4) + configurationVersion(1) + profile(1) + compat(1) + level(1)
            //           + lengthSizeMinusOne(1) + numSPS(1) + spsLen(2) + sps_data...
            let box_start = avcc_pos - 4;
            let box_size = u32::from_be_bytes([
                data[box_start],
                data[box_start + 1],
                data[box_start + 2],
                data[box_start + 3],
            ]) as usize;
            let content = &data[avcc_pos + 4..box_start + box_size];

            // content[0] = configVersion, [1] = profile, [2] = compat, [3] = level
            // content[4] = 0xFF (lengthSizeMinusOne), [5] = 0xE1 (numSPS)
            // content[6..8] = SPS length (big-endian)
            assert!(content.len() >= 8, "avcC content too short");
            let sps_len = u16::from_be_bytes([content[6], content[7]]) as usize;
            assert!(content.len() >= 8 + sps_len, "avcC too short for SPS data");
            let sps = &content[8..8 + sps_len];

            // An SPS with VUI (colour description) is significantly longer than without.
            // Minimal SPS (no VUI) is typically ~7-10 bytes; with BT.601 VUI colour
            // info it's 15+ bytes. Empirically, openh264 High profile + VuiConfig::bt601()
            // produces ~18-20 byte SPS. Threshold of 15 ensures VUI is present while
            // staying above any non-VUI SPS length.
            assert!(
                sps.len() >= 15,
                "SPS too short ({} bytes) — VUI parameters likely missing (expected >= 15 with colour description)",
                sps.len()
            );
        }
    }
}
