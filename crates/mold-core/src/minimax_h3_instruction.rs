//! The image-alignment instruction an H3 FL2VA prompt opens with.
//!
//! Upstream `h3-prompt-writing/references/base-en.txt` §2.1 (MiniMax-AI/MiniMax-H3
//! @d21241f): a frame-conditioned prompt's FIRST line states where each picture
//! sits on the target timeline, then one blank line, then the three core
//! fields. T2VA has no instruction. Duration video carries the previous clip's
//! last frame into the next clip, so mold — not the author — introduced that
//! picture and must say so.

const I2VA: &str = "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.";
const ALIGN: &str = "How the reference pictures align with the target video";

/// Replace any leading alignment instruction with the one this clip's own
/// endpoints need. `first`/`last` say whether the clip renders with an opening
/// and/or closing picture; `frames`/`fps` give its effective duration.
pub fn with_frame_reference_instruction(
    prompt: &str,
    first: bool,
    last: bool,
    frames: u32,
    fps: u32,
) -> String {
    let body = strip_instruction(prompt);
    let end = format!("{:.2}", f64::from(frames) / f64::from(fps.max(1)));
    let shot = final_shot(body);
    let instruction = match (first, last) {
        (false, false) => return body.to_owned(),
        (true, false) => I2VA.to_owned(),
        (true, true) => format!(
            "{ALIGN} — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot {shot}) aligns with the {end}-second mark of the target video."
        ),
        (false, true) => format!(
            "{ALIGN} — <Picture 1> (from [Shot {shot}]) aligns with the {end}-second mark of the target video."
        ),
    };
    format!("{instruction}\n\n{body}")
}

fn strip_instruction(prompt: &str) -> &str {
    let mut rest = prompt.trim_start();
    while rest.starts_with("For the target video,") || rest.starts_with(ALIGN) {
        rest = rest
            .split_once('\n')
            .map_or("", |(_, tail)| tail)
            .trim_start();
    }
    rest
}

/// The index of the last `[Shot n]` marker, or 1 when the prompt has none.
fn final_shot(body: &str) -> u32 {
    body.match_indices("[Shot ")
        .filter_map(|(at, marker)| {
            let digits = &body[at + marker.len()..];
            let end = digits.find(']')?;
            digits[..end].trim().parse::<u32>().ok()
        })
        .max()
        .unwrap_or(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    const BODY: &str = "integrated_multimodal_description: [Shot 1] A cat walks. [Shot 2] At 00:04.000, the camera cuts to a door.\n\noverall_soundscape: Rain.\n\nnon_diegetic_music: N/A";

    #[test]
    fn a_carried_first_frame_gets_the_upstream_i2va_line() {
        let prompt = with_frame_reference_instruction(BODY, true, false, 345, 24);
        assert_eq!(prompt, format!("{I2VA}\n\n{BODY}"));
    }

    #[test]
    fn an_authored_instruction_is_replaced_never_duplicated() {
        let authored = format!("{I2VA}\n\n{BODY}");
        assert_eq!(
            with_frame_reference_instruction(&authored, true, false, 345, 24),
            authored
        );
        assert_eq!(
            with_frame_reference_instruction(&authored, false, false, 345, 24),
            BODY
        );
    }

    #[test]
    fn first_and_last_name_the_final_shot_and_the_clip_end() {
        let prompt = with_frame_reference_instruction(BODY, true, true, 345, 24);
        assert!(prompt.starts_with(
            "How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot 2) aligns with the 14.38-second mark of the target video.\n\n"
        ), "{prompt}");
        let last = with_frame_reference_instruction(BODY, false, true, 124, 24);
        assert!(last.starts_with(
            "How the reference pictures align with the target video — <Picture 1> (from [Shot 2]) aligns with the 5.17-second mark of the target video.\n\n"
        ), "{last}");
    }
}
