//! Presentation only: retain original errors in logs and durable records.
//! The shared fixture contract is also consumed by Swift and TypeScript.
use serde::Serializer;

pub fn bytes(value: u64) -> String {
    let (scale, unit) = if value >= 1_000_000_000_000 {
        (1_000_000_000_000., "TB")
    } else if value >= 1_000_000_000 {
        (1_000_000_000., "GB")
    } else if value >= 1_000_000 {
        (1_000_000., "MB")
    } else if value >= 1_000 {
        (1_000., "KB")
    } else {
        return format!("{value} B");
    };
    let amount = format!("{:.2}", value as f64 / scale);
    format!(
        "{} {unit}",
        amount.trim_end_matches('0').trim_end_matches('.')
    )
}

fn number_after(raw: &str, marker: &str) -> Option<u64> {
    raw.split_once(marker)?
        .1
        .split_whitespace()
        .next()?
        .parse()
        .ok()
}

/// A compatibility boundary for existing engine diagnostics. New domain
/// failures should carry typed codes; never use this prose to decide admission.
pub fn message(raw: &str) -> String {
    let raw = raw.trim();
    let lower = raw.to_ascii_lowercase();
    if lower.contains("cuda") && (lower.contains("restart") || lower.contains("quarantined")) {
        return "The graphics device needs to recover. Restart Mold on that machine before trying again.".into();
    }
    if let Some(summary) = legacy_memory_summary(&lower) {
        return summary;
    }
    if lower.contains("cuda") && lower.contains("cooldown") {
        return "The machine is recovering from a memory failure. Wait for the cooldown or try a smaller output size.".into();
    }
    if lower.contains("requires cuda compute capability") {
        return "This graphics device cannot run that model. Choose another machine or model."
            .into();
    }
    if lower.contains("lora adapter is no longer readable") {
        return "A LoRA file is missing or unreadable. Restore it on the machine, then retry the job.".into();
    }
    if lower.contains("ffprobe is required") || lower.contains("ffmpeg is required") {
        return "Video processing tools are missing on the machine. Install ffmpeg and ffprobe, then try again.".into();
    }
    if lower.contains("admission sample")
        && (lower.contains("device bytes")
            || lower.contains("host bytes")
            || lower.contains("unified-memory bytes"))
    {
        if let (Some(required), Some(available)) = (
            number_after(&lower, "needs at least "),
            number_after(&lower, "exceeding the "),
        ) {
            if required > available {
                let resource = if lower.contains("device bytes") {
                    "graphics"
                } else if lower.contains("unified-memory bytes") {
                    "shared"
                } else {
                    "system"
                };
                return format!("Not enough {resource} memory. Estimated need: {}; available budget: {} ({} short). Close apps on that machine or try a smaller model or output size.", bytes(required), bytes(available), bytes(required - available));
            }
        }
    }
    if [
        "out of memory",
        "out_of_memory",
        "insufficient memory",
        "allocation failed",
        "memory allocation",
        "cuda_error_oom",
    ]
    .iter()
    .any(|s| lower.contains(s))
    {
        return "The machine ran out of memory. Try a smaller model, output size or batch.".into();
    }
    if lower.contains("no space left on device") {
        return "The machine is out of storage. Free some disk space and try again.".into();
    }
    if lower.contains("no such file or directory") {
        return "A required file is missing on the machine. Restore or download the model again."
            .into();
    }
    if lower.contains("permission denied") {
        return "The machine cannot access a required file. Check its file permissions.".into();
    }
    if lower.contains("no device could produce an execution plan") {
        return "This machine cannot run those settings. Try another machine, model or output size.".into();
    }
    if [
        "tensor shape",
        "tensor mismatch",
        "tensor(",
        "dtype",
        "shape mismatch",
    ]
    .iter()
    .any(|s| lower.contains(s))
    {
        return "The model’s data did not match what the renderer expected. Try another model or report this job’s failure details.".into();
    }
    if lower.contains("safetensors error") {
        return "The model file could not be read. Download that model again on the machine, then retry.".into();
    }
    if lower.contains("failed to authenticate the reviewed minimax h3 turbo adapter") {
        return "The H3 Turbo model file could not be verified. Check its installation on that machine, then retry or move the job.".into();
    }
    if lower.contains("minimax h3 preparation evidence was rejected") {
        return "The model could not be prepared on this machine. Check its installation or move the job to another machine.".into();
    }
    if lower.contains("cuda") || lower.contains("metal error") {
        return "The graphics device could not finish the render. Retry the job or move it to another machine.".into();
    }
    if lower.contains("backtrace") || lower.contains("panic") {
        return "Mold encountered an internal error while handling the request.".into();
    }
    if raw.is_empty() || raw.contains('\n') || raw.chars().count() > 240 {
        return "Mold encountered an unexpected error and could not complete the request.".into();
    }
    readable_byte_tokens(raw)
}

fn scaled_prefix(raw: &str) -> Option<u64> {
    let mut parts = raw.split_whitespace();
    let amount: f64 = parts.next()?.trim_start_matches('~').parse().ok()?;
    let unit = parts
        .next()?
        .trim_matches(|c: char| !c.is_ascii_alphabetic());
    let scale = match unit {
        "tb" => 1e12,
        "gb" => 1e9,
        "mb" => 1e6,
        "kb" => 1e3,
        "bytes" | "byte" => 1.,
        _ => return None,
    };
    let value = amount * scale;
    (value.is_finite() && value >= 0. && value <= u64::MAX as f64).then(|| value.round() as u64)
}

fn scaled_after(raw: &str, marker: &str) -> Option<u64> {
    scaled_prefix(raw.split_once(marker)?.1)
}

fn legacy_memory_summary(raw: &str) -> Option<String> {
    if !(raw.contains("needs more host memory")
        || raw.contains("needs more device memory")
        || raw.contains("effective vram capacity"))
    {
        return None;
    }
    let required = scaled_after(raw, "requires ").or_else(|| scaled_after(raw, "needs ~"))?;
    let available = scaled_after(raw, "over this request's ~")
        .or_else(|| scaled_after(raw, "over the "))
        .or_else(|| scaled_prefix(raw.split_once("requires ")?.1.split_once(',')?.1))?;
    if required <= available {
        return None;
    }
    let resource = if raw.contains("host memory") {
        "system"
    } else if raw.contains("metal:") || raw.contains("unified-memory") {
        "shared"
    } else {
        "graphics"
    };
    let advice = if raw.contains("cooldown") {
        "Wait for the cooldown or try a smaller output size."
    } else {
        "Close apps on that machine or try a smaller model or output size."
    };
    Some(format!("Not enough {resource} memory. Estimated need: {}; available budget: {} ({} short). {advice}", bytes(required), bytes(available), bytes(required - available)))
}

fn readable_byte_tokens(raw: &str) -> String {
    let data = raw.as_bytes();
    let mut result = String::new();
    let mut copied = 0;
    let mut i = 0;
    while i < data.len() {
        if data[i].is_ascii_digit()
            && (i == 0 || (!data[i - 1].is_ascii_alphanumeric() && data[i - 1] != b'.'))
        {
            let start = i;
            while i < data.len() && data[i].is_ascii_digit() {
                i += 1;
            }
            let end = i;
            let rest = &raw[end..];
            let suffix = if rest.starts_with(" bytes") {
                6
            } else if rest.starts_with(" byte") {
                5
            } else {
                0
            };
            if suffix > 0
                && data
                    .get(end + suffix)
                    .is_none_or(|c| !c.is_ascii_alphanumeric())
            {
                if let Ok(value) = raw[start..end].parse::<u64>() {
                    result.push_str(&raw[copied..start]);
                    result.push_str(&bytes(value));
                    copied = end + suffix;
                    i = copied;
                }
            }
        } else {
            i += 1;
        }
    }
    result.push_str(&raw[copied..]);
    result
}

pub fn serialize<S: Serializer>(raw: &str, serializer: S) -> Result<S::Ok, S::Error> {
    serializer.serialize_str(&message(raw))
}

pub fn serialize_optional<S: Serializer>(
    raw: &Option<String>,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    match raw {
        Some(raw) => serializer.serialize_some(&message(raw)),
        None => serializer.serialize_none(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cross_surface_contract() {
        let fixtures: serde_json::Value =
            serde_json::from_str(include_str!("../../../docs/contracts/user-errors.json")).unwrap();
        for fixture in fixtures.as_array().unwrap() {
            let raw = fixture["raw"].as_str().unwrap();
            let expected = fixture["message"].as_str().unwrap();
            assert_eq!(message(raw), expected, "{raw}");
            assert_eq!(message(expected), expected);
        }
    }
    #[test]
    fn byte_boundaries_and_non_memory_numbers() {
        assert_eq!(bytes(0), "0 B");
        assert_eq!(bytes(999), "999 B");
        assert_eq!(bytes(1000), "1 KB");
        assert_eq!(
            message("Seed 22683045704, size 512×512, 16 frames"),
            "Seed 22683045704, size 512×512, 16 frames"
        );
    }
    #[test]
    fn wire_errors_preserve_codes_lifecycle_and_original_values() {
        let raw = "tensor shape mismatch: [1, 512]";
        let event = crate::SseErrorEvent::failed_with_code(raw, Some("MODEL_NOT_FOUND".into()));
        let encoded = serde_json::to_value(&event).unwrap();
        assert_eq!(encoded["message"], message(raw));
        assert_eq!(encoded["code"], "MODEL_NOT_FOUND");
        assert_eq!(event.message, raw);
        let retained = crate::SseErrorEvent::retained(raw);
        let encoded = serde_json::to_value(retained).unwrap();
        assert_eq!(encoded["retained"], true);
        assert_eq!(encoded["code"], crate::SSE_ERROR_CODE_SERVER_RESTARTING);
        let download = crate::DownloadEvent::JobFailed {
            id: "download-1".into(),
            error: raw.into(),
        };
        assert_eq!(
            serde_json::to_value(download).unwrap()["error"],
            message(raw)
        );
        let entry = crate::QueueJobEntryWire {
            error: Some(raw.into()),
            held_reason: Some(raw.into()),
            state: "held".into(),
            retryable: Some(false),
            ..Default::default()
        };
        let encoded = serde_json::to_value(&entry).unwrap();
        assert_eq!(encoded["error"], message(raw));
        assert_eq!(encoded["held_reason"], message(raw));
        assert_eq!(encoded["state"], "held");
        assert_eq!(encoded["retryable"], false);
        assert_eq!(entry.error.as_deref(), Some(raw));
    }
}
