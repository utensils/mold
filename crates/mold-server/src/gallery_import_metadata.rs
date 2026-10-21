//! Exact-output mirror compatibility for completion-only archive facts.
use mold_core::OutputMetadata;

pub(crate) fn matches(embedded: &OutputMetadata, archive: &OutputMetadata) -> bool {
    let mut left = embedded.clone();
    let mut right = archive.clone();
    if left.job_id.is_none() || right.job_id.is_none() {
        left.job_id = None;
        right.job_id = None;
    }
    if left.generation_time_ms.is_none() || right.generation_time_ms.is_none() {
        left.generation_time_ms = None;
        right.generation_time_ms = None;
    }
    if left.has_alpha.is_none() || right.has_alpha.is_none() {
        left.has_alpha = None;
        right.has_alpha = None;
    }
    let left_base = base_version(&left.version);
    let right_base = base_version(&right.version);
    if left_base == right_base && (left.version == left_base || right.version == right_base) {
        let version = left_base.to_owned();
        left.version = version.clone();
        right.version = version;
    }
    left == right
}

fn base_version(version: &str) -> &str {
    let Some((base, suffix)) = version.rsplit_once(" (") else {
        return version;
    };
    let Some(suffix) = suffix.strip_suffix(')') else {
        return version;
    };
    let Some((revision, date)) = suffix.split_once(' ') else {
        return version;
    };
    let valid_revision = (7..=40).contains(&revision.len())
        && revision
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte));
    let valid_date = date.len() == 10
        && date.bytes().enumerate().all(|(index, byte)| {
            if index == 4 || index == 7 {
                byte == b'-'
            } else {
                byte.is_ascii_digit()
            }
        });
    if valid_revision && valid_date {
        base
    } else {
        version
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metadata() -> OutputMetadata {
        serde_json::from_value(serde_json::json!({
            "prompt":"literal", "model":"video", "seed":u64::MAX,
            "width":768, "height":512, "steps":3, "guidance":0.0, "version":"0.36.0"
        }))
        .unwrap()
    }

    #[test]
    fn completion_facts_do_not_discard_the_full_archive_recipe() {
        let embedded = metadata();
        let mut archive = embedded.clone();
        archive.job_id = Some("completed-job".into());
        archive.generation_time_ms = Some(1234);
        archive.has_alpha = Some(false);
        archive.version = "0.36.0 (abcdef012345 2026-10-20)".into();
        assert!(matches(&embedded, &archive));
        assert!(matches(&archive, &embedded));
        assert_eq!(archive.seed, u64::MAX);
        assert_eq!(archive.job_id.as_deref(), Some("completed-job"));
    }

    #[test]
    fn present_conflicts_and_inference_facts_remain_significant() {
        let mut first = metadata();
        first.job_id = Some("first".into());
        first.generation_time_ms = Some(10);
        first.has_alpha = Some(false);
        for change in [
            "job",
            "time",
            "alpha",
            "seed",
            "prompt",
            "model",
            "scheduler",
            "version",
        ] {
            let mut second = first.clone();
            match change {
                "job" => second.job_id = Some("second".into()),
                "time" => second.generation_time_ms = Some(11),
                "alpha" => second.has_alpha = Some(true),
                "seed" => second.seed -= 1,
                "prompt" => second.prompt.push(' '),
                "model" => second.model.push('x'),
                "scheduler" => second.scheduler = Some(mold_core::Scheduler::Euler),
                "version" => second.version = "0.37.0".into(),
                _ => unreachable!(),
            }
            assert!(!matches(&first, &second), "{change}");
        }
    }

    #[test]
    fn video_authoring_and_artifact_bindings_are_never_completion_facts() {
        let mut first = metadata();
        first.video_job = Some(serde_json::from_value(serde_json::json!({
            "id":"66687089-6429-4a48-8a5d-f302c8889092", "manifest_sha256":"a".repeat(64),
            "clip_count":"2", "part_count":"2", "added_frames":"144", "source_frames":"0", "total_frames":"144", "join_mode":"continuous",
            "authoring":{"duration_ms":"6000","boundary_frames":124,"seed":"18446744073709551615"},
            "artifacts":{"version":1,"bundle_id":"b".repeat(64),"provenance":{"sha256":"c".repeat(64),"size_bytes":"100"}}
        })).unwrap());
        for change in ["authoring", "artifact", "missing"] {
            let mut second = first.clone();
            match change {
                "authoring" => {
                    second
                        .video_job
                        .as_mut()
                        .unwrap()
                        .authoring
                        .as_mut()
                        .unwrap()
                        .duration_ms += 1
                }
                "artifact" => {
                    second
                        .video_job
                        .as_mut()
                        .unwrap()
                        .artifacts
                        .as_mut()
                        .unwrap()
                        .provenance
                        .sha256 = "d".repeat(64)
                }
                "missing" => second.video_job = None,
                _ => unreachable!(),
            }
            assert!(!matches(&first, &second), "{change}");
        }
    }

    #[test]
    fn build_suffix_must_be_valid_and_cannot_hide_different_revisions() {
        let mut first = metadata();
        let mut second = first.clone();
        for suffix in [
            " (ABCDEF0 2026-10-20)",
            " (abc 2026-10-20)",
            " (abcdef0 yesterday)",
            " (abcdef0 2026-1-20)",
        ] {
            second.version = format!("0.36.0{suffix}");
            assert!(!matches(&first, &second), "{suffix}");
        }
        first.version = "0.36.0 (abcdef0 2026-10-20)".into();
        second.version = "0.36.0 (1234567 2026-10-20)".into();
        assert!(!matches(&first, &second));
    }
}
