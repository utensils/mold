use crate::BatchExecutionCapability;

/// Static, pre-load batch contract for one production inference family.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FamilyBatchCapability {
    pub family: &'static str,
    pub aliases: &'static [&'static str],
    pub execution: BatchExecutionCapability,
}

const SINGLETON: BatchExecutionCapability = BatchExecutionCapability::SINGLETON_COOPERATIVE;

/// Authoritative production family registry.
///
/// Every factory family is explicit here, even when two families currently
/// share one runtime engine. All generation families remain singleton until a
/// measured engine contract returns multiple ordered outputs.
const PRODUCTION_BATCH_CAPABILITIES: &[FamilyBatchCapability] = &[
    FamilyBatchCapability {
        family: "flux",
        aliases: &[],
        execution: SINGLETON,
    },
    FamilyBatchCapability {
        family: "flux2",
        aliases: &["flux.2", "flux2-klein"],
        execution: SINGLETON,
    },
    FamilyBatchCapability {
        family: "sd15",
        aliases: &["sd1.5", "stable-diffusion-1.5"],
        execution: SINGLETON,
    },
    FamilyBatchCapability {
        family: "sdxl",
        aliases: &[],
        execution: SINGLETON,
    },
    FamilyBatchCapability {
        family: "sd3",
        aliases: &["sd3.5", "stable-diffusion-3", "stable-diffusion-3.5"],
        execution: SINGLETON,
    },
    FamilyBatchCapability {
        family: "z-image",
        aliases: &[],
        execution: SINGLETON,
    },
    FamilyBatchCapability {
        family: "qwen-image",
        aliases: &["qwen_image"],
        execution: SINGLETON,
    },
    FamilyBatchCapability {
        family: "qwen-image-edit",
        aliases: &[],
        execution: SINGLETON,
    },
    FamilyBatchCapability {
        family: "ltx-video",
        aliases: &["ltx_video"],
        execution: SINGLETON,
    },
    FamilyBatchCapability {
        family: "ltx2",
        aliases: &["ltx-2", "ltx2.3"],
        execution: SINGLETON,
    },
    FamilyBatchCapability {
        family: "wuerstchen",
        aliases: &["wuerstchen-v2"],
        execution: SINGLETON,
    },
];

pub fn production_batch_capabilities() -> &'static [FamilyBatchCapability] {
    PRODUCTION_BATCH_CAPABILITIES
}

pub fn batch_execution_capability_for_family(family: &str) -> Option<BatchExecutionCapability> {
    PRODUCTION_BATCH_CAPABILITIES
        .iter()
        .find(|entry| entry.family == family || entry.aliases.contains(&family))
        .map(|entry| entry.execution)
}

/// Fail closed when an instantiated engine disagrees with the capability used
/// before load by execution and parent planning.
pub fn validate_runtime_batch_capability(
    family: &str,
    runtime: BatchExecutionCapability,
) -> anyhow::Result<()> {
    runtime.validate()?;
    let expected = batch_execution_capability_for_family(family)
        .ok_or_else(|| anyhow::anyhow!("model family '{family}' has no static batch capability"))?;
    anyhow::ensure!(
        runtime == expected,
        "runtime batch capability for family '{family}' does not match the static registry: \
         expected {expected:?}, got {runtime:?}"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn production_family_batch_registry_is_explicit_valid_and_singleton_only() {
        let expected = [
            "flux",
            "flux2",
            "sd15",
            "sdxl",
            "sd3",
            "z-image",
            "qwen-image",
            "qwen-image-edit",
            "ltx-video",
            "ltx2",
            "wuerstchen",
        ];
        assert_eq!(
            production_batch_capabilities()
                .iter()
                .map(|entry| entry.family)
                .collect::<Vec<_>>(),
            expected
        );
        let mut names = BTreeSet::new();
        for entry in production_batch_capabilities() {
            assert!(
                names.insert(entry.family),
                "duplicate family {}",
                entry.family
            );
            for alias in entry.aliases {
                assert!(names.insert(alias), "duplicate family alias {alias}");
            }
            entry.execution.validate().unwrap();
            assert_eq!(
                entry.execution.native_batch_sizes,
                &[1],
                "{} must remain singleton until its engine returns multiple outputs",
                entry.family
            );
            assert!(entry.execution.cooperative_cancellation);
        }
    }

    #[test]
    fn family_batch_registry_resolves_factory_aliases_and_rejects_runtime_drift() {
        for alias in [
            "flux.2",
            "flux2-klein",
            "sd1.5",
            "stable-diffusion-1.5",
            "sd3.5",
            "stable-diffusion-3",
            "stable-diffusion-3.5",
            "qwen_image",
            "ltx_video",
            "ltx-2",
            "ltx2.3",
            "wuerstchen-v2",
        ] {
            assert_eq!(
                batch_execution_capability_for_family(alias)
                    .unwrap()
                    .native_batch_sizes,
                &[1],
                "{alias}"
            );
        }
        assert!(batch_execution_capability_for_family("unknown").is_none());
        assert!(validate_runtime_batch_capability(
            "flux",
            BatchExecutionCapability {
                native_batch_sizes: &[1, 2],
                cooperative_cancellation: true,
            }
        )
        .is_err());
    }
}
