//! Freeze the existing execution planner's ordered adapter artifact identities.

use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct FrozenVideoLora {
    pub path: PathBuf,
    pub scale_bits: u64,
    pub content_fingerprint: String,
}
impl FrozenVideoLora {
    pub(crate) fn validate(&self) -> Result<()> {
        ensure!(
            !self.path.as_os_str().is_empty()
                && f64::from_bits(self.scale_bits).is_finite()
                && f64::from_bits(self.scale_bits).abs() > 1e-8
                && self.content_fingerprint.len() == 64
                && self
                    .content_fingerprint
                    .bytes()
                    .all(|byte| byte.is_ascii_hexdigit()),
            "invalid frozen video adapter authority"
        );
        Ok(())
    }
}
/// Call after built-in adapter materialization, with the frozen model config.
pub(crate) fn capture(
    config: &mold_core::Config,
    request: &mold_core::GenerateRequest,
) -> Result<Vec<FrozenVideoLora>> {
    crate::execution_plan::effective_loras(config, request, None)
        .into_iter()
        .map(|lora| {
            ensure!(
                lora.path.is_file(),
                "video adapter is unavailable: {}",
                lora.path.display()
            );
            let frozen = FrozenVideoLora {
                path: lora.path,
                scale_bits: lora.scale_bits,
                content_fingerprint: lora.content_fingerprint.0,
            };
            frozen.validate()?;
            Ok(frozen)
        })
        .collect()
}
/// A resumed/new child must use precisely the originally selected artifacts.
pub(crate) fn verify(
    config: &mold_core::Config,
    request: &mold_core::GenerateRequest,
    frozen: &[FrozenVideoLora],
) -> Result<()> {
    ensure!(
        capture(config, request)? == frozen,
        "video adapter authority changed; start a new attempt to use the current adapters"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use mold_core::{Config, GenerateRequest, LoraWeight};
    fn request() -> GenerateRequest {
        serde_json::from_value(serde_json::json!({"model":"ltx-2-19b-distilled:fp8","prompt":"walk","width":32,"height":32,"steps":4})).unwrap()
    }
    #[test]
    fn frozen_adapters_preserve_order_scales_and_detect_same_name_replacement() {
        let dir = tempfile::tempdir().unwrap();
        let first = dir.path().join("first.safetensors");
        let second = dir.path().join("second.safetensors");
        std::fs::write(&first, b"first").unwrap();
        std::fs::write(&second, b"second").unwrap();
        let mut req = request();
        req.loras = Some(vec![
            LoraWeight {
                path: first.to_string_lossy().into(),
                scale: 0.5,
                expert: None,
            },
            LoraWeight {
                path: second.to_string_lossy().into(),
                scale: 1.0,
                expert: None,
            },
            LoraWeight {
                path: "missing-disabled".into(),
                scale: 0.0,
                expert: None,
            },
        ]);
        let frozen = capture(&Config::default(), &req).unwrap();
        assert_eq!(frozen.len(), 2);
        verify(&Config::default(), &req, &frozen).unwrap();
        let mut changed = req.clone();
        changed.loras.as_mut().unwrap().swap(0, 1);
        assert!(verify(&Config::default(), &changed, &frozen).is_err());
        changed = req.clone();
        changed.loras.as_mut().unwrap()[0].scale = 0.75;
        assert!(verify(&Config::default(), &changed, &frozen).is_err());
        let replacement = dir.path().join("replacement");
        std::fs::write(&replacement, b"first").unwrap();
        std::fs::rename(replacement, &first).unwrap();
        assert!(verify(&Config::default(), &req, &frozen).is_err());
    }
    #[test]
    fn absent_adapter_refuses_before_freezing_execution() {
        let mut req = request();
        req.lora = Some(LoraWeight {
            path: "missing-custom-lora.safetensors".into(),
            scale: 1.0,
            expert: None,
        });
        assert!(capture(&Config::default(), &req).is_err());
    }
}
