//! Private-UAT-only allocation observations from the production H3 math path.
//!
//! The capture is thread-local because one H3 attempt is pinned to one GPU
//! worker. Observers record sizes derived by the exact operators being run;
//! they do not allocate tensors, synchronize CUDA, or alter operation order.

use std::cell::Cell;

use candle::Result;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct H3PrivateWorkspaceObservation {
    pub attention_workspace_device_bytes: u64,
    pub ffn_workspace_device_bytes: u64,
}

thread_local! {
    static ACTIVE: Cell<Option<H3PrivateWorkspaceObservation>> = const { Cell::new(None) };
}

#[derive(Debug)]
pub struct H3PrivateWorkspaceCapture {
    active: bool,
}

impl H3PrivateWorkspaceCapture {
    pub fn begin() -> Result<Self> {
        ACTIVE.with(|active| {
            if active.get().is_some() {
                candle::bail!("private H3 workspace capture is already active")
            }
            active.set(Some(H3PrivateWorkspaceObservation::default()));
            Ok(Self { active: true })
        })
    }

    pub fn finish(mut self) -> Result<H3PrivateWorkspaceObservation> {
        let observation = ACTIVE
            .with(|active| active.take())
            .ok_or_else(|| candle::Error::Msg("private H3 workspace capture disappeared".into()))?;
        self.active = false;
        Ok(observation)
    }
}

impl Drop for H3PrivateWorkspaceCapture {
    fn drop(&mut self) {
        if self.active {
            ACTIVE.with(|active| active.set(None));
        }
    }
}

pub(super) fn observe_attention(bytes: u64) {
    ACTIVE.with(|active| {
        if let Some(mut observation) = active.get() {
            observation.attention_workspace_device_bytes =
                observation.attention_workspace_device_bytes.max(bytes);
            active.set(Some(observation));
        }
    });
}

pub(super) fn observe_ffn(bytes: u64) {
    ACTIVE.with(|active| {
        if let Some(mut observation) = active.get() {
            observation.ffn_workspace_device_bytes =
                observation.ffn_workspace_device_bytes.max(bytes);
            active.set(Some(observation));
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_retains_independent_workspace_maxima() -> Result<()> {
        let capture = H3PrivateWorkspaceCapture::begin()?;
        observe_attention(7);
        observe_attention(5);
        observe_ffn(11);
        assert_eq!(
            capture.finish()?,
            H3PrivateWorkspaceObservation {
                attention_workspace_device_bytes: 7,
                ffn_workspace_device_bytes: 11,
            }
        );
        Ok(())
    }
}
