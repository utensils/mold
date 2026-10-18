//! Additive diagnostics for live work whose last completed boundary is old.

use serde::{Deserialize, Serialize};

/// Observation only: a stalled worker still owns its device and resources.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct StallDiagnostic {
    pub stage: String,
    pub last_completed_work_unix_ms: u64,
    pub elapsed_stall_ms: u64,
    pub cancel_requested: bool,
    pub device_still_held: bool,
}

impl StallDiagnostic {
    pub fn summary(&self) -> String {
        let intent = if self.cancel_requested {
            "Cancel requested; device still held."
        } else {
            "Device still held; cancellation stops at the next safe boundary."
        };
        format!("Stalled: {} (no completed work for {}s). {} Inspect host diagnostics before an operator-controlled restart; account for other jobs first.",
            self.stage, self.elapsed_stall_ms / 1000, intent)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueueCancellationDisposition {
    Cancelled,
    Pending { stall: Option<StallDiagnostic> },
}
