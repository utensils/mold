//! Explicit new attempts; exact frozen Resume is a separate operation.
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct RestartAvailability {
    pub supported: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct ChainRestartOptions {
    pub source_instance_id: String,
    pub source_revision: String,
    pub restart: RestartAvailability,
    pub send: RestartAvailability,
}

#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(deny_unknown_fields)]
pub struct RestartExportRequest {
    pub destination_instance_id: String,
    pub source_instance_id: String,
    pub source_revision: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct ChainRestartExport {
    pub destination_instance_id: String,
    pub source_instance_id: String,
    pub source_revision: String,
    /// Wire-ready JSON for clients whose numeric parser cannot preserve u64.
    /// Forward unchanged to chain creation; never persist this private media.
    pub request_json: String,
}

impl crate::MoldClient {
    pub async fn chain_restart_options(&self, id: &str) -> anyhow::Result<ChainRestartOptions> {
        let path = crate::client::encode_path_segment(id);
        Ok(serde_json::from_value(
            self.queue_transfer_json(&format!("/api/chain-jobs/{path}/restart-options"), None)
                .await?,
        )?)
    }

    pub async fn export_chain_restart(
        &self,
        id: &str,
        authority: &RestartExportRequest,
    ) -> anyhow::Result<ChainRestartExport> {
        let path = crate::client::encode_path_segment(id);
        Ok(serde_json::from_value(
            self.queue_transfer_json(
                &format!("/api/chain-jobs/{path}/restart-export"),
                Some(&serde_json::to_value(authority)?),
            )
            .await?,
        )?)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn older_servers_do_not_implicitly_authorize_destination_instance_fencing() {
        let old: crate::QueueCapabilities =
            serde_json::from_value(serde_json::json!({"can_pause":true,"can_cancel_all":true}))
                .unwrap();
        assert!(!old.chain_restart);
        let current: crate::QueueCapabilities = serde_json::from_value(
            serde_json::json!({"can_pause":true,"can_cancel_all":true,"chain_restart":true}),
        )
        .unwrap();
        assert!(current.chain_restart);
    }
}
