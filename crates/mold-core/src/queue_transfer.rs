//! Explicit, idempotent movement of a held job between authenticated hosts.
use crate::{
    GenerationBatchAdmissionRequest, GenerationBatchChildState, GenerationBatchStatus, MoldClient,
};
use anyhow::{ensure, Context, Result};
use sha2::{Digest, Sha256};

pub fn queue_transfer_id(source: &str, job: &str, destination: &str) -> String {
    let key = serde_json::to_vec(&["mold.queue-transfer.v1", source, job, destination])
        .expect("strings serialize");
    let hash = Sha256::digest(key);
    let mut bytes = [0; 16];
    bytes.copy_from_slice(&hash[..16]);
    bytes[6] = (bytes[6] & 15) | 80;
    bytes[8] = (bytes[8] & 63) | 128;
    uuid::Uuid::from_bytes(bytes).to_string()
}

impl MoldClient {
    /// Keep the original until the destination durably accepts it. Retrying
    /// the same destination resolves the same batch, including after restart.
    pub async fn send_held_queue_job(
        &self,
        job_id: &str,
        destination: &MoldClient,
    ) -> Result<(GenerationBatchStatus, bool)> {
        let source_id = self
            .server_status()
            .await?
            .instance_id
            .context("Source has no instance identity")?;
        let destination_id = destination
            .server_status()
            .await?
            .instance_id
            .context("Destination has no instance identity")?;
        ensure!(
            !source_id.is_empty() && !destination_id.is_empty() && source_id != destination_id,
            "Choose another connected machine"
        );
        let client_id = queue_transfer_id(&source_id, job_id, &destination_id);
        let existing = destination
            .generation_batch_by_client_id(&client_id)
            .await?;
        let detail = self.queue_job(job_id).await?;
        let authority = if let Some(detail) = detail {
            ensure!(
                detail.job.state == "held",
                "Only a held job can be sent to another machine"
            );
            Some(
                detail
                    .job
                    .retry_request(&source_id)
                    .context("Job has no durable batch authority")?,
            )
        } else {
            ensure!(existing.is_some(), "The original job no longer exists");
            None
        };
        let batch = if let Some(batch) = existing {
            batch
        } else {
            let request = self
                .export_held_queue_job(authority.as_ref().context("Original job missing")?)
                .await?;
            match destination.admit_generation_transfer(&GenerationBatchAdmissionRequest { client_batch_id: client_id.clone(), requests: vec![request] }, &destination_id).await {
                Ok(batch) => batch,
                Err(error) => destination.generation_batch_by_client_id(&client_id).await?
                    .with_context(|| format!("Destination acceptance is unconfirmed ({error}); the original remains held. Retry the same destination safely."))?,
            }
        };
        ensure!(
            batch.instance_id == destination_id
                && batch.client_batch_id == client_id
                && batch.children.len() == 1
                && batch.durable,
            "Unexpected destination identity; original remains held"
        );
        ensure!(
            !matches!(
                batch.children[0].state,
                GenerationBatchChildState::Failed | GenerationBatchChildState::Cancelled
            ),
            "Destination job is unsuccessful; original remains held"
        );
        let removed = match authority {
            Some(authority) => self.complete_held_queue_transfer(&authority).await.is_ok(),
            None => true,
        };
        Ok((batch, removed))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stable_transfer_identity_is_destination_scoped() {
        assert_eq!(
            queue_transfer_id("source", "job", "dest"),
            "1bce2d47-3909-5d66-8d96-0d2d3a6ae7b3"
        );
        assert_eq!(
            queue_transfer_id("source", "job", "dest"),
            queue_transfer_id("source", "job", "dest")
        );
        assert_ne!(
            queue_transfer_id("source", "job", "dest"),
            queue_transfer_id("source", "job", "other")
        );
    }
    #[tokio::test]
    async fn transfer_only_removes_original_after_durable_destination_acceptance() {
        use wiremock::{
            matchers::{header, method, path},
            Mock, MockServer, ResponseTemplate,
        };
        let source = MockServer::start().await;
        let destination = MockServer::start().await;
        for (server, identity) in [(&source, "source"), (&destination, "destination")] {
            Mock::given(method("GET")).and(path("/api/status"))
                .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "version":"0.28.0", "models_loaded":[], "busy":false, "uptime_secs":1,"instance_id":identity
                }))).mount(server).await;
        }
        Mock::given(method("GET")).and(path("/api/queue/job"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"job":{
                "id":"job","model":"test","state":"held","position":0,"started_at_unix_ms":1,"batch_id":"batch","client_batch_id":"original"
            }}))).mount(&source).await;
        let client_id = queue_transfer_id("source", "job", "destination");
        Mock::given(method("GET"))
            .and(path(format!(
                "/api/generation-batches/by-client/{client_id}"
            )))
            .respond_with(ResponseTemplate::new(404))
            .mount(&destination)
            .await;
        Mock::given(method("POST")).and(path("/api/queue/job/transfer")).and(header("x-api-key", "source-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "model":"test","prompt":"original words","width":32,"height":32,"steps":4,"batch_size":1,"output_format":"png","seed":42,"source_image":"AQID"
            }))).expect(1).mount(&source).await;
        Mock::given(method("POST")).and(path("/api/generation-batches/transfer")).and(header("x-api-key", "destination-key"))
            .respond_with(ResponseTemplate::new(202).set_body_json(serde_json::json!({
                "id":"new","instance_id":"destination","client_batch_id":client_id,"durable":true,"children":[{"index":1,"job_id":"new-job","state":"accepted","created_at_ms":1,"updated_at_ms":1}]
            }))).expect(1).mount(&destination).await;
        Mock::given(method("POST"))
            .and(path("/api/queue/job/transfer/complete"))
            .respond_with(ResponseTemplate::new(204))
            .expect(1)
            .mount(&source)
            .await;
        let (batch, removed) = MoldClient::with_api_key(&source.uri(), "source-key".into())
            .send_held_queue_job(
                "job",
                &MoldClient::with_api_key(&destination.uri(), "destination-key".into()),
            )
            .await
            .unwrap();
        assert!(removed);
        assert_eq!(batch.children[0].job_id, "new-job");
        let requests = destination.received_requests().await.unwrap();
        let posted: serde_json::Value =
            serde_json::from_slice(&requests.iter().find(|r| r.method == "POST").unwrap().body)
                .unwrap();
        assert_eq!(posted["requests"][0]["seed"], 42);
        assert_eq!(posted["requests"][0]["source_image"], "AQID");
    }
}
