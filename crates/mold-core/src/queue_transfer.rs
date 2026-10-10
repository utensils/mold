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
        let job_path = crate::client::encode_path_segment(job_id);
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
        let source_caps = self.capabilities().await.ok();
        let modern = source_caps
            .as_ref()
            .is_some_and(|caps| caps.queue.pre_render_transfer);
        let destination_caps = if modern {
            Some(destination.capabilities().await?)
        } else {
            None
        };
        let destination_binding = destination_caps
            .as_ref()
            .and_then(|caps| caps.queue.transfer_identity.clone())
            .unwrap_or_else(|| destination_id.clone());
        ensure!(
            source_caps
                .as_ref()
                .and_then(|caps| caps.queue.transfer_identity.as_deref())
                != Some(destination_binding.as_str()),
            "Choose another machine; these addresses share the same queue owner"
        );
        if modern {
            ensure!(
                destination_caps
                    .as_ref()
                    .and_then(|caps| caps.queue.transfer_identity.as_deref())
                    .is_some_and(|id| !id.trim().is_empty()),
                "Update or refresh destination; durable transfer identity unavailable"
            );
        }
        let mut client_id = queue_transfer_id(&source_id, job_id, &destination_id);
        if modern {
            ensure!(
                destination_caps
                    .as_ref()
                    .is_some_and(|caps| caps.queue.pre_render_transfer),
                "Update destination server for safe transfer recovery"
            );
            let saved = self
                .queue_transfer_json(&format!("/api/queue/{job_path}/transfer/reservation"), None)
                .await?;
            if let Some(id) = saved["transfer_id"].as_str() {
                ensure!(
                    saved["destination_transfer_identity"].as_str() == Some(&destination_binding),
                    "Original reserved for another destination; retry that destination"
                );
                client_id = id.to_owned();
            } else {
                client_id = uuid::Uuid::new_v4().to_string();
            }
        }
        let existing = destination
            .generation_batch_by_client_id(&client_id)
            .await?;
        let detail = self.queue_job(job_id).await?;
        let authority = if let Some(detail) = detail {
            ensure!(
                detail.job.state == "held"
                    || (modern && matches!(detail.job.state.as_str(), "queued" | "paused")),
                "This job is already rendering or is no longer movable"
            );
            Some(
                detail
                    .job
                    .retry_request(&source_id)
                    .or_else(|| {
                        modern.then(|| crate::GenerationRetryRequest {
                            instance_id: source_id.clone(),
                            batch_id: String::new(),
                            client_batch_id: String::new(),
                            job_id: job_id.to_owned(),
                        })
                    })
                    .context("Job has no durable authority")?,
            )
        } else {
            ensure!(existing.is_some(), "The original job no longer exists");
            None
        };
        let batch = if let Some(batch) = existing {
            batch
        } else {
            let authority = authority.as_ref().context("Original job missing")?;
            let mut bound = serde_json::to_value(authority)?;
            bound["transfer_id"] = client_id.clone().into();
            bound["destination_transfer_identity"] = destination_binding.clone().into();
            let request = if modern {
                self.queue_transfer_json(
                    &format!("/api/queue/{job_path}/transfer/reserve"),
                    Some(&bound),
                )
                .await?;
                let exported = self
                    .queue_transfer_json(&format!("/api/queue/{job_path}/transfer"), Some(&bound))
                    .await;
                if exported.is_err() {
                    let _ = self
                        .queue_transfer_json(
                            &format!("/api/queue/{job_path}/transfer/release"),
                            Some(&bound),
                        )
                        .await;
                }
                let request = serde_json::from_value(exported?)?;
                self.queue_transfer_json(
                    &format!("/api/queue/{job_path}/transfer/seal"),
                    Some(&bound),
                )
                .await?;
                request
            } else {
                self.export_held_queue_job(authority).await?
            };
            match destination
                .admit_generation_transfer(
                    &GenerationBatchAdmissionRequest {
                        client_batch_id: client_id.clone(),
                        requests: vec![request],
                    },
                    &destination_id,
                )
                .await
            {
                Ok(batch) => batch,
                Err(error) => {
                    if modern {
                        let aborted=destination.queue_transfer_json("/api/generation-transfers/abort",Some(&serde_json::json!({"transfer_id":client_id,"destination_transfer_identity":destination_binding}))).await.context("Retry same destination to reconcile; original remains reserved")?;
                        ensure!(
                            aborted["transfer_id"].as_str() == Some(&client_id)
                                && aborted["destination_transfer_identity"].as_str()
                                    == Some(&destination_binding),
                            "Destination identity changed; original remains reserved"
                        );
                        if let Some(receipt) = aborted["abort_receipt"].as_str() {
                            bound["abort_receipt"] = receipt.into();
                            self.queue_transfer_json(
                                &format!("/api/queue/{job_path}/transfer/release"),
                                Some(&bound),
                            )
                            .await?;
                            anyhow::bail!("Destination refused ({error}); original restored");
                        }
                    }
                    destination.generation_batch_by_client_id(&client_id).await?.with_context(||format!("Destination acceptance unconfirmed ({error}); retry same destination"))?
                }
            }
        };
        ensure!(
            batch.instance_id == destination_id
                && batch.client_batch_id == client_id
                && batch.children.len() == 1
                && batch.durable,
            "Unexpected destination identity; original remains held"
        );
        if modern
            && matches!(
                batch.children[0].state,
                GenerationBatchChildState::Failed | GenerationBatchChildState::Cancelled
            )
        {
            let original = authority.as_ref().context("Original no longer exists")?;
            let mut bound = serde_json::to_value(original)?;
            bound["transfer_id"] = client_id.clone().into();
            bound["destination_transfer_identity"] = destination_binding.clone().into();
            let aborted=destination.queue_transfer_json("/api/generation-transfers/abort",Some(&serde_json::json!({"transfer_id":client_id,"destination_transfer_identity":destination_binding}))).await?;
            ensure!(
                aborted["transfer_id"].as_str() == Some(&client_id)
                    && aborted["destination_transfer_identity"].as_str()
                        == Some(&destination_binding),
                "Destination changed; original reserved"
            );
            let receipt = aborted["abort_receipt"]
                .as_str()
                .context("Destination terminal failure unconfirmed; retry same destination")?;
            bound["abort_receipt"] = receipt.into();
            self.queue_transfer_json(
                &format!("/api/queue/{job_path}/transfer/release"),
                Some(&bound),
            )
            .await?;
            anyhow::bail!("Destination failed or was cancelled; original restored");
        }
        ensure!(
            !matches!(
                batch.children[0].state,
                GenerationBatchChildState::Failed | GenerationBatchChildState::Cancelled
            ),
            "Destination job is unsuccessful; original remains held"
        );
        let removed = match authority {
            Some(authority) if modern => {
                let mut bound = serde_json::to_value(authority)?;
                bound["transfer_id"] = client_id.into();
                bound["destination_transfer_identity"] = destination_binding.into();
                self.queue_transfer_json(
                    &format!("/api/queue/{job_path}/transfer/complete"),
                    Some(&bound),
                )
                .await
                .is_ok()
            }
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
    #[tokio::test]
    async fn modern_cli_transfer_seals_before_admit_and_restores_after_abort() {
        use wiremock::{
            matchers::{method, path},
            Mock, MockServer, ResponseTemplate,
        };
        let source = MockServer::start().await;
        let dest = MockServer::start().await;
        let id = "11111111-1111-4111-8111-111111111111";
        for (server, identity) in [(&source, "source"), (&dest, "destination")] {
            Mock::given(method("GET")).and(path("/api/status")).respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"version":"1","models_loaded":[],"busy":false,"uptime_secs":1,"instance_id":identity}))).mount(server).await;
            let mut capabilities = crate::ServerCapabilities::default();
            capabilities.queue.pre_render_transfer = true;
            capabilities.queue.transfer_identity = Some(identity.to_owned());
            Mock::given(method("GET"))
                .and(path("/api/capabilities"))
                .respond_with(ResponseTemplate::new(200).set_body_json(capabilities))
                .mount(server)
                .await;
        }
        Mock::given(method("GET"))
            .and(path("/api/queue/job/transfer/reservation"))
            .respond_with(ResponseTemplate::new(200).set_body_json(
                serde_json::json!({"transfer_id":id,"destination_transfer_identity":"destination"}),
            ))
            .mount(&source)
            .await;
        Mock::given(method("GET")).and(path("/api/queue/job")).respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"job":{"id":"job","model":"test","state":"paused","position":0,"started_at_unix_ms":1,"durable":true}}))).mount(&source).await;
        Mock::given(method("GET"))
            .and(path(format!("/api/generation-batches/by-client/{id}")))
            .respond_with(ResponseTemplate::new(404))
            .mount(&dest)
            .await;
        for action in ["reserve", "seal"] {
            Mock::given(method("POST"))
                .and(path(format!("/api/queue/job/transfer/{action}")))
                .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({})))
                .expect(1)
                .mount(&source)
                .await;
        }
        Mock::given(method("POST"))
            .and(path("/api/queue/job/transfer"))
            .respond_with(ResponseTemplate::new(200).set_body_json(
                serde_json::json!({"model":"test","prompt":"original","source_image":"AQID","width":32,"height":32,"steps":4,"batch_size":1,"output_format":"png"}),
            ))
            .expect(1)
            .mount(&source)
            .await;
        Mock::given(method("POST"))
            .and(path("/api/generation-batches/transfer"))
            .respond_with(ResponseTemplate::new(422))
            .expect(1)
            .mount(&dest)
            .await;
        Mock::given(method("POST")).and(path("/api/generation-transfers/abort")).respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"transfer_id":id,"destination_transfer_identity":"destination","abort_receipt":"22222222-2222-4222-8222-222222222222"}))).expect(1).mount(&dest).await;
        Mock::given(method("POST"))
            .and(path("/api/queue/job/transfer/release"))
            .respond_with(ResponseTemplate::new(204))
            .expect(1)
            .mount(&source)
            .await;
        let error = MoldClient::new(&source.uri())
            .send_held_queue_job("job", &MoldClient::new(&dest.uri()))
            .await
            .unwrap_err();
        assert!(error.to_string().contains("original restored"), "{error:#}");
        let requests = source.received_requests().await.unwrap();
        let stages: Vec<_> = requests
            .iter()
            .filter(|r| r.method == "POST")
            .map(|r| r.url.path())
            .collect();
        assert_eq!(
            stages,
            vec![
                "/api/queue/job/transfer/reserve",
                "/api/queue/job/transfer",
                "/api/queue/job/transfer/seal",
                "/api/queue/job/transfer/release"
            ]
        );
        let body: serde_json::Value = serde_json::from_slice(
            &requests
                .iter()
                .find(|r| r.url.path() == "/api/queue/job/transfer")
                .unwrap()
                .body,
        )
        .unwrap();
        assert_eq!(body["transfer_id"], id);
    }
}
