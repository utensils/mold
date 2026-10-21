//! Installation-bound new attempts and constant-memory prefix relay.
use super::*;
use crate::video_job::{VideoSourceUploadRef, VideoSourceUploadStatus};
use crate::video_restart::{VideoRestartExport, VideoRestartOptions, VideoRestartPrefix};
const CHUNK: usize = 4 * 1024 * 1024;
fn instance(value: &str) -> Result<()> {
    anyhow::ensure!(
        !value.is_empty(),
        "Capture the original installation before reading restart media"
    );
    Ok(())
}
impl MoldClient {
    /// Stored metadata is literal data, not legacy request escape syntax.
    /// The server's Gallery/committed-archive reader must preserve it too.
    pub async fn gallery_video_source(
        &self,
        filename: &str,
    ) -> Result<Option<crate::GalleryImage>> {
        let response = self
            .client
            .get(format!("{}/api/gallery", self.base_url))
            .query(&[("filename", filename)])
            .send()
            .await?;
        let rows: Vec<serde_json::Value> =
            error_for_status_with_body(response).await?.json().await?;
        for raw in rows {
            if raw.get("filename").and_then(|v| v.as_str()) != Some(filename) {
                continue;
            }
            let metadata = raw
                .get("metadata")
                .cloned()
                .context("Library source has no metadata")?;
            let mut row: crate::GalleryImage = serde_json::from_value(raw)?;
            row.metadata = crate::video_job::deserialize_literal_metadata(metadata)?;
            return Ok(Some(row));
        }
        Ok(None)
    }
    pub async fn video_restart_options(
        &self,
        id: &str,
        source_instance: &str,
    ) -> Result<VideoRestartOptions> {
        instance(source_instance)?;
        let response = self
            .client
            .get(format!(
                "{}/api/video-jobs/{}/restart-options",
                self.base_url,
                encode_path_segment(id)
            ))
            .header("x-mold-destination-instance", source_instance)
            .send()
            .await?;
        let options: VideoRestartOptions =
            error_for_status_with_body(response).await?.json().await?;
        anyhow::ensure!(
            options.source_instance_id == source_instance && !options.source_revision.is_empty(),
            "Original video installation or revision changed"
        );
        Ok(options)
    }
    pub async fn export_video_restart(
        &self,
        id: &str,
        authority: &crate::job_restart::RestartExportRequest,
    ) -> Result<VideoRestartExport> {
        instance(&authority.source_instance_id)?;
        instance(&authority.destination_instance_id)?;
        let response = self
            .client
            .post(format!(
                "{}/api/video-jobs/{}/restart-export",
                self.base_url,
                encode_path_segment(id)
            ))
            .header("x-mold-destination-instance", &authority.source_instance_id)
            .json(authority)
            .send()
            .await?;
        let exported: VideoRestartExport =
            error_for_status_with_body(response).await?.json().await?;
        anyhow::ensure!(
            exported.source_instance_id == authority.source_instance_id
                && exported.destination_instance_id == authority.destination_instance_id
                && exported.source_revision == authority.source_revision
                && !exported.request_json.trim().is_empty(),
            "Original video changed during export"
        );
        Ok(exported)
    }
    /// Keep only one bounded chunk in memory. Each client supplies only its own
    /// credential. Uploads intentionally remain available after uncertain admission.
    #[allow(clippy::too_many_arguments)]
    pub async fn transfer_video_restart_prefix(
        &self,
        id: &str,
        source_instance: &str,
        prefix: &VideoRestartPrefix,
        destination: &MoldClient,
        destination_instance: &str,
        upload_id: &str,
    ) -> Result<VideoSourceUploadRef> {
        instance(source_instance)?;
        instance(destination_instance)?;
        anyhow::ensure!(
            uuid::Uuid::parse_str(upload_id)?.to_string() == upload_id,
            "Upload requires a canonical UUID"
        );
        anyhow::ensure!(
            prefix.size_bytes > 0
                && !prefix.source_ref.is_empty()
                && !prefix.upload_spec_json.trim().is_empty()
                && digest(&prefix.sha256)
                && digest(&prefix.archive_identity_sha256),
            "Original prefix is incomplete"
        );
        if let Some(retained) = &prefix.retained_upload {
            anyhow::ensure!(
                source_instance == destination_instance
                    && retained.sha256 == prefix.sha256
                    && retained.size_bytes == prefix.size_bytes
                    && uuid::Uuid::parse_str(&retained.handle).is_ok(),
                "Local prefix grant cannot be sent to another installation"
            );
            return Ok(retained.clone());
        }
        let route = format!(
            "{}/api/video-source-uploads/{}",
            destination.base_url, upload_id
        );
        let response = destination
            .client
            .post(&route)
            .header("x-mold-destination-instance", destination_instance)
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .body(prefix.upload_spec_json.clone())
            .send()
            .await?;
        let status: VideoSourceUploadStatus =
            error_for_status_with_body(response).await?.json().await?;
        anyhow::ensure!(
            status.handle == upload_id
                && status.offset_bytes <= prefix.size_bytes
                && (!status.completed || status.offset_bytes == prefix.size_bytes),
            "Destination returned another prefix upload"
        );
        let mut offset = status.offset_bytes;
        while offset < prefix.size_bytes {
            let expected = (prefix.size_bytes - offset).min(CHUNK as u64) as usize;
            let response = self
                .client
                .get(format!(
                    "{}/api/video-jobs/{}/restart-source",
                    self.base_url,
                    encode_path_segment(id)
                ))
                .header("x-mold-source-instance", source_instance)
                .header("x-mold-video-source-ref", &prefix.source_ref)
                .query(&[("offset_bytes", offset)])
                .send()
                .await?;
            let mut response = error_for_status_with_body(response).await?;
            anyhow::ensure!(
                response
                    .content_length()
                    .is_none_or(|len| len == expected as u64),
                "Original prefix chunk size changed"
            );
            let mut bytes = Vec::with_capacity(expected);
            while let Some(chunk) = response.chunk().await? {
                anyhow::ensure!(
                    chunk.len() <= expected - bytes.len(),
                    "Original prefix exceeded its bounded chunk size"
                );
                bytes.extend_from_slice(&chunk);
            }
            anyhow::ensure!(
                bytes.len() == expected,
                "Original prefix chunk was truncated"
            );
            let response = destination
                .client
                .put(format!("{route}/chunks"))
                .header("x-mold-destination-instance", destination_instance)
                .header(reqwest::header::CONTENT_TYPE, "application/octet-stream")
                .query(&[("offset_bytes", offset)])
                .body(bytes)
                .send()
                .await?;
            let status: VideoSourceUploadStatus =
                error_for_status_with_body(response).await?.json().await?;
            offset += expected as u64;
            anyhow::ensure!(
                status.handle == upload_id && status.offset_bytes == offset,
                "Destination prefix progress changed"
            );
        }
        let response = destination
            .client
            .post(format!("{route}/complete"))
            .header("x-mold-destination-instance", destination_instance)
            .send()
            .await?;
        let sealed: VideoSourceUploadRef =
            error_for_status_with_body(response).await?.json().await?;
        anyhow::ensure!(
            sealed.handle == upload_id
                && sealed.sha256 == prefix.sha256
                && sealed.size_bytes == prefix.size_bytes,
            "Destination retained a different video prefix"
        );
        Ok(sealed)
    }
}
fn digest(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|b| b.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{body_bytes, header, method, path, query_param};
    use wiremock::{Mock, MockServer, ResponseTemplate};
    fn prefix(size: u64) -> VideoRestartPrefix {
        use sha2::Digest;
        let sha256 = format!("{:x}", sha2::Sha256::digest(vec![7; size as usize]));
        let archive = "b".repeat(64);
        let spec=serde_json::json!({"filename":"original.mp4","archive_identity_sha256":archive,"sha256":sha256,"size_bytes":size.to_string(),"metadata":{"model":"video","prompt":r"literal\nscene","seed":u64::MAX,"width":768,"height":512,"steps":2,"guidance":1.0,"fps":24,"frames":107,"version":"test"}}).to_string();
        serde_json::from_str::<crate::video_job::VideoSourceUploadSpec>(&spec).unwrap();
        VideoRestartPrefix {
            filename: "original.mp4".into(),
            archive_identity_sha256: archive,
            sha256,
            size_bytes: size,
            source_ref: "private-ref".into(),
            retained_upload: None,
            upload_spec_json: spec,
        }
    }
    #[tokio::test]
    async fn video_library_source_preserves_literal_metadata_and_full_seed() {
        let server = MockServer::start().await;
        Mock::given(method("GET")).and(path("/api/gallery")).and(query_param("filename","original.mp4"))
          .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!([{"filename":"original.mp4","timestamp":1,"format":"mp4","metadata":{"model":"installed-video","prompt":r"old\night","negative_prompt":r"avoid\new","seed":u64::MAX,"width":768,"height":512,"steps":2,"guidance":1.0,"fps":24,"frames":107,"version":"test"}}]))).expect(1).mount(&server).await;
        let source = MoldClient::new(&server.uri())
            .gallery_video_source("original.mp4")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            source.metadata.negative_prompt.as_deref(),
            Some(r"avoid\new")
        );
        assert_eq!(source.metadata.prompt, r"old\night");
        assert_eq!(source.metadata.seed, u64::MAX);
    }
    #[tokio::test]
    async fn local_prefix_grant_avoids_all_transfer_and_refuses_another_installation() {
        let source = MockServer::start().await;
        let target = MockServer::start().await;
        let mut prefix = prefix(10);
        let handle = uuid::Uuid::new_v4().to_string();
        prefix.retained_upload = Some(VideoSourceUploadRef {
            handle: handle.clone(),
            sha256: prefix.sha256.clone(),
            size_bytes: 10,
        });
        let client = MoldClient::new(&source.uri());
        let target_client = MoldClient::new(&target.uri());
        let accepted = client
            .transfer_video_restart_prefix(
                "original",
                "same",
                &prefix,
                &target_client,
                "same",
                &uuid::Uuid::new_v4().to_string(),
            )
            .await
            .unwrap();
        assert_eq!(accepted.handle, handle);
        assert!(client
            .transfer_video_restart_prefix(
                "original",
                "same",
                &prefix,
                &target_client,
                "replacement",
                &uuid::Uuid::new_v4().to_string()
            )
            .await
            .is_err());
        assert!(source.received_requests().await.unwrap().is_empty());
        assert!(target.received_requests().await.unwrap().is_empty());
    }
    #[tokio::test]
    async fn remote_prefix_relay_is_bounded_exact_and_uses_each_machines_own_credential() {
        let source = MockServer::start().await;
        let target = MockServer::start().await;
        let id = uuid::Uuid::new_v4().to_string();
        let prefix = prefix(CHUNK as u64 + 3);
        let route = format!("/api/video-source-uploads/{id}");
        Mock::given(method("POST"))
            .and(path(&route))
            .and(header("x-api-key", "target-key"))
            .and(header("x-mold-destination-instance", "target"))
            .and(body_bytes(prefix.upload_spec_json.as_bytes().to_vec()))
            .respond_with(ResponseTemplate::new(200).set_body_json(
                serde_json::json!({"handle":id,"offset_bytes":"0","completed":false}),
            ))
            .expect(1)
            .mount(&target)
            .await;
        for (offset, len) in [(0u64, CHUNK), (CHUNK as u64, 3)] {
            let bytes = vec![7; len];
            Mock::given(method("GET"))
                .and(path("/api/video-jobs/original/restart-source"))
                .and(query_param("offset_bytes", offset.to_string()))
                .and(header("x-api-key", "source-key"))
                .and(header("x-mold-source-instance", "source"))
                .and(header("x-mold-video-source-ref", "private-ref"))
                .respond_with(ResponseTemplate::new(200).set_body_bytes(bytes.clone()))
                .expect(1)
                .mount(&source)
                .await;
            Mock::given(method("PUT")).and(path(format!("{route}/chunks"))).and(query_param("offset_bytes",offset.to_string()))
              .and(header("x-api-key", "target-key")).and(header("x-mold-destination-instance","target")).and(body_bytes(bytes))
              .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"handle":id,"offset_bytes":(offset+len as u64).to_string(),"completed":false}))).expect(1).mount(&target).await;
        }
        Mock::given(method("POST")).and(path(format!("{route}/complete"))).and(header("x-api-key", "target-key")).and(header("x-mold-destination-instance","target"))
          .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"handle":id,"sha256":prefix.sha256,"size_bytes":prefix.size_bytes.to_string()}))).expect(1).mount(&target).await;
        let accepted = MoldClient::with_api_key(&source.uri(), "source-key".into())
            .transfer_video_restart_prefix(
                "original",
                "source",
                &prefix,
                &MoldClient::with_api_key(&target.uri(), "target-key".into()),
                "target",
                &id,
            )
            .await
            .unwrap();
        assert_eq!(accepted.size_bytes, CHUNK as u64 + 3);
    }
    #[tokio::test]
    async fn changed_export_echo_and_oversized_prefix_chunk_never_reach_destination_admission() {
        let source = MockServer::start().await;
        Mock::given(method("POST")).and(path("/api/video-jobs/original/restart-export")).and(header("x-mold-destination-instance","source"))
          .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"source_instance_id":"replacement","destination_instance_id":"target","source_revision":"revision","request_json":"{}"}))).mount(&source).await;
        assert!(MoldClient::new(&source.uri())
            .export_video_restart(
                "original",
                &crate::job_restart::RestartExportRequest {
                    source_instance_id: "source".into(),
                    destination_instance_id: "target".into(),
                    source_revision: "revision".into()
                }
            )
            .await
            .is_err());
        let target = MockServer::start().await;
        let id = uuid::Uuid::new_v4().to_string();
        let prefix = prefix(3);
        Mock::given(method("POST"))
            .and(path(format!("/api/video-source-uploads/{id}")))
            .respond_with(ResponseTemplate::new(200).set_body_json(
                serde_json::json!({"handle":id,"offset_bytes":"0","completed":false}),
            ))
            .mount(&target)
            .await;
        Mock::given(method("GET"))
            .and(path("/api/video-jobs/original/restart-source"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(vec![1; 4]))
            .mount(&source)
            .await;
        assert!(MoldClient::new(&source.uri())
            .transfer_video_restart_prefix(
                "original",
                "source",
                &prefix,
                &MoldClient::new(&target.uri()),
                "target",
                &id
            )
            .await
            .is_err());
        assert_eq!(target.received_requests().await.unwrap().len(), 1);
    }
}
