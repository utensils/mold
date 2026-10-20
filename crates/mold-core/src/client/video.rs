//! Literal, installation-fenced durable video transport.
use super::*;
use crate::video_job::{VideoJob, VideoJobPlan, VideoJobRequest};

impl MoldClient {
    pub async fn validate_video_job(
        &self,
        request: &VideoJobRequest,
        instance: &str,
    ) -> Result<VideoJobPlan> {
        require_instance(instance)?;
        request.validate_shape().map_err(anyhow::Error::msg)?;
        let response = self
            .client
            .post(format!("{}/api/video-jobs/validate", self.base_url))
            .header("x-mold-destination-instance", instance)
            .json(request)
            .send()
            .await?;
        Ok(error_for_status_with_body(response).await?.json().await?)
    }

    /// This protocol uses literal JSON, including prompt backslashes.
    pub async fn create_video_job(
        &self,
        request: &VideoJobRequest,
        operation_id: &str,
        instance: &str,
    ) -> Result<VideoJob> {
        request.validate_shape().map_err(anyhow::Error::msg)?;
        anyhow::ensure!(
            request
                .validation_revision
                .as_ref()
                .is_some_and(|proof| !proof.is_empty()),
            "Video requires fresh validation before admission"
        );
        require_admission_identity(operation_id, instance)?;
        let response = self
            .client
            .post(format!("{}/api/video-jobs", self.base_url))
            .header("x-mold-operation-id", operation_id)
            .header("x-mold-destination-instance", instance)
            .json(request)
            .send()
            .await?;
        checked_job(
            error_for_status_with_body(response).await?.json().await?,
            operation_id,
        )
    }

    pub async fn get_video_job(&self, id: &str, instance: &str) -> Result<VideoJob> {
        require_instance(instance)?;
        let response = self
            .client
            .get(self.video_job_url(id, ""))
            .header("x-mold-destination-instance", instance)
            .send()
            .await?;
        checked_job(
            error_for_status_with_body(response).await?.json().await?,
            id,
        )
    }

    pub async fn resume_video_job(&self, id: &str, instance: &str) -> Result<VideoJob> {
        anyhow::ensure!(
            !instance.is_empty(),
            "Video requires a captured destination installation"
        );
        let response = self
            .client
            .post(self.video_job_url(id, "/resume"))
            .header("x-mold-destination-instance", instance)
            .send()
            .await?;
        checked_job(
            error_for_status_with_body(response).await?.json().await?,
            id,
        )
    }

    pub async fn cancel_video_job(&self, id: &str, instance: &str) -> Result<()> {
        anyhow::ensure!(
            !instance.is_empty(),
            "Video requires a captured destination installation"
        );
        let response = self
            .client
            .post(self.video_job_url(id, "/cancel"))
            .header("x-mold-destination-instance", instance)
            .send()
            .await?;
        let receipt: serde_json::Value = error_for_status_with_body(response).await?.json().await?;
        anyhow::ensure!(
            receipt.get("accepted").and_then(serde_json::Value::as_bool) == Some(true),
            "Server did not acknowledge video cancellation"
        );
        Ok(())
    }

    pub async fn delete_video_job(&self, id: &str, instance: &str) -> Result<()> {
        anyhow::ensure!(
            !instance.is_empty(),
            "Video requires a captured destination installation"
        );
        let response = self
            .client
            .delete(self.video_job_url(id, ""))
            .header("x-mold-destination-instance", instance)
            .send()
            .await?;
        error_for_status_with_body(response).await?;
        Ok(())
    }

    /// A new attempt forwards the exact canonical exported body. Proof and
    /// destination-owned prefix references travel independently in headers.
    pub async fn validate_video_job_json_on_instance(
        &self,
        request_json: &str,
        instance: &str,
        source: Option<&crate::video_job::VideoSourceUploadRef>,
    ) -> Result<VideoJobPlan> {
        let builder = self
            .client
            .post(format!("{}/api/video-jobs/validate", self.base_url))
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .body(request_json.to_owned());
        let response = source_headers(builder, instance, source)?.send().await?;
        Ok(error_for_status_with_body(response).await?.json().await?)
    }

    pub async fn create_video_job_json_on_instance(
        &self,
        request_json: &str,
        operation_id: &str,
        instance: &str,
        proof: &str,
        source: Option<&crate::video_job::VideoSourceUploadRef>,
    ) -> Result<VideoJob> {
        require_admission_identity(operation_id, instance)?;
        anyhow::ensure!(
            !proof.is_empty(),
            "Video requires fresh validation before admission"
        );
        let builder = self
            .client
            .post(format!("{}/api/video-jobs", self.base_url))
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .header("x-mold-operation-id", operation_id)
            .header("x-mold-video-validation-revision", proof)
            .body(request_json.to_owned());
        let response = source_headers(builder, instance, source)?.send().await?;
        checked_job(
            error_for_status_with_body(response).await?.json().await?,
            operation_id,
        )
    }

    fn video_job_url(&self, id: &str, suffix: &str) -> String {
        format!(
            "{}/api/video-jobs/{}{}",
            self.base_url,
            encode_path_segment(id),
            suffix
        )
    }
}

fn require_instance(instance: &str) -> Result<()> {
    anyhow::ensure!(
        !instance.is_empty(),
        "Video requires a captured destination installation"
    );
    Ok(())
}

fn require_admission_identity(operation_id: &str, instance: &str) -> Result<()> {
    anyhow::ensure!(
        !instance.is_empty(),
        "Video requires a captured destination installation"
    );
    anyhow::ensure!(
        uuid::Uuid::parse_str(operation_id)?.to_string() == operation_id,
        "Video operation identifier must be a canonical UUID"
    );
    Ok(())
}
fn checked_job(job: VideoJob, id: &str) -> Result<VideoJob> {
    anyhow::ensure!(job.id == id, "Server returned another video job");
    Ok(job)
}
fn source_headers(
    mut builder: crate::relay_transport::RelayRequest,
    instance: &str,
    source: Option<&crate::video_job::VideoSourceUploadRef>,
) -> Result<crate::relay_transport::RelayRequest> {
    anyhow::ensure!(
        !instance.is_empty(),
        "Video requires a captured destination installation"
    );
    builder = builder.header("x-mold-destination-instance", instance);
    if let Some(source) = source {
        builder = builder.header("x-mold-video-source-upload", serde_json::to_string(source)?);
    }
    Ok(builder)
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{body_json, header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};
    fn request() -> VideoJobRequest {
        serde_json::from_value(serde_json::json!({"generation":{"model":"h3-pdmd2:bf16","prompt":r"literal\nscene","width":768,"height":768,"steps":4,"seed":u64::MAX},"duration_ms":"60000","validation_revision":"proof"})).unwrap()
    }
    fn response(id: &str) -> serde_json::Value {
        serde_json::json!({"id":id,"state":"queued","plan":{"validation_revision":"proof","model":"h3-pdmd2:bf16","width":768,"height":768,"fps":24,"single_clip_max_frames":107,"overlap_frames":1,"join_mode":"continuous","clip_count":"14","part_count":"1","added_frames":"1440","source_frames":"0","total_frames":"1440","added_duration_ms":"60000","total_duration_ms":"60000","warnings":[]},"completed_clips":"0","active_child_id":null,"output_filename":null,"error":null,"error_detail":null,"created_at_ms":1,"updated_at_ms":1,"can_cancel":true,"can_resume":false,"can_delete":false})
    }
    #[tokio::test]
    async fn reads_and_validation_require_the_captured_installation() {
        let server = MockServer::start().await;
        let id = uuid::Uuid::new_v4().to_string();
        Mock::given(method("GET"))
            .and(path(format!("/api/video-jobs/{id}")))
            .and(header("x-mold-destination-instance", "captured"))
            .respond_with(ResponseTemplate::new(200).set_body_json(response(&id)))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/api/video-jobs/validate"))
            .and(header("x-mold-destination-instance", "captured"))
            .respond_with(ResponseTemplate::new(200).set_body_json(response(&id)["plan"].clone()))
            .expect(1)
            .mount(&server)
            .await;
        let client = MoldClient::new(&server.uri());
        assert_eq!(client.get_video_job(&id, "captured").await.unwrap().id, id);
        assert_eq!(
            client
                .validate_video_job(&request(), "captured")
                .await
                .unwrap()
                .clip_count,
            14
        );
        assert!(client.get_video_job(&id, "").await.is_err());
        assert!(client.validate_video_job(&request(), "").await.is_err());
    }

    #[tokio::test]
    async fn cancellation_requires_an_explicit_accepted_receipt() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/video-jobs/operation/cancel"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(serde_json::json!({"accepted":false})),
            )
            .mount(&server)
            .await;
        assert!(MoldClient::new(&server.uri())
            .cancel_video_job("operation", "captured")
            .await
            .is_err());
    }

    #[tokio::test]
    async fn admission_preserves_literals_and_lossless_numbers_on_the_captured_installation() {
        let server = MockServer::start().await;
        let id = uuid::Uuid::new_v4().to_string();
        let request = request();
        Mock::given(method("POST"))
            .and(path("/api/video-jobs"))
            .and(header("x-mold-operation-id", id.as_str()))
            .and(header("x-mold-destination-instance", "captured"))
            .and(body_json(serde_json::to_value(&request).unwrap()))
            .respond_with(ResponseTemplate::new(202).set_body_json(response(&id)))
            .expect(1)
            .mount(&server)
            .await;
        let client = MoldClient::new(&server.uri());
        assert_eq!(
            client
                .create_video_job(&request, &id, "captured")
                .await
                .unwrap()
                .id,
            id
        );
    }
    #[tokio::test]
    async fn a_wrong_parent_reply_or_missing_validation_cannot_be_accepted() {
        let server = MockServer::start().await;
        let client = MoldClient::new(&server.uri());
        let id = uuid::Uuid::new_v4().to_string();
        Mock::given(method("POST"))
            .and(path("/api/video-jobs"))
            .respond_with(ResponseTemplate::new(202).set_body_json(response("different-parent")))
            .expect(1)
            .mount(&server)
            .await;
        assert!(client
            .create_video_job(&request(), &id, "captured")
            .await
            .is_err());
        let mut unvalidated = request();
        unvalidated.validation_revision = None;
        assert!(client
            .create_video_job(&unvalidated, &id, "captured")
            .await
            .is_err());
    }
}
