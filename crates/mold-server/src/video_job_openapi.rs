//! Duration routes are documented only in builds that can execute them.
#[derive(utoipa::OpenApi)]
#[openapi(
    paths(
        crate::video_jobs::validate,
        crate::video_job_create::create,
        crate::video_jobs::get,
        crate::video_jobs::list,
        crate::video_jobs::resume,
        crate::video_jobs::cancel,
        crate::video_jobs::remove,
        crate::video_jobs::events,
        crate::video_job_restart_routes::options,
        crate::video_job_restart_routes::export,
        crate::video_job_restart_routes::source,
    ),
    components(schemas(
        mold_core::video_job::VideoJobRequest,
        mold_core::video_job::VideoJobSource,
        mold_core::video_job::VideoJobScene,
        mold_core::video_job::VideoJobPlan,
        mold_core::video_job::VideoJoinMode,
        mold_core::video_job::VideoJobState,
        mold_core::video_job::VideoJobProgressPhase,
        mold_core::video_job::VideoJob,
        mold_core::video_job::VideoSourceUploadRef,
        crate::video_jobs::Listing,
        mold_core::job_restart::RestartExportRequest,
        mold_core::job_restart::RestartAvailability,
        mold_core::video_restart::VideoRestartOptions,
        mold_core::video_restart::VideoRestartExport,
        mold_core::video_restart::VideoRestartPrefix,
    ))
)]
pub(crate) struct VideoJobApiDoc;

#[cfg(test)]
mod tests {
    #[test]
    fn public_document_includes_lifecycle_and_literal_transfer_contracts() {
        let document = serde_json::to_value(crate::routes::api_document()).unwrap();
        for path in [
            "/api/video-jobs",
            "/api/video-jobs/validate",
            "/api/video-jobs/{id}",
            "/api/video-jobs/{id}/events",
            "/api/video-jobs/{id}/cancel",
            "/api/video-jobs/{id}/resume",
            "/api/video-jobs/{id}/restart-options",
            "/api/video-jobs/{id}/restart-export",
            "/api/video-jobs/{id}/restart-source",
        ] {
            assert!(document["paths"].get(path).is_some(), "{path}");
        }
        let schemas = &document["components"]["schemas"];
        assert_eq!(
            schemas["VideoJobRequest"]["properties"]["duration_ms"]["type"],
            "string"
        );
        assert_eq!(
            schemas["VideoRestartExport"]["properties"]["request_json"]["type"],
            "string"
        );
        assert_eq!(
            schemas["VideoRestartPrefix"]["properties"]["size_bytes"]["type"],
            "string"
        );
    }
}
