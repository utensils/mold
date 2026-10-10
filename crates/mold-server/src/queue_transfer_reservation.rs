//! Reserve the source atomically before portable transfer. Dispatch owns the
//! same scheduler fence: once work is transported to a renderer this refuses.
use crate::{routes::ApiError, state::AppState};
use axum::{
    extract::{Path, State},
    http::StatusCode,
    Json,
};
use serde::{Deserialize, Serialize};

#[derive(Deserialize, utoipa::ToSchema)]
pub(crate) struct ReservationRequest {
    #[serde(flatten)]
    authority: mold_core::GenerationRetryRequest,
    transfer_id: String,
    destination_transfer_identity: String,
    #[serde(default)]
    abort_receipt: Option<String>,
}

#[derive(Serialize, utoipa::ToSchema)]
pub(crate) struct Reservation {
    transfer_id: String,
    destination_transfer_identity: String,
}

fn conflict(message: &str) -> ApiError {
    ApiError::with_code(message, "QUEUE_TRANSFER_CONFLICT", StatusCode::CONFLICT)
}

async fn authority(state: &AppState, id: &str, body: &ReservationRequest) -> Result<(), ApiError> {
    if body.transfer_id.len() > 64
        || uuid::Uuid::parse_str(&body.transfer_id).is_err()
        || body.destination_transfer_identity.is_empty()
        || body.destination_transfer_identity.len() > 256
        || body.destination_transfer_identity == *state.instance_id
        || Some(body.destination_transfer_identity.as_str())
            == state.queue_journal.transfer_identity().as_deref()
    {
        return Err(conflict(
            "Choose another machine with a valid transfer identity.",
        ));
    }
    let journal = state.queue_journal.clone();
    let lookup = id.to_owned();
    let row = tokio::task::spawn_blocking(move || journal.row_projection(&lookup))
        .await
        .map_err(|_| ApiError::internal("Could not read source authority"))?
        .map_err(|_| ApiError::internal("Could not read source authority"))?
        .ok_or_else(|| ApiError::queue_job_not_found("Source job no longer exists"))?;
    let a = &body.authority;
    if a.instance_id != *state.instance_id
        || a.job_id != id
        || row.batch_id.as_deref().unwrap_or_default() != a.batch_id
        || row.client_batch_id.as_deref().unwrap_or_default() != a.client_batch_id
    {
        return Err(conflict(
            "The source job's identity changed. Refresh and try again.",
        ));
    }
    Ok(())
}

#[utoipa::path(get, path = "/api/queue/{id}/transfer/reservation", tag = "queue", params(("id" = String, Path, description = "Source job id")), responses((status = 200, description = "Durable transfer reconciliation", body = Option<Reservation>), (status = 409, description = "Transfer identity or state conflict")))]
pub(crate) async fn lookup(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<Option<Reservation>>, ApiError> {
    let journal = state.queue_journal.clone();
    let found = tokio::task::spawn_blocking(move || journal.transfer_reservation(&id))
        .await
        .map_err(|_| ApiError::internal("Could not read transfer reservation"))?
        .map_err(|_| ApiError::internal("Could not read transfer reservation"))?;
    Ok(Json(found.map(
        |(transfer_id, destination_transfer_identity)| Reservation {
            transfer_id,
            destination_transfer_identity,
        },
    )))
}

#[utoipa::path(post, path = "/api/queue/{id}/transfer/reserve", tag = "queue", params(("id" = String, Path, description = "Source job id")), request_body = ReservationRequest, responses((status = 204, description = "Durable transfer reconciliation"), (status = 409, description = "Transfer identity or state conflict")))]
pub(crate) async fn reserve(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(body): Json<ReservationRequest>,
) -> Result<Json<Reservation>, ApiError> {
    let _transition = state.queue_journal.lock_durable_transition().await;
    authority(&state, &id, &body).await?;
    let journal = state.queue_journal.clone();
    let preflight_id = id.clone();
    tokio::task::spawn_blocking(move || {
        crate::queue_transfer::validate_portability(&journal, &preflight_id)
    })
    .await
    .map_err(|_| ApiError::internal("Could not validate transfer"))?
    .map_err(|error| conflict(&error.to_string()))?;
    let token = {
        let _dispatch = state.scheduler_mutation_fence.lock().await;
        match state.job_registry.begin_queue_patch(&id) {
            Ok(token) => Some(token),
            Err(crate::job_registry::TargetGpuUpdateError::NotFound) => None,
            Err(crate::job_registry::TargetGpuUpdateError::AlreadyRunning) => {
                return Err(conflict(
                    "This job is already rendering. Nothing was moved.",
                ))
            }
        }
    };
    let journal = state.queue_journal.clone();
    let job = id.clone();
    let transfer = body.transfer_id.clone();
    let destination = body.destination_transfer_identity.clone();
    let result = tokio::task::spawn_blocking(move || {
        journal.reserve_transfer(&job, &transfer, &destination)
    })
    .await
    .ok()
    .and_then(Result::ok);
    let accepted = matches!(
        result,
        Some(
            mold_db::queue_transfer::ReserveOutcome::Reserved
                | mold_db::queue_transfer::ReserveOutcome::Existing
        )
    );
    if let Some(token) = token {
        let _dispatch = state.scheduler_mutation_fence.lock().await;
        if accepted {
            state.job_registry.finish_queue_patch_state(
                &id,
                token,
                crate::job_registry::JobLifecycle::Paused,
            );
        } else {
            state.job_registry.finish_queue_patch(&id, token);
        }
    }
    if !accepted {
        return Err(conflict("The original is unavailable or reserved for another destination. Retry its original destination to reconcile acceptance."));
    }
    Ok(Json(Reservation {
        transfer_id: body.transfer_id,
        destination_transfer_identity: body.destination_transfer_identity,
    }))
}

#[utoipa::path(post, path = "/api/queue/{id}/transfer/seal", tag = "queue", params(("id" = String, Path, description = "Source job id")), request_body = ReservationRequest, responses((status = 204, description = "Durable transfer reconciliation"), (status = 409, description = "Transfer identity or state conflict")))]
pub(crate) async fn seal(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(body): Json<ReservationRequest>,
) -> Result<Json<Reservation>, ApiError> {
    let _transition = state.queue_journal.lock_durable_transition().await;
    authority(&state, &id, &body).await?;
    let journal = state.queue_journal.clone();
    let transfer = body.transfer_id.clone();
    let destination = body.destination_transfer_identity.clone();
    let sealed =
        tokio::task::spawn_blocking(move || journal.seal_transfer(&id, &transfer, &destination))
            .await
            .map_err(|_| ApiError::internal("Could not seal transfer"))?
            .map_err(|_| ApiError::internal("Could not seal transfer"))?;
    if !sealed {
        return Err(conflict(
            "Transfer reservation changed. Destination admission is prohibited.",
        ));
    }
    Ok(Json(Reservation {
        transfer_id: body.transfer_id,
        destination_transfer_identity: body.destination_transfer_identity,
    }))
}

#[derive(Deserialize, utoipa::ToSchema)]
pub(crate) struct AbortRequest {
    transfer_id: String,
    destination_transfer_identity: String,
}
#[derive(Serialize, utoipa::ToSchema)]
pub(crate) struct AbortResult {
    transfer_id: String,
    destination_transfer_identity: String,
    abort_receipt: Option<String>,
}
#[utoipa::path(post, path = "/api/generation-transfers/abort", tag = "queue", request_body = AbortRequest, responses((status = 200, description = "Durable transfer reconciliation", body = AbortResult), (status = 409, description = "Transfer identity or state conflict")))]
pub(crate) async fn abort_destination(
    State(state): State<AppState>,
    Json(body): Json<AbortRequest>,
) -> Result<Json<AbortResult>, ApiError> {
    if Some(body.destination_transfer_identity.as_str())
        != state.queue_journal.transfer_identity().as_deref()
        || uuid::Uuid::parse_str(&body.transfer_id).is_err()
    {
        return Err(conflict("Destination identity changed"));
    }
    let journal = state.queue_journal.clone();
    let transfer = body.transfer_id.clone();
    let receipt =
        tokio::task::spawn_blocking(move || journal.abort_destination_transfer(&transfer))
            .await
            .map_err(|_| ApiError::internal("Could not reconcile destination"))?
            .map_err(|_| ApiError::internal("Could not reconcile destination"))?;
    Ok(Json(AbortResult {
        transfer_id: body.transfer_id,
        destination_transfer_identity: body.destination_transfer_identity,
        abort_receipt: receipt,
    }))
}

#[utoipa::path(post, path = "/api/queue/{id}/transfer/release", tag = "queue", params(("id" = String, Path, description = "Source job id")), request_body = ReservationRequest, responses((status = 204, description = "Durable transfer reconciliation"), (status = 409, description = "Transfer identity or state conflict")))]
pub(crate) async fn release(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(body): Json<ReservationRequest>,
) -> Result<StatusCode, ApiError> {
    let _transition = state.queue_journal.lock_durable_transition().await;
    authority(&state, &id, &body).await?;
    let journal = state.queue_journal.clone();
    let job = id.clone();
    if journal
        .transfer_reservation(&id)
        .map_err(|_| ApiError::internal("Could not read transfer"))?
        != Some((
            body.transfer_id.clone(),
            body.destination_transfer_identity.clone(),
        ))
    {
        return Err(conflict(
            "Transfer destination changed. Nothing was resumed.",
        ));
    }
    let transfer = body.transfer_id;
    let destination = body.destination_transfer_identity;
    let aborted = body
        .abort_receipt
        .as_deref()
        .is_some_and(|receipt| uuid::Uuid::parse_str(receipt).is_ok());
    let released = tokio::task::spawn_blocking(move || {
        if aborted {
            journal.release_aborted_transfer(&job, &transfer, &destination)
        } else {
            journal.release_transfer(&job, &transfer)
        }
    })
    .await
    .map_err(|_| ApiError::internal("Could not release transfer"))?
    .map_err(|_| ApiError::internal("Could not release transfer"))?;
    if !released {
        return Err(conflict(
            "Transfer reservation changed. Nothing was resumed.",
        ));
    }
    // The durable state, not the client's old cached state, decides whether
    // this runtime becomes queued again. Held has no live registry owner.
    let journal = state.queue_journal.clone();
    let job = id.clone();
    let row = tokio::task::spawn_blocking(move || journal.row_projection(&job))
        .await
        .map_err(|_| ApiError::internal("Could not read released source"))?
        .map_err(|_| ApiError::internal("Could not read released source"))?;
    let _dispatch = state.scheduler_mutation_fence.lock().await;
    if let Ok(token) = state.job_registry.begin_queue_patch(&id) {
        let lifecycle = if row
            .is_some_and(|row| row.state == mold_db::generation_queue::QueueRowState::Queued)
        {
            crate::job_registry::JobLifecycle::Queued
        } else {
            crate::job_registry::JobLifecycle::Paused
        };
        state
            .job_registry
            .finish_queue_patch_state(&id, token, lifecycle);
    }
    Ok(StatusCode::NO_CONTENT)
}
