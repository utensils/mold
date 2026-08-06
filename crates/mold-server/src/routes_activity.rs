use axum::{extract::State, Json};
use mold_core::{ActiveWorkItem, ActiveWorkSnapshot, QueueActivityPhase};
use std::collections::{HashMap, HashSet};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::routes::ApiError;
use crate::state::AppState;

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX)
}

fn nonnegative_ms(value: i64) -> u64 {
    value.try_into().unwrap_or_default()
}

fn scheduler_phase(phase: &QueueActivityPhase) -> &'static str {
    match phase {
        QueueActivityPhase::Active | QueueActivityPhase::Cpu => "running",
        QueueActivityPhase::Dispatching => "loading",
        QueueActivityPhase::Blocked
        | QueueActivityPhase::WarmWait
        | QueueActivityPhase::Queued
        | QueueActivityPhase::Unknown(_) => "preparing",
    }
}

fn phase_rank(phase: &str) -> u8 {
    match phase {
        "running" => 4,
        "loading" => 3,
        "preparing" => 2,
        "queued" => 1,
        _ => 0,
    }
}

#[derive(Clone)]
struct SchedulerActivity {
    kind: String,
    phase: &'static str,
    position: Option<usize>,
    current: Option<u64>,
    total: Option<u64>,
}

fn scheduler_activity(state: &AppState) -> HashMap<String, SchedulerActivity> {
    let mut by_parent: HashMap<String, SchedulerActivity> = HashMap::new();
    let Some(plan) = state.scheduled_work.latest_plan() else {
        return by_parent;
    };
    for work in plan.work_items {
        let phase = scheduler_phase(&work.activity_phase);
        let kind = match work.work_kind.as_str() {
            "chain_stage" => "sequence",
            "generation" | "prepared_sibling" => "generation",
            other => other,
        }
        .to_string();
        let candidate = SchedulerActivity {
            kind,
            phase,
            position: usize::try_from(work.queue_rank).ok(),
            current: work.chain_stage.map(u64::from),
            total: work
                .batch_partition
                .map(|partition| u64::from(partition.count)),
        };
        by_parent
            .entry(work.parent_id)
            .and_modify(|current| {
                if phase_rank(candidate.phase) > phase_rank(current.phase) {
                    current.phase = candidate.phase;
                }
                current.position = current.position.min(candidate.position);
                current.current = current.current.max(candidate.current);
                current.total = current.total.max(candidate.total);
            })
            .or_insert(candidate);
    }
    by_parent
}

/// Return the host-owned, nonterminal work snapshot used to reconcile Now
/// Developing after a client restart or reconnect.
#[utoipa::path(
    get,
    path = "/api/activity",
    tag = "activity",
    responses((status = 200, description = "Active work snapshot", body = ActiveWorkSnapshot))
)]
pub async fn list_active_work(
    State(state): State<AppState>,
) -> Result<Json<ActiveWorkSnapshot>, ApiError> {
    let observed_at_unix_ms = now_ms();
    let queue = state.job_registry.snapshot();
    let mut items = Vec::new();
    let mut represented = HashSet::new();
    let scheduler = scheduler_activity(&state);

    for entry in queue.entries {
        represented.insert(entry.id.clone());
        let scheduled = scheduler.get(&entry.id);
        let phase = scheduled.map_or_else(
            || match entry.state {
                crate::job_registry::JobLifecycle::Queued => "queued",
                crate::job_registry::JobLifecycle::Running => "running",
            },
            |activity| activity.phase,
        );
        let position = scheduled
            .and_then(|activity| activity.position)
            .or(Some(entry.position));
        items.push(ActiveWorkItem {
            id: entry.id,
            kind: "generation".into(),
            phase: phase.into(),
            model: Some(entry.model),
            created_at_unix_ms: entry.started_at_unix_ms,
            updated_at_unix_ms: observed_at_unix_ms,
            position,
            current: None,
            total: None,
            can_cancel: matches!(entry.state, crate::job_registry::JobLifecycle::Queued),
        });
    }

    if let Some(db) = state.metadata_db.as_ref().as_ref() {
        let rows = mold_db::chain_jobs::list_jobs(db).map_err(|error| {
            ApiError::internal(format!("failed to list active chains: {error:#}"))
        })?;
        for row in rows.into_iter().filter(|row| {
            matches!(
                row.state,
                mold_core::chain_job::ChainJobState::Queued
                    | mold_core::chain_job::ChainJobState::Running
            )
        }) {
            represented.insert(row.id.clone());
            let cancelling = row.state == mold_core::chain_job::ChainJobState::Running
                && state
                    .chain_jobs
                    .as_ref()
                    .is_some_and(|runner| runner.is_cancelling(&row.id));
            let phase = if cancelling {
                "cancelling"
            } else if let Some(activity) = scheduler.get(&row.id) {
                activity.phase
            } else if row.state == mold_core::chain_job::ChainJobState::Queued {
                "queued"
            } else {
                "running"
            };
            items.push(ActiveWorkItem {
                id: row.id,
                kind: "sequence".into(),
                phase: phase.into(),
                model: Some(row.model),
                created_at_unix_ms: nonnegative_ms(row.created_at_ms),
                updated_at_unix_ms: nonnegative_ms(row.updated_at_ms),
                position: None,
                current: Some(u64::from(row.current_stage)),
                total: Some(u64::from(row.stage_count)),
                can_cancel: true,
            });
        }
    }

    // Scheduler-only work includes preparation/utility phases that never
    // enter either durable registry. Collapse partitions/stages to one parent
    // row so a batch or chain never double-renders.
    for (parent_id, activity) in &scheduler {
        // Generation and sequence registries own their terminal boundary. A
        // separately published scheduler plan may lag completion briefly and
        // is enrichment only for those work kinds, never resurrection.
        if matches!(activity.kind.as_str(), "generation" | "sequence") {
            continue;
        }
        if !represented.insert(parent_id.clone()) {
            continue;
        }
        items.push(ActiveWorkItem {
            id: parent_id.clone(),
            kind: activity.kind.clone(),
            phase: activity.phase.into(),
            model: None,
            created_at_unix_ms: observed_at_unix_ms,
            updated_at_unix_ms: observed_at_unix_ms,
            position: activity.position,
            current: activity.current,
            total: activity.total,
            can_cancel: false,
        });
    }

    let downloads = state.downloads.listing().await;
    for (position, job) in downloads
        .active_jobs
        .into_iter()
        .chain(downloads.queued)
        .enumerate()
    {
        let active = job.status == mold_core::JobStatus::Active;
        items.push(ActiveWorkItem {
            id: job.id,
            kind: "download".into(),
            phase: if active { "downloading" } else { "queued" }.into(),
            model: Some(job.model),
            created_at_unix_ms: job
                .started_at
                .map(nonnegative_ms)
                .unwrap_or(observed_at_unix_ms),
            updated_at_unix_ms: observed_at_unix_ms,
            position: Some(position),
            current: Some(job.bytes_done),
            total: Some(job.bytes_total),
            can_cancel: true,
        });
    }

    items.sort_by_key(|item| (item.position.unwrap_or(usize::MAX), item.created_at_unix_ms));
    Ok(Json(ActiveWorkSnapshot {
        instance_id: state.instance_id.as_ref().clone(),
        observed_at_unix_ms,
        items,
    }))
}
