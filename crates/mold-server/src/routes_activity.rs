use axum::{extract::State, Json};
use mold_core::{ActiveWorkItem, ActiveWorkSnapshot, QueueActivityPhase};
use std::collections::{HashMap, HashSet};
use std::time::{SystemTime, UNIX_EPOCH};

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

fn scheduler_phase(
    work: &mold_core::QueueWorkItem,
    progress: Option<&mold_core::queue_progress::QueueJobProgress>,
) -> &'static str {
    match work.activity_phase {
        QueueActivityPhase::Cpu => "running",
        QueueActivityPhase::Active
            if progress
                .and_then(|value| value.stage_current.or(value.step))
                .is_some() =>
        {
            "running"
        }
        // Only registry-backed generations have a model-loading progress
        // record. Scheduler-owned GPU work (chain stages, upscales, utilities)
        // is already executing once its lease is Active and must not inherit
        // the generation fallback label indefinitely.
        QueueActivityPhase::Active if progress.is_some() => "loading",
        QueueActivityPhase::Active => "running",
        QueueActivityPhase::Dispatching => "loading",
        QueueActivityPhase::Blocked
            if work.blocked_reason == Some(mold_core::QueueBlockedReason::Preparing) =>
        {
            "preparing"
        }
        QueueActivityPhase::Blocked
        | QueueActivityPhase::WarmWait
        | QueueActivityPhase::Queued
        | QueueActivityPhase::Unknown(_) => "queued",
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
    current: Option<u64>,
    total: Option<u64>,
    stage: Option<String>,
    preparation_progress: Option<mold_core::QueuePreparationProgress>,
}

#[derive(Default)]
struct SchedulerActivityIndex {
    by_work: HashMap<String, SchedulerActivity>,
    by_parent: HashMap<String, SchedulerActivity>,
}

fn merge_scheduler_activity(current: &mut SchedulerActivity, candidate: &SchedulerActivity) {
    let candidate_rank = phase_rank(candidate.phase);
    let current_rank = phase_rank(current.phase);
    if candidate_rank > current_rank {
        *current = candidate.clone();
    } else if candidate_rank == current_rank {
        if candidate.stage.is_some() {
            current.stage = candidate.stage.clone();
        }
        if candidate.preparation_progress.is_some() {
            current.preparation_progress = candidate.preparation_progress.clone();
            current.current = candidate.current;
            current.total = candidate.total;
        } else {
            current.current = current.current.max(candidate.current);
            current.total = current.total.max(candidate.total);
        }
    }
}

fn scheduler_activity(state: &AppState) -> SchedulerActivityIndex {
    let mut index = SchedulerActivityIndex::default();
    let Some(plan) = state.scheduled_work.latest_plan() else {
        return index;
    };
    for work in plan.work_items {
        let runtime_progress = state
            .job_registry
            .progress_snapshot(&work.work_id)
            .flatten();
        let phase = scheduler_phase(&work, runtime_progress.as_ref());
        let kind = match work.work_kind.as_str() {
            "chain_stage" => "sequence",
            "generation" | "prepared_sibling" => "generation",
            other => other,
        }
        .to_string();
        // `queue_rank` is deliberately not read here. It is the coordinator's
        // monotonic submission counter, not a place in line; the registry
        // ordering below is the one authority for that, and it is what
        // `GET /api/queue` reports.
        let preparation_progress = work.preparation_progress.clone();
        let stage = runtime_progress.as_ref().and_then(|progress| {
            progress.stage.clone().or_else(|| {
                progress
                    .weight_load
                    .as_ref()
                    .map(|load| format!("Loading {}", load.component))
            })
        });
        let runtime_current = runtime_progress.as_ref().and_then(|progress| {
            progress
                .stage_current
                .or(progress.step)
                .map(|value| value.try_into().unwrap_or(u64::MAX))
                .or_else(|| progress.weight_load.as_ref().map(|load| load.bytes_loaded))
        });
        let runtime_total = runtime_progress.as_ref().and_then(|progress| {
            progress
                .stage_total
                .or(progress.total)
                .map(|value| value.try_into().unwrap_or(u64::MAX))
                .or_else(|| progress.weight_load.as_ref().map(|load| load.bytes_total))
        });
        let candidate = SchedulerActivity {
            kind,
            phase,
            current: preparation_progress
                .as_ref()
                .map(|progress| progress.bytes_done)
                .or(runtime_current)
                .or_else(|| work.chain_stage.map(u64::from)),
            total: preparation_progress
                .as_ref()
                .map(|progress| progress.bytes_total)
                .or(runtime_total)
                .or_else(|| {
                    work.batch_partition
                        .map(|partition| u64::from(partition.count))
                }),
            stage: stage.or_else(|| (phase == "loading").then(|| "Loading model".to_string())),
            preparation_progress,
        };
        index
            .by_work
            .insert(work.work_id.clone(), candidate.clone());
        index
            .by_parent
            .entry(work.parent_id)
            .and_modify(|current| merge_scheduler_activity(current, &candidate))
            .or_insert(candidate);
    }
    index
}

/// Return the host-owned, nonterminal work snapshot used to reconcile Now
/// Developing after a client restart or reconnect.
#[utoipa::path(
    get,
    path = "/api/activity",
    tag = "activity",
    responses((status = 200, description = "Active work snapshot", body = ActiveWorkSnapshot))
)]
pub async fn list_active_work(State(state): State<AppState>) -> Json<ActiveWorkSnapshot> {
    let observed_at_unix_ms = now_ms();
    let queue = state.job_registry.snapshot();
    let mut items = Vec::new();
    let mut represented = HashSet::new();
    let mut unavailable_kinds = Vec::new();
    let scheduler = scheduler_activity(&state);

    for entry in queue.entries {
        represented.insert(entry.id.clone());
        let scheduled = scheduler
            .by_work
            .get(&entry.id)
            .or_else(|| scheduler.by_parent.get(&entry.id));
        let phase = scheduled.map_or_else(
            || match entry.state {
                crate::job_registry::JobLifecycle::Queued => "queued",
                crate::job_registry::JobLifecycle::Running => "running",
                crate::job_registry::JobLifecycle::Paused => "paused",
                // Held work is parked, not in flight. The activity strip is
                // present-tense only.
                crate::job_registry::JobLifecycle::Held => "held",
            },
            |activity| activity.phase,
        );
        let cancelling = entry.state == crate::job_registry::JobLifecycle::Running
            && state.job_registry.cancel_requested(&entry.id);
        items.push(ActiveWorkItem {
            id: entry.id,
            kind: "generation".into(),
            execution: None,
            phase: if cancelling { "cancelling" } else { phase }.into(),
            model: Some(entry.model),
            created_at_unix_ms: entry.started_at_unix_ms,
            updated_at_unix_ms: observed_at_unix_ms,
            position: Some(entry.position),
            current: scheduled.and_then(|activity| activity.current),
            total: scheduled.and_then(|activity| activity.total),
            stage: scheduled.and_then(|activity| activity.stage.clone()),
            preparation_progress: scheduled
                .and_then(|activity| activity.preparation_progress.clone()),
            can_cancel: !cancelling
                && matches!(
                    entry.state,
                    crate::job_registry::JobLifecycle::Queued
                        | crate::job_registry::JobLifecycle::Running
                        | crate::job_registry::JobLifecycle::Paused
                ),
        });
    }

    if let Some(db) = state.metadata_db.as_ref().as_ref() {
        if let Some(owner_uuid) = state.queue_journal.owner_uuid() {
            match mold_db::generation_queue::projections_in_state(
                db,
                owner_uuid,
                mold_db::generation_queue::QueueRowState::Paused,
                state.queue_capacity,
            ) {
                Ok(rows) => {
                    for row in rows {
                        if !represented.insert(row.id.clone()) {
                            continue;
                        }
                        items.push(ActiveWorkItem {
                            id: row.id,
                            kind: "generation".into(),
                            execution: None,
                            phase: "paused".into(),
                            model: Some(row.model),
                            created_at_unix_ms: nonnegative_ms(row.created_at_ms),
                            updated_at_unix_ms: observed_at_unix_ms,
                            position: None,
                            current: None,
                            total: None,
                            stage: None,
                            preparation_progress: None,
                            can_cancel: true,
                        });
                    }
                }
                Err(error) => {
                    tracing::warn!(%error, "failed to read restart-paused generation work");
                    unavailable_kinds.push("generation".to_string());
                }
            }
        }
        let rows = match mold_db::chain_jobs::list_jobs(db) {
            Ok(rows) => rows,
            Err(error) => {
                tracing::warn!(%error, "failed to read active sequence work");
                unavailable_kinds.push("sequence".to_string());
                unavailable_kinds.push("chain_generation".to_string());
                Vec::new()
            }
        };
        for row in rows.into_iter().filter(|row| {
            matches!(
                row.state,
                mold_core::chain_job::ChainJobState::Queued
                    | mold_core::chain_job::ChainJobState::Running
                    | mold_core::chain_job::ChainJobState::Paused
            )
        }) {
            represented.insert(row.id.clone());
            let ephemeral = mold_core::chain_job::ChainJobManifest::read_from_dir(&row.job_dir)
                .is_ok_and(|manifest| manifest.ephemeral);
            let cancelling = row.state == mold_core::chain_job::ChainJobState::Running
                && state
                    .chain_jobs
                    .as_ref()
                    .is_some_and(|runner| runner.is_cancelling(&row.id));
            let scheduled = scheduler.by_parent.get(&row.id);
            let phase = if cancelling {
                "cancelling"
            } else if let Some(activity) = scheduled {
                activity.phase
            } else if row.state == mold_core::chain_job::ChainJobState::Queued {
                "queued"
            } else if row.state == mold_core::chain_job::ChainJobState::Paused {
                "paused"
            } else {
                "running"
            };
            items.push(ActiveWorkItem {
                id: row.id,
                kind: if ephemeral { "generation" } else { "sequence" }.into(),
                execution: ephemeral.then(|| "chain".to_string()),
                phase: phase.into(),
                model: Some(row.model),
                created_at_unix_ms: nonnegative_ms(row.created_at_ms),
                updated_at_unix_ms: nonnegative_ms(row.updated_at_ms),
                position: None,
                current: Some(u64::from(row.current_stage)),
                total: Some(u64::from(row.stage_count)),
                stage: scheduled.and_then(|activity| activity.stage.clone()),
                preparation_progress: scheduled
                    .and_then(|activity| activity.preparation_progress.clone()),
                can_cancel: row.state != mold_core::chain_job::ChainJobState::Paused,
            });
        }
    } else {
        // A disabled or unresolved metadata database is not evidence that
        // durable sequences have disappeared. Let clients retain their last
        // verified sequence rows until this authority is available again.
        unavailable_kinds.push("sequence".to_string());
        unavailable_kinds.push("chain_generation".to_string());
    }

    // Scheduler-only work includes preparation/utility phases that never
    // enter either durable registry. Collapse partitions/stages to one parent
    // row so a batch or chain never double-renders.
    for (parent_id, activity) in &scheduler.by_parent {
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
            execution: None,
            phase: activity.phase.into(),
            model: None,
            created_at_unix_ms: observed_at_unix_ms,
            updated_at_unix_ms: observed_at_unix_ms,
            // Scheduler-only work never entered the queue registry, so it has
            // no dispatch-order place to report — the same as a sequence row.
            position: None,
            current: activity.current,
            total: activity.total,
            stage: activity.stage.clone(),
            preparation_progress: activity.preparation_progress.clone(),
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
            execution: None,
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
            stage: None,
            preparation_progress: None,
            can_cancel: true,
        });
    }

    items.sort_by_key(|item| (item.position.unwrap_or(usize::MAX), item.created_at_unix_ms));
    Json(ActiveWorkSnapshot {
        instance_id: state.instance_id.as_ref().clone(),
        observed_at_unix_ms,
        items,
        unavailable_kinds,
    })
}
