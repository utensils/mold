//! What one waiting queue row is actually doing, decided once for every Rust
//! surface.
//!
//! This is the Rust half of `studio/lib/queuePosition.ts`. The browser shells
//! resolve a queued row's copy there; the CLI and TUI resolve it here, and the
//! two must agree — the same host describing four identical queued jobs three
//! different ways is exactly the defect that policy exists to prevent.
//!
//! The load-bearing half is [`is_benign_queue_reason`]. A one-GPU host reports
//! `no_idle_device` for every job behind the running one, `warm_wait` while it
//! holds a slot for a warm device, and `lower_priority_opening` when
//! higher-priority work took the opening this pass
//! (`mold-scheduler/src/planner.rs`). Those are ordinary serialization, not
//! faults, so a row carrying one keeps counting its place in line.

use crate::types::{QueueBlockedReason, QueuePlan, QueueWorkItem};

/// Copy the whole fleet says for a reason nobody has taught it yet. Never a
/// raw underscored identifier, and never nothing.
pub const UNKNOWN_REASON_LABEL: &str = "Waiting on the host";

/// Progress of a job the host is still preparing (weights, references,
/// admission), as far as the plan reported it.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct QueuePreparation {
    /// What is being prepared, when the preparer named it.
    pub component: Option<String>,
    /// `0.0..=1.0`, or `None` when the preparer reported no byte total.
    pub fraction: Option<f32>,
}

impl QueuePreparation {
    /// Build the browser's own fraction rule from a byte pair: a zero or
    /// missing total is no evidence, never a zero percent.
    pub fn from_bytes(component: Option<String>, done: u64, total: u64) -> Self {
        Self {
            component,
            fraction: (total > 0).then(|| (done as f32 / total as f32).clamp(0.0, 1.0)),
        }
    }
}

/// `"Preparing"`, or `"Preparing · Verifying MiniMax H3 artifacts 41%"`.
pub fn preparation_label(preparation: Option<&QueuePreparation>) -> String {
    let Some(component) = preparation
        .and_then(|preparation| preparation.component.as_deref())
        .filter(|component| !component.is_empty())
    else {
        return "Preparing".to_string();
    };
    match preparation.and_then(|preparation| preparation.fraction) {
        Some(fraction) => format!("Preparing · {component} {}%", (fraction * 100.0).round()),
        None => format!("Preparing · {component}"),
    }
}

/// What each reason means to a person waiting on a print.
///
/// `None` is the load-bearing half — see the module comment. It mirrors
/// `BLOCKED_REASON_COPY` in `studio/lib/queuePosition.ts` entry for entry, and
/// the exhaustive match is what makes a new variant fail to compile until it
/// is classified here too.
fn known_copy(reason: &QueueBlockedReason) -> Option<Option<&'static str>> {
    Some(match reason {
        QueueBlockedReason::DeviceDisabled => Some("Device turned off"),
        QueueBlockedReason::DeviceDraining => Some("Device draining"),
        QueueBlockedReason::DeviceStartupExcluded => Some("Device excluded at startup"),
        QueueBlockedReason::DeviceUnavailable => Some("Waiting for a device"),
        QueueBlockedReason::DeviceDegraded => Some("Device degraded"),
        QueueBlockedReason::HardPinUnavailable => Some("Pinned device unavailable"),
        QueueBlockedReason::BackendUnsupported => Some("Not supported on this machine"),
        QueueBlockedReason::ModelNotInstalled => Some("Model not installed"),
        QueueBlockedReason::InsufficientVram => Some("Waiting for GPU memory"),
        QueueBlockedReason::InsufficientHostRam => Some("Waiting for memory"),
        QueueBlockedReason::AggregateHostRamReserved => Some("Waiting for memory"),
        QueueBlockedReason::ExecutionPlanIncompatible => Some("Cannot run as planned"),
        QueueBlockedReason::DependencyWait => None,
        QueueBlockedReason::Preparing => Some("Preparing"),
        QueueBlockedReason::WarmWait => None,
        QueueBlockedReason::QueuePaused => Some("Queue paused"),
        QueueBlockedReason::MaintenanceMode => Some("Host in maintenance"),
        QueueBlockedReason::Cancelling => Some("Cancelling"),
        QueueBlockedReason::NoSchedulableDevice => Some("No usable device"),
        QueueBlockedReason::NoIdleDevice => None,
        QueueBlockedReason::LowerPriorityOpening => None,
        QueueBlockedReason::Unknown(_) => return None,
    })
}

/// `QueueWorkItem::reason` is a display alias that may carry an
/// `AssignmentReason` rather than a blocking one. Those describe why work WON
/// a device, so they are never a reason to say anything at all.
const ASSIGNMENT_REASONS: [&str; 3] = ["priority", "starvation_forced", "warm_resident"];

/// True when a reason is ordinary bookkeeping rather than something worth
/// saying. The same predicate a device panel filters its Blocked list on, so
/// the two surfaces never disagree about what counts as blocked.
pub fn is_benign_queue_reason(reason: Option<&QueueBlockedReason>) -> bool {
    let Some(reason) = reason else {
        return true;
    };
    if let QueueBlockedReason::Unknown(raw) = reason {
        if ASSIGNMENT_REASONS.contains(&raw.as_str()) {
            return true;
        }
    }
    matches!(known_copy(reason), Some(None))
}

/// Short, plain-language copy for a queued row, or `None` when the reason is
/// ordinary bookkeeping and the row should keep counting its place in line.
pub fn blocked_reason_label(reason: Option<&QueueBlockedReason>) -> Option<String> {
    if is_benign_queue_reason(reason) {
        return None;
    }
    Some(match known_copy(reason?) {
        Some(Some(copy)) => copy.to_string(),
        _ => UNKNOWN_REASON_LABEL.to_string(),
    })
}

/// Resolved wait state for one row. Surfaces choose casing only; none of them
/// decides the vocabulary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueueWaitStatus {
    /// Parked by the host: never dispatched on its own, so never "in line".
    Held,
    /// An actionable reason outranks the position: say what to fix.
    Blocked(String),
    /// Head of the line — running next, with nobody in front.
    Next,
    /// 0-based dispatch order, so `position` is how many jobs are ahead.
    Position(usize),
    /// No evidence at all — the host never listed a position.
    Queued,
}

/// One row's evidence, in whatever shape the caller already holds.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct QueueWaitInput<'a> {
    /// The row is `held`. It still carries a listing position, and reading
    /// that as a place in line is how a parked job rendered as "Next up".
    pub held: bool,
    pub position: Option<usize>,
    pub blocked_reason: Option<&'a QueueBlockedReason>,
    pub preparation: Option<QueuePreparation>,
}

/// Resolve one waiting row. Absent evidence degrades to a plain `Queued`.
pub fn resolve_queue_wait(input: &QueueWaitInput<'_>) -> QueueWaitStatus {
    if input.held {
        return QueueWaitStatus::Held;
    }
    if matches!(input.blocked_reason, Some(QueueBlockedReason::Preparing)) {
        return QueueWaitStatus::Blocked(preparation_label(input.preparation.as_ref()));
    }
    if let Some(label) = blocked_reason_label(input.blocked_reason) {
        return QueueWaitStatus::Blocked(label);
    }
    match input.position {
        None => QueueWaitStatus::Queued,
        Some(0) => QueueWaitStatus::Next,
        Some(position) => QueueWaitStatus::Position(position),
    }
}

/// The plan's own answer for one job: the parked reason worth surfacing, and
/// the preparation report when it is preparing.
///
/// Mirrors `planBlockedReason` / `planPreparation` in
/// `studio/lib/queuePosition.ts`. A work item belongs to a job when it IS the
/// job or names it as parent — matching only `work_id` answers nothing for
/// exactly the batch parent whose phase a caller is asking about.
fn plan_reason_for_job(
    plan: &QueuePlan,
    job_id: &str,
) -> (Option<QueueBlockedReason>, Option<QueuePreparation>) {
    let owns = |item: &QueueWorkItem| item.work_id == job_id || item.parent_id == job_id;
    let mut reason = None;
    for item in plan.work_items.iter().filter(|item| owns(item)) {
        // The legacy `reason` alias is a bare string on the wire, so it is
        // parsed through the enum's own mapping rather than compared raw.
        let candidate = item
            .blocked_reason
            .clone()
            .or_else(|| item.reason.as_deref().map(QueueBlockedReason::parse));
        if candidate
            .as_ref()
            .is_some_and(|candidate| !is_benign_queue_reason(Some(candidate)))
        {
            reason = candidate;
            break;
        }
    }
    if !matches!(reason, Some(QueueBlockedReason::Preparing)) {
        return (reason, None);
    }
    let preparation = plan
        .work_items
        .iter()
        .filter(|item| owns(item))
        .find(|item| item.blocked_reason == Some(QueueBlockedReason::Preparing))
        .map(|item| match item.preparation_progress.as_ref() {
            Some(progress) => QueuePreparation::from_bytes(
                Some(progress.component.clone()),
                progress.bytes_done,
                progress.bytes_total,
            ),
            None => QueuePreparation::default(),
        });
    (reason, preparation)
}

/// Number a merged listing's rows in dispatch order, skipping held ones.
///
/// The server applies this rule per page; a client that walks every page and
/// concatenates them must restate it, or a held row at the head of the walk
/// puts the next runnable job at "#1 in line". A held row keeps the position
/// of the next runnable row so the field stays a plain index; every renderer
/// reads the state first.
pub fn assign_listed_positions(entries: &mut [crate::QueueJobEntryWire]) {
    let mut next = 0;
    for entry in entries.iter_mut() {
        entry.position = next;
        if entry.state != "held" {
            next += 1;
        }
    }
}

/// Resolve one listed row against the plan the same listing carried.
///
/// `position` is the row's own 0-based dispatch order. A host that reported
/// no plan contributes no reason, which is exactly right: absence of a plan
/// is absence of evidence, never a fault.
pub fn resolve_listed_wait(
    plan: Option<&QueuePlan>,
    job_id: &str,
    position: Option<usize>,
    held: bool,
) -> QueueWaitStatus {
    let (reason, preparation) = match plan {
        Some(plan) => plan_reason_for_job(plan, job_id),
        None => (None, None),
    };
    resolve_queue_wait(&QueueWaitInput {
        held,
        position,
        blocked_reason: reason.as_ref(),
        preparation,
    })
}

/// Sentence-case copy, the idiom every list and pill uses.
pub fn queue_wait_label(wait: &QueueWaitStatus) -> String {
    match wait {
        QueueWaitStatus::Held => "Held".to_string(),
        QueueWaitStatus::Blocked(label) => label.clone(),
        QueueWaitStatus::Next => "Next up".to_string(),
        QueueWaitStatus::Position(position) => format!("#{position} in line"),
        QueueWaitStatus::Queued => "Queued".to_string(),
    }
}

/// Compact uppercase code, for surfaces whose existing idiom is a code column.
pub fn queue_wait_code(wait: &QueueWaitStatus) -> String {
    match wait {
        QueueWaitStatus::Held => "HELD".to_string(),
        QueueWaitStatus::Blocked(label) => label.to_uppercase(),
        QueueWaitStatus::Next => "NEXT UP".to_string(),
        QueueWaitStatus::Position(position) => format!("QUEUED #{position}"),
        QueueWaitStatus::Queued => "QUEUED".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::QueueWorkItem;

    #[test]
    fn a_merged_walk_numbers_only_rows_that_can_run() {
        let row = |id: &str, state: &str| crate::QueueJobEntryWire {
            id: id.to_string(),
            state: state.to_string(),
            position: 99,
            ..Default::default()
        };
        let mut entries = vec![
            row("held-a", "held"),
            row("queued-b", "queued"),
            row("held-c", "held"),
            row("queued-d", "queued"),
        ];
        assign_listed_positions(&mut entries);
        let positions = entries
            .iter()
            .map(|entry| (entry.id.as_str(), entry.position))
            .collect::<Vec<_>>();
        assert_eq!(
            positions,
            vec![
                ("held-a", 0),
                ("queued-b", 0),
                ("held-c", 1),
                ("queued-d", 1)
            ]
        );
    }

    #[test]
    fn a_held_row_is_held_whatever_position_the_listing_gave_it() {
        let held = resolve_queue_wait(&QueueWaitInput {
            held: true,
            position: Some(0),
            blocked_reason: Some(&QueueBlockedReason::InsufficientVram),
            preparation: None,
        });
        assert_eq!(held, QueueWaitStatus::Held);
        assert_eq!(queue_wait_label(&held), "Held");
        assert_eq!(queue_wait_code(&held), "HELD");
        assert_eq!(
            resolve_listed_wait(None, "job", Some(0), true),
            QueueWaitStatus::Held
        );
        assert_eq!(
            resolve_listed_wait(None, "job", Some(0), false),
            QueueWaitStatus::Next
        );
    }

    fn wait(position: Option<usize>, reason: Option<QueueBlockedReason>) -> QueueWaitStatus {
        resolve_queue_wait(&QueueWaitInput {
            held: false,
            position,
            blocked_reason: reason.as_ref(),
            preparation: None,
        })
    }

    #[test]
    fn head_of_line_is_next_up_and_everyone_behind_counts() {
        assert_eq!(wait(Some(0), None), QueueWaitStatus::Next);
        assert_eq!(wait(Some(3), None), QueueWaitStatus::Position(3));
        assert_eq!(queue_wait_label(&wait(Some(0), None)), "Next up");
        assert_eq!(queue_wait_label(&wait(Some(3), None)), "#3 in line");
        assert_eq!(queue_wait_code(&wait(Some(0), None)), "NEXT UP");
        assert_eq!(queue_wait_code(&wait(Some(3), None)), "QUEUED #3");
    }

    #[test]
    fn absent_evidence_degrades_to_a_plain_queued() {
        assert_eq!(wait(None, None), QueueWaitStatus::Queued);
        assert_eq!(queue_wait_label(&wait(None, None)), "Queued");
        assert_eq!(queue_wait_code(&wait(None, None)), "QUEUED");
    }

    #[test]
    fn ordinary_serialization_on_a_busy_host_keeps_the_position() {
        for benign in [
            QueueBlockedReason::NoIdleDevice,
            QueueBlockedReason::LowerPriorityOpening,
            QueueBlockedReason::WarmWait,
            QueueBlockedReason::DependencyWait,
        ] {
            assert!(is_benign_queue_reason(Some(&benign)), "{benign:?}");
            assert_eq!(
                wait(Some(2), Some(benign.clone())),
                QueueWaitStatus::Position(2),
                "{benign:?} must fall through to the position"
            );
        }
    }

    #[test]
    fn an_actionable_reason_outranks_the_position() {
        assert_eq!(
            wait(Some(2), Some(QueueBlockedReason::InsufficientVram)),
            QueueWaitStatus::Blocked("Waiting for GPU memory".into())
        );
        assert_eq!(
            wait(Some(0), Some(QueueBlockedReason::QueuePaused)),
            QueueWaitStatus::Blocked("Queue paused".into())
        );
    }

    #[test]
    fn an_unknown_reason_reads_as_prose_never_an_identifier() {
        let unknown = QueueBlockedReason::Unknown("brand_new_reason".into());
        assert_eq!(
            wait(Some(1), Some(unknown)),
            QueueWaitStatus::Blocked(UNKNOWN_REASON_LABEL.into())
        );
    }

    #[test]
    fn assignment_reasons_are_never_a_reason_to_say_anything() {
        for alias in ["priority", "starvation_forced", "warm_resident"] {
            let reason = QueueBlockedReason::Unknown(alias.into());
            assert!(is_benign_queue_reason(Some(&reason)), "{alias}");
            assert_eq!(wait(Some(4), Some(reason)), QueueWaitStatus::Position(4));
        }
    }

    fn plan_with(items: Vec<QueueWorkItem>) -> QueuePlan {
        QueuePlan {
            work_items: items,
            ..Default::default()
        }
    }

    fn item(work_id: &str, reason: Option<&str>) -> QueueWorkItem {
        QueueWorkItem {
            work_id: work_id.to_string(),
            parent_id: work_id.to_string(),
            work_kind: "generation".to_string(),
            blocked_reason: reason.map(QueueBlockedReason::parse),
            ..Default::default()
        }
    }

    #[test]
    fn a_listed_row_reads_its_reason_off_the_plan_that_came_with_it() {
        let plan = plan_with(vec![
            item("other", Some("insufficient_vram")),
            item("job-1", Some("model_not_installed")),
        ]);
        assert_eq!(
            resolve_listed_wait(Some(&plan), "job-1", Some(2), false),
            QueueWaitStatus::Blocked("Model not installed".into())
        );
        // A benign reason on the plan still leaves the row counting.
        let benign = plan_with(vec![item("job-1", Some("no_idle_device"))]);
        assert_eq!(
            resolve_listed_wait(Some(&benign), "job-1", Some(2), false),
            QueueWaitStatus::Position(2)
        );
        // No plan is absence of evidence, never a fault.
        assert_eq!(
            resolve_listed_wait(None, "job-1", Some(0), false),
            QueueWaitStatus::Next
        );
    }

    #[test]
    fn a_batch_parent_is_answered_by_the_child_that_names_it() {
        let mut child = item("job-1:child", Some("insufficient_vram"));
        child.parent_id = "job-1".to_string();
        let plan = plan_with(vec![child]);
        assert_eq!(
            resolve_listed_wait(Some(&plan), "job-1", Some(1), false),
            QueueWaitStatus::Blocked("Waiting for GPU memory".into())
        );
    }

    #[test]
    fn the_legacy_reason_alias_is_parsed_not_compared_raw() {
        let mut legacy = item("job-1", None);
        legacy.reason = Some("device_disabled".to_string());
        assert_eq!(
            resolve_listed_wait(Some(&plan_with(vec![legacy])), "job-1", Some(1), false),
            QueueWaitStatus::Blocked("Device turned off".into())
        );
        // An assignment reason describes why work WON a device and is never
        // a reason to say anything.
        let mut assignment = item("job-1", None);
        assignment.reason = Some("warm_resident".to_string());
        assert_eq!(
            resolve_listed_wait(Some(&plan_with(vec![assignment])), "job-1", Some(1), false),
            QueueWaitStatus::Position(1)
        );
    }

    #[test]
    fn a_preparing_row_carries_the_plan_s_own_component_and_bytes() {
        let mut preparing = item("job-1", Some("preparing"));
        preparing.preparation_progress = Some(crate::types::QueuePreparationProgress {
            component: "MiniMax H3 artifacts".into(),
            bytes_done: 1,
            bytes_total: 4,
            phase_elapsed_ms: None,
        });
        assert_eq!(
            resolve_listed_wait(Some(&plan_with(vec![preparing])), "job-1", Some(0), false),
            QueueWaitStatus::Blocked("Preparing · MiniMax H3 artifacts 25%".into())
        );
    }

    #[test]
    fn preparing_names_its_component_and_percentage() {
        let status = resolve_queue_wait(&QueueWaitInput {
            held: false,
            position: Some(0),
            blocked_reason: Some(&QueueBlockedReason::Preparing),
            preparation: Some(QueuePreparation::from_bytes(
                Some("Verifying MiniMax H3 artifacts".into()),
                41,
                100,
            )),
        });
        assert_eq!(
            status,
            QueueWaitStatus::Blocked("Preparing · Verifying MiniMax H3 artifacts 41%".into())
        );
        // No byte total is no evidence, never a zero percent.
        assert_eq!(
            preparation_label(Some(&QueuePreparation::from_bytes(
                Some("Weights".into()),
                7,
                0
            ))),
            "Preparing · Weights"
        );
        assert_eq!(preparation_label(None), "Preparing");
    }
}
