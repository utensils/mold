//! Multi-host machine registry, generation target, and Machines state.
//!
//! This module owns everything the Machines workspace knows that isn't
//! rendering: the persisted host registry (`tui.hosts.v1` in the mold.db
//! settings table, parity with the web's `mold.web.hosts.v1`), per-host
//! API keys (`tui.host_key.<host_id>` rows, deleted on forget), the
//! sticky generation target (`tui.generate_target`), the connect-flow
//! state machine, and the background fetch tasks that feed
//! [`crate::app::BackgroundEvent`]s back into the event loop.
//!
//! Keeping this out of `app.rs` keeps the dispatcher thin; the follow-up
//! multi-host Library PR reuses [`HostRegistry`] and [`client_for`]
//! directly.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::{LazyLock, RwLock};
use tokio::sync::mpsc;

use mold_core::{DeviceInfo, DeviceState, MoldClient, ServerCapabilities, ServerStatus};
use mold_db::settings::{self as keys, Settings};
use mold_db::MetadataDb;

use crate::app::BackgroundEvent;

/// Row id of the always-present local machine.
pub(crate) const LOCAL_HOST_ID: &str = "local";

/// Display name for the local machine row (mirrors the chrome host chip).
pub(crate) fn local_display_name() -> &'static str {
    if cfg!(target_os = "macos") {
        "This Mac"
    } else {
        "This machine"
    }
}

// ── Registry ────────────────────────────────────────────────────────

/// One remembered remote host.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct HostEntry {
    /// URL slug — host-id parity with the desktop/web identity rules
    /// (`hostname-port`, lowercased, non-alphanumerics collapsed to `-`).
    pub id: String,
    /// Normalized origin URL (via [`mold_core::client::normalize_host`]).
    pub url: String,
    /// Display name; backfilled from `ServerStatus.hostname` on connect.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Last-seen `/api/status.instance_id`, used to dedupe re-adds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instance_id: Option<String>,
    /// Explicit lifecycle state. Missing in older registry JSON means
    /// connected so upgrades preserve prior behavior.
    #[serde(default = "default_connected")]
    pub connected: bool,
}

fn default_connected() -> bool {
    true
}

impl HostEntry {
    /// Name to show in rows/errors: the backfilled hostname when known,
    /// otherwise the `host:port` part of the URL.
    pub fn display_name(&self) -> String {
        self.name
            .clone()
            .filter(|n| !n.is_empty())
            .unwrap_or_else(|| host_port_label(&self.url))
    }
}

/// Why [`HostRegistry::add`] rejected an entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum AddHostError {
    /// The same server is already registered (matching `instance_id`,
    /// URL, or id slug). Carries the existing entry's display name.
    AlreadyKnown { name: String },
}

/// The persisted list of remembered remote hosts.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct HostRegistry {
    pub hosts: Vec<HostEntry>,
}

impl HostRegistry {
    /// Load the registry from the settings DB (`tui.hosts.v1`). Missing
    /// or unparseable rows yield an empty registry — the TUI still works,
    /// hosts just have to be re-added.
    pub fn load() -> Self {
        let Some(db) = open_db() else {
            return Self::default();
        };
        Self::load_from(&db)
    }

    pub(crate) fn load_from(db: &MetadataDb) -> Self {
        let hosts = Settings::new(db)
            .get_json::<Vec<HostEntry>>(keys::TUI_HOSTS_V1)
            .unwrap_or(None)
            .unwrap_or_default();
        Self { hosts }
    }

    /// Persist the registry. Silent no-op when the DB is unavailable.
    pub fn save(&self) {
        let Some(db) = open_db() else {
            return;
        };
        self.save_to(&db);
    }

    pub(crate) fn save_to(&self, db: &MetadataDb) {
        if let Err(e) = Settings::new(db).set_json(keys::TUI_HOSTS_V1, &self.hosts) {
            tracing::warn!(error = %e, "hosts: registry save failed");
        }
    }

    /// Add a host, rejecting duplicates by `instance_id`, URL, or id slug.
    pub fn add(&mut self, entry: HostEntry) -> Result<(), AddHostError> {
        let dup = self.hosts.iter().find(|h| {
            (entry.instance_id.is_some() && h.instance_id == entry.instance_id)
                || h.url == entry.url
                || h.id == entry.id
        });
        if let Some(existing) = dup {
            return Err(AddHostError::AlreadyKnown {
                name: existing.display_name(),
            });
        }
        self.hosts.push(entry);
        Ok(())
    }

    /// Remove a host by id. The caller persists via [`Self::save`];
    /// the API-key row is deleted here so it can never outlive the entry.
    pub fn forget(&mut self, id: &str) {
        self.hosts.retain(|h| h.id != id);
        delete_api_key(id);
    }

    pub fn get(&self, id: &str) -> Option<&HostEntry> {
        self.hosts.iter().find(|h| h.id == id)
    }

    /// Every connected machine, local first. The local entry is synthetic
    /// (empty URL — it is not an HTTP target); disconnected registry rows
    /// stay remembered but are deliberately absent from active operations.
    pub fn all(&self) -> Vec<HostEntry> {
        let mut out = Vec::with_capacity(self.hosts.len() + 1);
        out.push(HostEntry {
            id: LOCAL_HOST_ID.to_string(),
            url: String::new(),
            name: Some(local_display_name().to_string()),
            instance_id: None,
            connected: true,
        });
        out.extend(self.hosts.iter().filter(|host| host.connected).cloned());
        out
    }
}

/// Derive the stable slug id from an origin URL — the same rule as the
/// web registry's `hostIdFromUrl`: `hostname-port`, lowercased, runs of
/// non-alphanumerics collapsed to `-`, trimmed.
pub(crate) fn host_id_from_url(url: &str) -> String {
    let stripped = host_port_label(url);
    let mut slug = String::with_capacity(stripped.len());
    let mut last_dash = true; // suppress leading dashes
    for c in stripped.to_lowercase().chars() {
        if c.is_ascii_alphanumeric() {
            slug.push(c);
            last_dash = false;
        } else if !last_dash {
            slug.push('-');
            last_dash = true;
        }
    }
    let slug = slug.trim_matches('-').to_string();
    if slug.is_empty() {
        "host".to_string()
    } else if slug == LOCAL_HOST_ID {
        // Never collide with the reserved local row id.
        format!("{slug}-1")
    } else {
        slug
    }
}

/// The `host:port` portion of a URL — scheme and trailing slash stripped.
pub(crate) fn host_port_label(url: &str) -> String {
    let s = url.trim();
    let s = s.split_once("://").map(|(_, rest)| rest).unwrap_or(s);
    s.trim_end_matches('/').to_string()
}

// ── API keys + client construction ──────────────────────────────────

fn host_key_setting(id: &str) -> String {
    format!("{}{id}", keys::TUI_HOST_KEY_PREFIX)
}

/// Per-process credentials supplied by the command that launched this TUI.
/// They deliberately never enter SQLite: `MOLD_API_KEY` remains an ephemeral
/// environment authority unless the user explicitly saves a host key.
static SESSION_API_KEYS: LazyLock<RwLock<HashMap<String, String>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));

pub(crate) fn set_session_api_key(id: &str, key: &str) {
    if key.is_empty() {
        return;
    }
    SESSION_API_KEYS
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .insert(id.to_string(), key.to_string());
}

pub(crate) fn clear_session_api_key(id: &str) {
    SESSION_API_KEYS
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .remove(id);
}

/// Read the saved API key for a host, if any.
pub(crate) fn api_key_for(id: &str) -> Option<String> {
    if let Some(key) = SESSION_API_KEYS
        .read()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .get(id)
        .cloned()
    {
        return Some(key);
    }
    let db = open_db()?;
    Settings::new(&db)
        .get_str(&host_key_setting(id))
        .unwrap_or(None)
        .filter(|k| !k.is_empty())
}

/// Persist a host's API key (`tui.host_key.<id>`).
pub(crate) fn save_api_key(id: &str, key: &str) {
    let Some(db) = open_db() else {
        return;
    };
    if let Err(e) = Settings::new(&db).set_str(&host_key_setting(id), key) {
        tracing::warn!(error = %e, host = id, "hosts: api key save failed");
    }
}

/// Delete a host's API key row. Called from [`HostRegistry::forget`].
pub(crate) fn delete_api_key(id: &str) {
    let Some(db) = open_db() else {
        return;
    };
    let _ = Settings::new(&db).delete(&host_key_setting(id));
}

/// The one client constructor for every host-scoped HTTP call the
/// Machines workspace (and the follow-up multi-host Library) makes.
pub(crate) fn client_for(url: &str, api_key: Option<&str>) -> MoldClient {
    match api_key {
        Some(key) if !key.is_empty() => MoldClient::with_api_key(url, key.to_string()),
        _ => MoldClient::new(url),
    }
}

/// [`client_for`] with the host's saved API key looked up from the DB.
pub(crate) fn client_for_host(entry: &HostEntry) -> MoldClient {
    client_for(&entry.url, api_key_for(&entry.id).as_deref())
}

fn open_db() -> Option<MetadataDb> {
    match mold_db::open_default() {
        Ok(Some(db)) => Some(db),
        Ok(None) => None,
        Err(e) => {
            tracing::warn!(error = %e, "hosts: metadata DB open failed");
            None
        }
    }
}

// ── Generation target ───────────────────────────────────────────────

/// Where `Generate` sends work. Sticky — persisted at
/// `tui.generate_target` and set from the Machines workspace.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) enum GenTarget {
    /// Today's behavior: remote when connected, local fallback.
    #[default]
    Auto,
    /// Force the in-process local engine.
    Local,
    /// Force one registered host — no silent fallback.
    Host(String),
}

impl GenTarget {
    pub fn to_setting(&self) -> String {
        match self {
            Self::Auto => "auto".to_string(),
            Self::Local => "local".to_string(),
            Self::Host(id) => format!("host:{id}"),
        }
    }

    /// Parse the persisted form; anything unknown falls back to Auto.
    pub fn from_setting(s: &str) -> Self {
        match s.trim() {
            "local" => Self::Local,
            other => match other.strip_prefix("host:") {
                Some(id) if !id.is_empty() => Self::Host(id.to_string()),
                _ => Self::Auto,
            },
        }
    }

    pub fn load() -> Self {
        let Some(db) = open_db() else {
            return Self::Auto;
        };
        Settings::new(&db)
            .get_str(keys::TUI_GENERATE_TARGET)
            .unwrap_or(None)
            .map(|s| Self::from_setting(&s))
            .unwrap_or_default()
    }

    pub fn save(&self) {
        let Some(db) = open_db() else {
            return;
        };
        if let Err(e) = Settings::new(&db).set_str(keys::TUI_GENERATE_TARGET, &self.to_setting()) {
            tracing::warn!(error = %e, "hosts: generate target save failed");
        }
    }
}

// ── Machines view state ─────────────────────────────────────────────

/// Connection health of a registered host row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum HostHealth {
    Ready,
    Connecting,
    Offline(String),
}

/// Last-known status for one host.
#[derive(Debug)]
pub(crate) struct HostStatus {
    pub health: HostHealth,
    pub status: Option<Box<ServerStatus>>,
}

/// Which machine a Machines row refers to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum MachineRowId {
    Local,
    Host(String),
}

/// Which pane of the Machines workspace holds the keyboard.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum MachinesFocus {
    #[default]
    HostList,
    Detail,
}

/// How often hosts are re-polled while due (driven by the 2 s resource
/// tick in `lib.rs`, so effective cadence is the next tick ≥ this).
pub(crate) const POLL_INTERVAL: std::time::Duration = std::time::Duration::from_secs(4);

/// A telemetry poll gets at most one full polling interval. This is derived
/// from the scheduling contract: if a request cannot finish before the host
/// would otherwise be due again, retaining its socket would make fixed-rate
/// polling accumulate work instead of observing the host.
const POLL_TIMEOUT: std::time::Duration = POLL_INTERVAL;

/// State of the Machines workspace.
#[derive(Default)]
pub(crate) struct MachinesState {
    pub registry: HostRegistry,
    /// Selected row index: 0 = local, 1.. = `registry.hosts` order.
    pub selected: usize,
    pub focus: MachinesFocus,
    /// Last-known status per host id.
    pub statuses: HashMap<String, HostStatus>,
    /// Authoritative `/api/devices` snapshots, keyed by machine row id.
    pub devices: HashMap<String, DeviceState>,
    /// Capability gate paired with each device inventory.
    pub capabilities: HashMap<String, ServerCapabilities>,
    /// Last accepted lifecycle mutation, keyed by machine row id. This is
    /// intentionally non-error feedback and survives the authoritative poll
    /// that immediately follows the mutation response.
    pub device_feedback: HashMap<String, String>,
    /// Selected device in the detail pane. `g` cycles this independently of
    /// the queue-row selection, preserving the existing Up/Down queue keys.
    pub device_selected: usize,
    /// Queue snapshot for the selected remote host.
    pub queue: Option<(String, mold_core::QueueListingWire)>,
    /// Single-flight guard for explicit continuation reads.
    pub queue_loading_more: bool,
    pub(crate) queue_continuation: Option<(String, String)>,
    pub(crate) queue_extended: bool,
    /// Job selection inside the detail pane's queue lanes.
    pub queue_selected: usize,
    /// Periodic polling owns at most one task/socket group per host. The next
    /// due time is written only when that task finishes (including timeout),
    /// so scheduling is completion-relative instead of fixed-rate.
    pub(crate) polls_in_flight: HashSet<String>,
    pub(crate) poll_next_due: HashMap<String, std::time::Instant>,
    pub(crate) queue_refresh_pending: HashSet<String>,
}

/// What [`MachinesState::poll_plan`] decided to fetch this tick.
#[derive(Debug, Default)]
pub(crate) struct PollPlan {
    pub hosts: Vec<HostPollRequest>,
}

#[derive(Debug)]
pub(crate) struct HostPollRequest {
    pub entry: HostEntry,
    pub include_queue: bool,
}

impl MachinesState {
    /// Load persisted state (registry) — view state starts fresh.
    pub fn load() -> Self {
        Self {
            registry: HostRegistry::load(),
            ..Self::default()
        }
    }

    pub fn row_count(&self) -> usize {
        1 + self.registry.hosts.len()
    }

    /// The machine the selected row refers to. Row 0 is always local.
    pub fn selected_row(&self) -> MachineRowId {
        if self.selected == 0 {
            MachineRowId::Local
        } else {
            self.registry
                .hosts
                .get(self.selected - 1)
                .map(|h| MachineRowId::Host(h.id.clone()))
                .unwrap_or(MachineRowId::Local)
        }
    }

    /// The registry entry behind the selected row, when it is remote.
    pub fn selected_host(&self) -> Option<&HostEntry> {
        match self.selected {
            0 => None,
            n => self.registry.hosts.get(n - 1),
        }
    }

    /// Move the selection up. Returns true when it changed.
    pub fn select_prev(&mut self) -> bool {
        if self.selected > 0 {
            self.selected -= 1;
            self.on_selection_changed();
            true
        } else {
            false
        }
    }

    /// Move the selection down. Returns true when it changed.
    pub fn select_next(&mut self) -> bool {
        if self.selected + 1 < self.row_count() {
            self.selected += 1;
            self.on_selection_changed();
            true
        } else {
            false
        }
    }

    fn on_selection_changed(&mut self) {
        self.queue = None;
        self.queue_loading_more = false;
        self.queue_continuation = None;
        self.queue_extended = false;
        self.queue_selected = 0;
        self.device_selected = 0;
    }

    pub fn queue_select_prev(&mut self) {
        self.queue_selected = self.queue_selected.saturating_sub(1);
    }

    pub fn queue_select_next(&mut self) {
        let len = self
            .queue
            .as_ref()
            .map(|(_, q)| q.entries.len())
            .unwrap_or(0);
        if self.queue_selected + 1 < len {
            self.queue_selected += 1;
        }
    }

    /// The cancellable job under the queue selection. Running rows require
    /// the additive host capability so older servers retain their safe gate.
    pub fn selected_cancellable_job(&self) -> Option<(String, mold_core::QueueJobEntryWire)> {
        let (host_id, listing) = self.queue.as_ref()?;
        let job = listing.entries.get(self.queue_selected)?;
        let running_supported = job.state == "running"
            && self
                .capabilities
                .get(host_id)
                .is_some_and(|caps| caps.queue.cooperative_cancellation);
        if job.state == "queued" || running_supported {
            Some((host_id.clone(), job.clone()))
        } else {
            None
        }
    }

    fn selected_machine_id(&self) -> String {
        match self.selected_row() {
            MachineRowId::Local => LOCAL_HOST_ID.to_string(),
            MachineRowId::Host(id) => id,
        }
    }

    pub fn selected_device(&self) -> Option<&DeviceInfo> {
        let id = self.selected_machine_id();
        self.devices.get(&id)?.devices.get(self.device_selected)
    }

    pub fn can_mutate_selected_device(&self) -> bool {
        let device = match self.selected_device() {
            Some(device) => device,
            None => return false,
        };
        if device.admin_state == mold_core::DeviceAdminState::StartupExcluded
            || device.restart_required
        {
            return false;
        }
        let Some(capabilities) = self.capabilities.get(&self.selected_machine_id()) else {
            return false;
        };
        (capabilities.devices.lifecycle && capabilities.dispatch.v2_authoritative)
            || (!device.desired_enabled && capabilities.devices.restart_enable)
    }

    pub fn selected_device_uses_restart_enable(&self) -> bool {
        let Some(device) = self.selected_device() else {
            return false;
        };
        if device.desired_enabled || device.restart_required {
            return false;
        }
        let Some(capabilities) = self.capabilities.get(&self.selected_machine_id()) else {
            return false;
        };
        let live_lifecycle =
            capabilities.devices.lifecycle && capabilities.dispatch.v2_authoritative;
        !live_lifecycle && capabilities.devices.restart_enable
    }

    pub fn selected_device_restart_required(&self) -> bool {
        self.selected_device()
            .is_some_and(|device| device.restart_required)
    }

    pub fn select_next_device(&mut self) {
        let id = self.selected_machine_id();
        let len = self
            .devices
            .get(&id)
            .map(|state| state.devices.len())
            .unwrap_or(0);
        if len > 0 {
            self.device_selected = (self.device_selected + 1) % len;
        }
    }

    pub fn select_prev_device(&mut self) {
        let id = self.selected_machine_id();
        let len = self
            .devices
            .get(&id)
            .map(|state| state.devices.len())
            .unwrap_or(0);
        if len > 0 {
            self.device_selected = (self.device_selected + len - 1) % len;
        }
    }

    pub fn apply_devices(&mut self, host_id: String, devices: Option<DeviceState>) {
        match devices {
            Some(devices) => {
                if self.device_selected >= devices.devices.len() {
                    self.device_selected = devices.devices.len().saturating_sub(1);
                }
                self.devices.insert(host_id, devices);
            }
            None => {
                self.devices.remove(&host_id);
            }
        }
    }

    pub fn apply_device_mutation(&mut self, host_id: String, device: DeviceInfo) {
        let feedback = if device.restart_required {
            format!("{} enabled on restart", device.name)
        } else if device.desired_enabled {
            format!("{} enabled", device.name)
        } else if device.admin_state == mold_core::DeviceAdminState::Draining {
            format!("{} will be disabled after current work", device.name)
        } else {
            format!("{} disabled", device.name)
        };
        let state = self.devices.entry(host_id.clone()).or_default();
        if let Some(existing) = state
            .devices
            .iter_mut()
            .find(|existing| existing.id == device.id)
        {
            *existing = device;
        } else {
            state.devices.push(device);
        }
        self.device_feedback.insert(host_id, feedback);
    }

    pub fn apply_capabilities(
        &mut self,
        host_id: String,
        capabilities: Option<ServerCapabilities>,
    ) {
        match capabilities {
            Some(capabilities) => {
                self.capabilities.insert(host_id, capabilities);
            }
            None => {
                self.capabilities.remove(&host_id);
            }
        }
    }

    /// The target Enter on the selected row switches to. Pressing Enter
    /// on the row that is already the target reverts to Auto so the
    /// sticky pick is always escapable.
    pub fn target_for_selected(&self, current: &GenTarget) -> GenTarget {
        let picked = match self.selected_row() {
            MachineRowId::Local => GenTarget::Local,
            MachineRowId::Host(id) => {
                if self.registry.get(&id).is_some_and(|host| !host.connected) {
                    return current.clone();
                }
                GenTarget::Host(id)
            }
        };
        if picked == *current {
            GenTarget::Auto
        } else {
            picked
        }
    }

    /// Force every idle host to be due immediately. An in-flight host remains
    /// single-flight and will be re-scheduled when its current poll completes.
    pub fn force_poll(&mut self) {
        self.poll_next_due.clear();
        self.queue_refresh_pending
            .extend(self.polls_in_flight.iter().cloned());
    }

    pub fn request_queue_refresh(&mut self, host_id: &str) {
        self.poll_next_due.remove(host_id);
        if self.polls_in_flight.contains(host_id) {
            self.queue_refresh_pending.insert(host_id.to_string());
        }
    }

    /// Decide which hosts to fetch this tick: every registered host (and
    /// the selected host's queue) while Machines is active, only the
    /// generation-target host otherwise. Marks not-yet-seen hosts as
    /// Connecting so their rows render honestly while the fetch runs.
    pub fn poll_plan(&mut self, machines_active: bool, target: &GenTarget) -> PollPlan {
        let mut plan = PollPlan::default();
        let mut desired = if machines_active {
            self.registry
                .hosts
                .iter()
                .filter(|host| host.connected)
                .cloned()
                .collect::<Vec<_>>()
        } else if let GenTarget::Host(id) = target {
            self.registry
                .get(id)
                .filter(|host| host.connected)
                .cloned()
                .into_iter()
                .collect()
        } else {
            Vec::new()
        };
        let selected_id = machines_active
            .then(|| self.selected_host().map(|host| host.id.clone()))
            .flatten();
        let now = std::time::Instant::now();
        desired.retain(|entry| {
            !self.polls_in_flight.contains(&entry.id)
                && self
                    .poll_next_due
                    .get(&entry.id)
                    .is_none_or(|due| *due <= now)
        });
        for entry in desired {
            self.statuses.entry(entry.id.clone()).or_insert(HostStatus {
                health: HostHealth::Connecting,
                status: None,
            });
            self.polls_in_flight.insert(entry.id.clone());
            plan.hosts.push(HostPollRequest {
                include_queue: selected_id.as_deref() == Some(entry.id.as_str()),
                entry,
            });
        }
        plan
    }

    /// Release one host's polling lease and schedule from completion, not
    /// from start. Returns true when a refresh requested while the poll was in
    /// flight should be dispatched immediately.
    pub fn finish_poll(&mut self, host_id: &str) -> bool {
        self.polls_in_flight.remove(host_id);
        if self.queue_refresh_pending.remove(host_id) {
            self.poll_next_due.remove(host_id);
            true
        } else {
            self.poll_next_due.insert(
                host_id.to_string(),
                std::time::Instant::now() + POLL_INTERVAL,
            );
            false
        }
    }

    /// Apply a background status result. `None` marks the row Offline —
    /// it stays listed and self-heals on a later successful poll.
    pub fn apply_status(&mut self, host_id: String, status: Option<Box<ServerStatus>>) {
        let entry = match status {
            Some(s) => HostStatus {
                health: HostHealth::Ready,
                status: Some(s),
            },
            None => HostStatus {
                health: HostHealth::Offline("unreachable".to_string()),
                status: self.statuses.remove(&host_id).and_then(|p| p.status),
            },
        };
        self.statuses.insert(host_id, entry);
    }

    /// Apply a background queue result for a host. Ignored when the
    /// selection moved to a different host in the meantime.
    pub fn apply_queue(&mut self, host_id: String, queue: Option<mold_core::QueueListingWire>) {
        let selected_id = match self.selected_row() {
            MachineRowId::Host(id) => id,
            MachineRowId::Local => return,
        };
        if selected_id != host_id {
            return;
        }
        match (self.queue.take(), queue) {
            (Some((current_host, current)), Some(mut head)) if current_host == host_id => {
                let selected_id = current
                    .entries
                    .get(self.queue_selected)
                    .map(|entry| entry.id.clone());
                let mut seen = head
                    .entries
                    .iter()
                    .map(|entry| entry.id.clone())
                    .collect::<HashSet<_>>();
                let preserve_loaded_window = self.queue_extended || self.queue_loading_more;
                head.entries.extend(
                    current
                        .entries
                        .into_iter()
                        .filter(|entry| preserve_loaded_window || entry.state == "running")
                        .filter(|entry| seen.insert(entry.id.clone())),
                );
                if self.queue_extended {
                    head.page = current.page;
                }
                self.queue_selected = selected_id
                    .as_ref()
                    .and_then(|id| head.entries.iter().position(|entry| &entry.id == id))
                    .unwrap_or_else(|| {
                        self.queue_selected
                            .min(head.entries.len().saturating_sub(1))
                    });
                self.queue = Some((host_id, head));
            }
            (_, Some(listing)) => {
                self.queue_selected = self
                    .queue_selected
                    .min(listing.entries.len().saturating_sub(1));
                self.queue = Some((host_id, listing));
                self.queue_extended = false;
            }
            (current, None) => self.queue = current,
        }
    }

    /// Capture the exact continuation authority and mark it in flight.
    pub fn begin_queue_continuation(&mut self) -> Option<(HostEntry, usize, String)> {
        if self.queue_loading_more {
            return None;
        }
        let entry = self.selected_host()?.clone();
        let (host_id, listing) = self.queue.as_ref()?;
        if *host_id != entry.id {
            return None;
        }
        let page = listing.page.as_ref()?;
        let cursor = page.next_cursor.clone()?;
        self.queue_loading_more = true;
        self.queue_continuation = Some((entry.id.clone(), cursor.clone()));
        Some((entry, page.limit, cursor))
    }

    /// Append a continuation only if the selected host still exposes the
    /// cursor this request followed. A concurrent head poll invalidates it.
    pub fn apply_queue_continuation(
        &mut self,
        host_id: String,
        cursor: &str,
        queue: Option<mold_core::QueueListingWire>,
    ) {
        if self
            .queue_continuation
            .as_ref()
            .map(|(host, next)| (host.as_str(), next.as_str()))
            != Some((host_id.as_str(), cursor))
        {
            return;
        }
        self.queue_loading_more = false;
        self.queue_continuation = None;
        let Some((current_host, current)) = self.queue.as_mut() else {
            return;
        };
        if *current_host != host_id {
            return;
        }
        let Some(mut continuation) = queue else {
            return;
        };
        let mut seen = current
            .entries
            .iter()
            .map(|entry| entry.id.clone())
            .collect::<std::collections::HashSet<_>>();
        current.entries.extend(
            continuation
                .entries
                .drain(..)
                .filter(|entry| seen.insert(entry.id.clone())),
        );
        current.page = continuation.page;
        self.queue_extended = true;
        if continuation.plan.is_some() {
            current.plan = continuation.plan;
        }
    }

    /// Forget a host: registry row, API key, cached status/queue, and
    /// selection clamp. The caller resets the generation target if it
    /// pointed at this host.
    pub fn forget(&mut self, id: &str) {
        self.registry.forget(id);
        self.registry.save();
        self.statuses.remove(id);
        self.devices.remove(id);
        if let Some((qid, _)) = &self.queue {
            if qid == id {
                self.queue = None;
                self.queue_selected = 0;
            }
        }
        self.capabilities.remove(id);
        self.device_feedback.remove(id);
        self.polls_in_flight.remove(id);
        self.poll_next_due.remove(id);
        self.queue_refresh_pending.remove(id);
        if self.selected >= self.row_count() {
            self.selected = self.row_count() - 1;
        }
    }

    /// Toggle a remembered host in or out of every active mix. Returns the
    /// new connected state; the saved API key and registry row are untouched.
    pub fn toggle_connection(&mut self, id: &str) -> Option<bool> {
        let host = self.registry.hosts.iter_mut().find(|host| host.id == id)?;
        host.connected = !host.connected;
        let connected = host.connected;
        self.registry.save();
        self.statuses.remove(id);
        self.devices.remove(id);
        self.capabilities.remove(id);
        self.device_feedback.remove(id);
        if self
            .queue
            .as_ref()
            .is_some_and(|(host_id, _)| host_id == id)
        {
            self.queue = None;
            self.queue_selected = 0;
        }
        self.force_poll();
        Some(connected)
    }

    /// Finish a successful connect test: dedupe, register, persist, save
    /// the key, and select the new row. Returns the new host id.
    pub fn complete_connect(
        &mut self,
        url: &str,
        api_key: Option<&str>,
        status: Box<ServerStatus>,
    ) -> Result<String, AddHostError> {
        let id = host_id_from_url(url);
        let entry = HostEntry {
            id: id.clone(),
            url: url.to_string(),
            name: status.hostname.clone(),
            instance_id: status.instance_id.clone(),
            connected: true,
        };
        self.registry.add(entry)?;
        self.registry.save();
        if let Some(key) = api_key.filter(|k| !k.is_empty()) {
            save_api_key(&id, key);
        }
        self.statuses.insert(
            id.clone(),
            HostStatus {
                health: HostHealth::Ready,
                status: Some(status),
            },
        );
        self.selected = self.row_count() - 1;
        self.on_selection_changed();
        Ok(id)
    }
}

// ── Connect-a-machine flow ──────────────────────────────────────────

/// Step of the stepped connect form.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum ConnectStep {
    #[default]
    Url,
    ApiKey,
    Testing,
    Failed,
}

/// State of the `Popup::MachineConnect` form.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct ConnectForm {
    pub step: ConnectStep,
    pub name: String,
    pub url: String,
    pub api_key: String,
    pub error: Option<String>,
}

/// Input fed into [`connect_advance`] — key events plus the async test
/// results routed back through the background channel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ConnectInput<'a> {
    Char(char),
    Backspace,
    Enter,
    Esc,
    TestOk { hostname: Option<&'a str> },
    TestErr(&'a str),
}

/// Side effect the caller must perform after [`connect_advance`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ConnectEffect {
    None,
    /// Spawn the `server_status` test against `form.url` (+ key).
    StartTest,
    /// The test succeeded — register the host (`MachinesState::complete_connect`).
    Save,
    /// Close the popup.
    Close,
}

/// Pure state machine for the connect flow: Url → ApiKey (optional,
/// Enter skips) → Testing → saved, or Failed with retry (Enter) / edit
/// (`e`/Backspace back to Url) / Esc.
pub(crate) fn connect_advance(form: &mut ConnectForm, input: ConnectInput) -> ConnectEffect {
    match form.step {
        ConnectStep::Url => match input {
            ConnectInput::Char(c) => {
                form.url.push(c);
                ConnectEffect::None
            }
            ConnectInput::Backspace => {
                form.url.pop();
                ConnectEffect::None
            }
            ConnectInput::Enter => {
                let trimmed = form.url.trim().to_string();
                if trimmed.is_empty() {
                    return ConnectEffect::None;
                }
                form.url = mold_core::client::normalize_host(&trimmed);
                form.error = None;
                form.step = ConnectStep::ApiKey;
                ConnectEffect::None
            }
            ConnectInput::Esc => ConnectEffect::Close,
            _ => ConnectEffect::None,
        },
        ConnectStep::ApiKey => match input {
            ConnectInput::Char(c) => {
                form.api_key.push(c);
                ConnectEffect::None
            }
            ConnectInput::Backspace => {
                form.api_key.pop();
                ConnectEffect::None
            }
            ConnectInput::Enter => {
                // Empty key = no auth — Enter skips straight to testing.
                form.step = ConnectStep::Testing;
                form.error = None;
                ConnectEffect::StartTest
            }
            ConnectInput::Esc => {
                form.step = ConnectStep::Url;
                ConnectEffect::None
            }
            _ => ConnectEffect::None,
        },
        ConnectStep::Testing => match input {
            ConnectInput::TestOk { hostname } => {
                form.name = hostname
                    .map(|h| h.to_string())
                    .unwrap_or_else(|| host_port_label(&form.url));
                ConnectEffect::Save
            }
            ConnectInput::TestErr(e) => {
                form.step = ConnectStep::Failed;
                form.error = Some(e.to_string());
                ConnectEffect::None
            }
            ConnectInput::Esc => ConnectEffect::Close,
            _ => ConnectEffect::None,
        },
        ConnectStep::Failed => match input {
            ConnectInput::Enter => {
                form.step = ConnectStep::Testing;
                form.error = None;
                ConnectEffect::StartTest
            }
            ConnectInput::Char('e') | ConnectInput::Backspace => {
                form.step = ConnectStep::Url;
                form.error = None;
                ConnectEffect::None
            }
            ConnectInput::Esc => ConnectEffect::Close,
            _ => ConnectEffect::None,
        },
    }
}

// ── Background fetch tasks ──────────────────────────────────────────

/// Fetch one host's complete periodic snapshot under a single polling lease.
/// Every independent request is cancelled at the polling interval, and the
/// completion event always releases the lease so a wedged host cannot grow a
/// new task/socket group on every UI tick.
pub(crate) async fn fetch_host_poll(
    entry: HostEntry,
    include_queue: bool,
    tx: mpsc::UnboundedSender<BackgroundEvent>,
) {
    fetch_host_poll_with_timeout(entry, include_queue, tx, POLL_TIMEOUT).await;
}

async fn fetch_host_poll_with_timeout(
    entry: HostEntry,
    include_queue: bool,
    tx: mpsc::UnboundedSender<BackgroundEvent>,
    timeout: std::time::Duration,
) {
    let client = client_for_host(&entry);
    let status = tokio::time::timeout(timeout, client.server_status());
    let devices = tokio::time::timeout(timeout, client.devices());
    let capabilities = tokio::time::timeout(timeout, client.server_capabilities());
    let queue = async {
        if include_queue {
            tokio::time::timeout(timeout, client.list_queue())
                .await
                .ok()
                .and_then(|result| result.ok())
        } else {
            None
        }
    };
    let (status, devices, capabilities, queue) = tokio::join!(status, devices, capabilities, queue);
    let status = status.ok().and_then(|result| result.ok()).map(Box::new);
    let devices = devices.ok().and_then(|result| result.ok());
    let capabilities = capabilities
        .ok()
        .and_then(|result| result.ok())
        .map(Box::new);
    let _ = tx.send(BackgroundEvent::HostStatusUpdate {
        host_id: entry.id.clone(),
        status,
    });
    let _ = tx.send(BackgroundEvent::HostDevicesUpdate {
        host_id: entry.id.clone(),
        devices,
    });
    let _ = tx.send(BackgroundEvent::HostCapabilitiesUpdate {
        host_id: entry.id.clone(),
        capabilities,
    });
    if include_queue {
        let _ = tx.send(BackgroundEvent::HostQueueUpdate {
            host_id: entry.id.clone(),
            queue,
        });
    }
    let _ = tx.send(BackgroundEvent::HostPollFinished { host_id: entry.id });
}

pub(crate) async fn set_host_device_enabled(
    entry: HostEntry,
    device_id: String,
    enabled: bool,
    tx: mpsc::UnboundedSender<BackgroundEvent>,
) {
    let client = client_for_host(&entry);
    let result = client.set_device_enabled(&device_id, enabled).await;
    match result {
        Ok(device) => {
            let _ = tx.send(BackgroundEvent::HostDeviceMutationApplied {
                host_id: entry.id.clone(),
                device: Box::new(device),
            });
        }
        Err(error) => {
            let _ = tx.send(BackgroundEvent::Error(format!(
                "GPU lifecycle update failed: {error:#}"
            )));
        }
    }
    let _ = tx.send(BackgroundEvent::HostDevicesUpdate {
        host_id: entry.id,
        devices: client.devices().await.ok(),
    });
}

pub(crate) async fn set_local_device_enabled(
    url: String,
    device_id: String,
    enabled: bool,
    tx: mpsc::UnboundedSender<BackgroundEvent>,
) {
    let api_key = std::env::var("MOLD_API_KEY").ok();
    let client = client_for(&url, api_key.as_deref());
    let result = client.set_device_enabled(&device_id, enabled).await;
    match result {
        Ok(device) => {
            let _ = tx.send(BackgroundEvent::HostDeviceMutationApplied {
                host_id: LOCAL_HOST_ID.to_string(),
                device: Box::new(device),
            });
        }
        Err(error) => {
            let _ = tx.send(BackgroundEvent::Error(format!(
                "GPU lifecycle update failed: {error:#}"
            )));
        }
    }
    let _ = tx.send(BackgroundEvent::HostDevicesUpdate {
        host_id: LOCAL_HOST_ID.to_string(),
        devices: client.devices().await.ok(),
    });
}

/// Fetch the queue listing for one host and report back.
pub(crate) async fn fetch_host_queue(entry: HostEntry, tx: mpsc::UnboundedSender<BackgroundEvent>) {
    let client = client_for_host(&entry);
    let (queue, devices, capabilities) =
        tokio::join!(client.list_queue(), client.devices(), client.capabilities());
    let host_id = entry.id;
    let _ = tx.send(BackgroundEvent::HostQueueUpdate {
        host_id: host_id.clone(),
        queue: queue.ok(),
    });
    let _ = tx.send(BackgroundEvent::HostDevicesUpdate {
        host_id: host_id.clone(),
        devices: devices.ok(),
    });
    let _ = tx.send(BackgroundEvent::HostCapabilitiesUpdate {
        host_id,
        capabilities: capabilities.ok().map(Box::new),
    });
}

/// Fetch one user-requested durable continuation page. Periodic polling never
/// calls this and therefore remains bounded to the host's live capacity.
pub(crate) async fn fetch_host_queue_page(
    entry: HostEntry,
    limit: usize,
    cursor: String,
    tx: mpsc::UnboundedSender<BackgroundEvent>,
) {
    let client = client_for_host(&entry);
    let queue = tokio::time::timeout(POLL_TIMEOUT, client.list_queue_page(limit, Some(&cursor)))
        .await
        .ok()
        .and_then(|result| result.ok());
    let _ = tx.send(BackgroundEvent::HostQueuePageLoaded {
        host_id: entry.id,
        cursor,
        queue,
    });
}

/// Run the connect-flow test fetch and report back.
pub(crate) async fn test_connection(
    url: String,
    api_key: Option<String>,
    tx: mpsc::UnboundedSender<BackgroundEvent>,
) {
    let client = client_for(&url, api_key.as_deref());
    let result = client
        .server_status()
        .await
        .map(Box::new)
        .map_err(|e| format!("{e:#}"));
    let _ = tx.send(BackgroundEvent::MachineConnectTested {
        url,
        api_key,
        result,
    });
}

/// Cancel a queued job on a host, then refresh that host's queue. A
/// failure surfaces the server's response body (e.g. the 409
/// already-running reason) as an error event.
pub(crate) async fn cancel_host_job(
    entry: HostEntry,
    job_id: String,
    tx: mpsc::UnboundedSender<BackgroundEvent>,
) {
    let client = client_for_host(&entry);
    if let Err(e) = client.cancel_queue_job(&job_id).await {
        let _ = tx.send(BackgroundEvent::Error(format!("Cancel failed: {e:#}")));
    }
    let queue = client.list_queue().await.ok();
    let devices = client.devices().await.ok();
    let host_id = entry.id;
    let _ = tx.send(BackgroundEvent::HostQueueUpdate {
        host_id: host_id.clone(),
        queue,
    });
    let _ = tx.send(BackgroundEvent::HostDevicesUpdate { host_id, devices });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_env::with_isolated_env;
    use serial_test::serial;

    fn entry(id: &str, url: &str) -> HostEntry {
        HostEntry {
            id: id.to_string(),
            url: url.to_string(),
            name: None,
            instance_id: None,
            connected: true,
        }
    }

    fn status_with(hostname: Option<&str>, instance_id: Option<&str>) -> Box<ServerStatus> {
        Box::new(ServerStatus {
            version: "0.20.2".into(),
            git_sha: None,
            build_date: None,
            models_loaded: vec![],
            busy: false,
            current_generation: None,
            gpu_info: None,
            uptime_secs: 0,
            hostname: hostname.map(String::from),
            memory_status: None,
            gpus: None,
            queue_depth: None,
            queue_capacity: None,
            queue_paused: None,
            instance_id: instance_id.map(String::from),
            models_disk: None,
            host_memory: None,
            durable_media: None,
        })
    }

    // ── host id slug parity ─────────────────────────────────────

    #[test]
    fn host_id_is_url_slug_with_port() {
        assert_eq!(host_id_from_url("http://hal9000:7680"), "hal9000-7680");
        assert_eq!(
            host_id_from_url("https://Mold.Example.COM"),
            "mold-example-com"
        );
        assert_eq!(
            host_id_from_url("http://192.168.1.5:7680"),
            "192-168-1-5-7680"
        );
    }

    #[test]
    fn host_id_never_collides_with_local_row() {
        assert_eq!(host_id_from_url("http://local"), "local-1");
        assert_eq!(host_id_from_url(""), "host");
    }

    #[test]
    fn host_port_label_strips_scheme_and_slash() {
        assert_eq!(host_port_label("http://hal9000:7680/"), "hal9000:7680");
        assert_eq!(host_port_label("hal9000:7680"), "hal9000:7680");
        assert_eq!(
            host_port_label("https://mold.example.com"),
            "mold.example.com"
        );
    }

    // ── registry ────────────────────────────────────────────────

    #[test]
    #[serial(mold_env)]
    fn registry_round_trips_through_db_and_forget_deletes_key() {
        with_isolated_env(|_home| {
            let mut reg = HostRegistry::default();
            reg.add(entry("hal9000-7680", "http://hal9000:7680"))
                .unwrap();
            reg.add(HostEntry {
                id: "bender-7680".into(),
                url: "http://bender:7680".into(),
                name: Some("bender".into()),
                instance_id: Some("uuid-b".into()),
                connected: true,
            })
            .unwrap();
            reg.save();
            save_api_key("bender-7680", "secret");
            assert_eq!(api_key_for("bender-7680").as_deref(), Some("secret"));

            let mut loaded = HostRegistry::load();
            assert_eq!(loaded, reg);

            loaded.forget("bender-7680");
            loaded.save();
            assert_eq!(
                api_key_for("bender-7680"),
                None,
                "forget deletes the key row"
            );
            let reloaded = HostRegistry::load();
            assert_eq!(reloaded.hosts.len(), 1);
            assert_eq!(reloaded.hosts[0].id, "hal9000-7680");
        });
    }

    #[test]
    fn add_rejects_duplicate_instance_id_url_and_id() {
        let mut reg = HostRegistry::default();
        reg.add(HostEntry {
            id: "hal9000-7680".into(),
            url: "http://hal9000:7680".into(),
            name: Some("hal9000".into()),
            instance_id: Some("uuid-a".into()),
            connected: true,
        })
        .unwrap();

        // Same instance id via a different URL — one box, two routes.
        let err = reg
            .add(HostEntry {
                id: "192-168-1-5-7680".into(),
                url: "http://192.168.1.5:7680".into(),
                name: None,
                instance_id: Some("uuid-a".into()),
                connected: true,
            })
            .unwrap_err();
        assert_eq!(
            err,
            AddHostError::AlreadyKnown {
                name: "hal9000".into()
            }
        );

        // Same URL.
        assert!(reg.add(entry("other", "http://hal9000:7680")).is_err());
        // Same id slug.
        assert!(reg
            .add(entry("hal9000-7680", "http://elsewhere:1"))
            .is_err());
        assert_eq!(reg.hosts.len(), 1);
    }

    #[test]
    fn all_lists_local_first() {
        let mut reg = HostRegistry::default();
        reg.add(entry("hal9000-7680", "http://hal9000:7680"))
            .unwrap();
        let all = reg.all();
        assert_eq!(all.len(), 2);
        assert_eq!(all[0].id, LOCAL_HOST_ID);
        assert_eq!(all[0].name.as_deref(), Some(local_display_name()));
        assert_eq!(all[1].id, "hal9000-7680");
    }

    #[test]
    #[serial(mold_env)]
    fn disconnected_host_stays_remembered_but_out_of_active_work() {
        with_isolated_env(|_home| {
            let mut st = MachinesState::default();
            st.registry
                .add(entry("hal9000-7680", "http://hal9000:7680"))
                .unwrap();
            st.registry.save();

            assert_eq!(st.toggle_connection("hal9000-7680"), Some(false));
            assert_eq!(st.registry.hosts.len(), 1);
            assert!(st
                .registry
                .all()
                .iter()
                .all(|host| host.id != "hal9000-7680"));
            assert!(st.poll_plan(true, &GenTarget::Auto).hosts.is_empty());
            st.finish_poll("hal9000-7680");
            assert!(!HostRegistry::load().get("hal9000-7680").unwrap().connected);

            assert_eq!(st.toggle_connection("hal9000-7680"), Some(true));
            assert_eq!(
                st.poll_plan(true, &GenTarget::Auto).hosts[0].entry.id,
                "hal9000-7680"
            );
        });
    }

    // ── generation target ───────────────────────────────────────

    #[test]
    fn gen_target_serialization_round_trips() {
        for t in [
            GenTarget::Auto,
            GenTarget::Local,
            GenTarget::Host("hal9000-7680".into()),
        ] {
            assert_eq!(GenTarget::from_setting(&t.to_setting()), t);
        }
        assert_eq!(GenTarget::from_setting("auto"), GenTarget::Auto);
        assert_eq!(GenTarget::from_setting("host:"), GenTarget::Auto);
        assert_eq!(GenTarget::from_setting("garbage"), GenTarget::Auto);
    }

    #[test]
    #[serial(mold_env)]
    fn gen_target_persists_through_db() {
        with_isolated_env(|_home| {
            GenTarget::Host("hal9000-7680".into()).save();
            assert_eq!(GenTarget::load(), GenTarget::Host("hal9000-7680".into()));
            GenTarget::Auto.save();
            assert_eq!(GenTarget::load(), GenTarget::Auto);
        });
    }

    // ── machines state ──────────────────────────────────────────

    fn state_with_hosts(n: usize) -> MachinesState {
        let mut st = MachinesState::default();
        for i in 0..n {
            st.registry
                .add(entry(&format!("h{i}"), &format!("http://h{i}:7680")))
                .unwrap();
        }
        st
    }

    #[test]
    fn selected_row_zero_is_local() {
        let st = state_with_hosts(2);
        assert_eq!(st.selected_row(), MachineRowId::Local);
        assert_eq!(st.row_count(), 3);
    }

    #[test]
    fn selection_moves_and_clamps() {
        let mut st = state_with_hosts(1);
        assert!(!st.select_prev());
        assert!(st.select_next());
        assert_eq!(st.selected_row(), MachineRowId::Host("h0".into()));
        assert!(!st.select_next(), "selection clamps at the last row");
    }

    fn test_device(id: &str, ordinal: usize) -> mold_core::DeviceInfo {
        serde_json::from_value(serde_json::json!({
            "id": id,
            "backend": "cuda",
            "ordinal": ordinal,
            "device_kind": "full_gpu",
            "nvml_uuid": null,
            "physical_uuid": null,
            "mig_uuid": null,
            "mig_parent_uuid": null,
            "mig_profile": null,
            "name": format!("GPU {ordinal}"),
            "pci_bus_id": null,
            "compute_capability": null,
            "memory": {
                "total_bytes": null,
                "used_bytes": null,
                "mold_used_bytes": null,
                "other_used_bytes": null
            },
            "telemetry": {
                "utilization_percent": null,
                "temperature_c": null,
                "power_w": null
            },
            "desired_enabled": true,
            "admin_state": "enabled",
            "health": "healthy",
            "activity": "idle",
            "schedulable": true,
            "unschedulable_reason": null,
            "loaded_models": [],
            "active_work_id": null,
            "planned_work_ids": []
        }))
        .unwrap()
    }

    #[test]
    fn stable_device_selection_is_independent_and_clamped() {
        let mut st = state_with_hosts(1);
        st.select_next();
        st.apply_devices(
            "h0".into(),
            Some(mold_core::DeviceState {
                devices: vec![test_device("cuda:first", 0), test_device("cuda:second", 1)],
                plan_version: 1,
            }),
        );
        assert_eq!(st.selected_device().unwrap().id, "cuda:first");
        st.select_next_device();
        assert_eq!(st.selected_device().unwrap().id, "cuda:second");
        st.select_next_device();
        assert_eq!(st.device_selected, 0);
        st.select_prev_device();
        assert_eq!(st.device_selected, 1);
        st.select_prev_device();
        assert_eq!(st.selected_device().unwrap().id, "cuda:first");
    }

    #[test]
    fn device_selection_cycles_without_stealing_queue_selection() {
        let device = |ordinal: usize| mold_core::DeviceInfo {
            id: format!("cuda:{ordinal:032x}"),
            backend: mold_core::GpuBackend::Cuda,
            ordinal: Some(ordinal),
            device_kind: mold_core::DeviceKind::FullGpu,
            nvml_uuid: None,
            physical_uuid: None,
            mig_uuid: None,
            mig_parent_uuid: None,
            mig_profile: None,
            name: format!("GPU {ordinal}"),
            pci_bus_id: None,
            compute_capability: Some("8.6".into()),
            memory: mold_core::DeviceMemoryInfo {
                metal_memory: None,
                total_bytes: Some(24 * 1024_u64.pow(3)),
                used_bytes: Some(0),
                mold_used_bytes: None,
                other_used_bytes: None,
            },
            telemetry: mold_core::DeviceTelemetry {
                utilization_percent: Some(0),
                temperature_c: None,
                power_w: None,
            },
            desired_enabled: true,
            restart_required: false,
            admin_state: mold_core::DeviceAdminState::Enabled,
            health: mold_core::DeviceHealth::Healthy,
            activity: mold_core::DeviceActivity::Idle,
            schedulable: true,
            unschedulable_reason: None,
            loaded_models: Vec::new(),
            active_work_id: None,
            planned_work_ids: Vec::new(),
        };
        let mut st = MachinesState {
            queue_selected: 3,
            ..Default::default()
        };
        st.apply_devices(
            LOCAL_HOST_ID.to_string(),
            Some(mold_core::DeviceState {
                devices: vec![device(0), device(7)],
                plan_version: 4,
            }),
        );

        assert_eq!(st.selected_device().and_then(|gpu| gpu.ordinal), Some(0));
        st.select_next_device();
        assert_eq!(st.selected_device().and_then(|gpu| gpu.ordinal), Some(7));
        st.select_next_device();
        assert_eq!(st.selected_device().and_then(|gpu| gpu.ordinal), Some(0));
        assert_eq!(st.queue_selected, 3);

        let mut legacy = mold_core::ServerCapabilities::default();
        legacy.devices.restart_enable = true;
        st.apply_capabilities(LOCAL_HOST_ID.to_string(), Some(legacy));
        assert!(!st.can_mutate_selected_device());
        st.devices.get_mut(LOCAL_HOST_ID).unwrap().devices[0].desired_enabled = false;
        assert!(st.can_mutate_selected_device());
        assert!(st.selected_device_uses_restart_enable());

        let capabilities = st.capabilities.get_mut(LOCAL_HOST_ID).unwrap();
        capabilities.devices.lifecycle = true;
        capabilities.dispatch.v2_authoritative = false;
        assert!(st.can_mutate_selected_device());
        assert!(
            st.selected_device_uses_restart_enable(),
            "restart recovery is still the only supported action without authoritative dispatch"
        );

        st.devices.get_mut(LOCAL_HOST_ID).unwrap().devices[0].restart_required = true;
        assert!(
            !st.can_mutate_selected_device(),
            "a pending restart is not another actionable enable request"
        );
        assert!(st.selected_device_restart_required());
        assert!(!st.selected_device_uses_restart_enable());

        let mut v2 = mold_core::ServerCapabilities::default();
        v2.devices.lifecycle = true;
        v2.dispatch.v2_authoritative = true;
        st.apply_capabilities(LOCAL_HOST_ID.to_string(), Some(v2));
        let selected = &mut st.devices.get_mut(LOCAL_HOST_ID).unwrap().devices[0];
        selected.desired_enabled = true;
        selected.restart_required = false;
        assert!(st.can_mutate_selected_device());
        assert!(!st.selected_device_uses_restart_enable());
    }

    #[tokio::test]
    async fn device_mutation_emits_accepted_state_before_authoritative_poll() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let accepted = mold_core::DeviceInfo {
            id: "cuda:00000000000000000000000000000000".into(),
            backend: mold_core::GpuBackend::Cuda,
            ordinal: Some(0),
            device_kind: mold_core::DeviceKind::FullGpu,
            nvml_uuid: None,
            physical_uuid: None,
            mig_uuid: None,
            mig_parent_uuid: None,
            mig_profile: None,
            name: "GPU 0".into(),
            pci_bus_id: None,
            compute_capability: Some("8.6".into()),
            memory: mold_core::DeviceMemoryInfo {
                metal_memory: None,
                total_bytes: Some(24 * 1024_u64.pow(3)),
                used_bytes: Some(0),
                mold_used_bytes: None,
                other_used_bytes: None,
            },
            telemetry: mold_core::DeviceTelemetry {
                utilization_percent: Some(0),
                temperature_c: None,
                power_w: None,
            },
            desired_enabled: true,
            restart_required: true,
            admin_state: mold_core::DeviceAdminState::Disabled,
            health: mold_core::DeviceHealth::Unavailable,
            activity: mold_core::DeviceActivity::Idle,
            schedulable: false,
            unschedulable_reason: Some("restart required".into()),
            loaded_models: Vec::new(),
            active_work_id: None,
            planned_work_ids: Vec::new(),
        };
        let polled = mold_core::DeviceState {
            devices: vec![accepted.clone()],
            plan_version: 7,
        };
        let responses = vec![
            (
                "202 Accepted",
                serde_json::to_string(&accepted).expect("serialize accepted device"),
            ),
            (
                "200 OK",
                serde_json::to_string(&polled).expect("serialize device poll"),
            ),
        ];
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind test server");
        let address = listener.local_addr().expect("test server address");
        let server = tokio::spawn(async move {
            for (status, body) in responses {
                let (mut socket, _) = listener.accept().await.expect("accept request");
                let mut request = vec![0_u8; 8192];
                let _ = socket.read(&mut request).await.expect("read request");
                let response = format!(
                    "HTTP/1.1 {status}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                    body.len()
                );
                socket
                    .write_all(response.as_bytes())
                    .await
                    .expect("write response");
            }
        });

        let entry = HostEntry {
            id: "test-host".into(),
            url: format!("http://{address}"),
            name: Some("Test host".into()),
            instance_id: None,
            connected: true,
        };
        let (tx, mut rx) = mpsc::unbounded_channel();
        set_host_device_enabled(entry, accepted.id.clone(), true, tx).await;

        let first = rx.recv().await.expect("accepted response event");
        match first {
            BackgroundEvent::HostDeviceMutationApplied { host_id, device } => {
                assert_eq!(host_id, "test-host");
                assert!(device.restart_required);
                let mut machines = MachinesState::default();
                machines.apply_device_mutation(LOCAL_HOST_ID.to_string(), *device);
                assert!(machines.selected_device_restart_required());
                assert_eq!(
                    machines
                        .device_feedback
                        .get(LOCAL_HOST_ID)
                        .map(String::as_str),
                    Some("GPU 0 enabled on restart")
                );
                machines.apply_devices(LOCAL_HOST_ID.to_string(), Some(polled.clone()));
                assert!(
                    machines.selected_device_restart_required(),
                    "the authoritative poll preserves the accepted pending-restart presentation"
                );
            }
            _ => panic!("expected accepted mutation event"),
        }
        let second = rx.recv().await.expect("poll event");
        match second {
            BackgroundEvent::HostDevicesUpdate { host_id, devices } => {
                assert_eq!(host_id, "test-host");
                assert_eq!(devices.expect("polled devices"), polled);
            }
            _ => panic!("expected authoritative poll event"),
        }
        assert!(rx.try_recv().is_err());
        server.await.expect("test server task");
    }

    #[test]
    fn selection_change_clears_queue_snapshot() {
        let mut st = state_with_hosts(2);
        st.select_next();
        st.apply_queue("h0".into(), Some(mold_core::QueueListingWire::default()));
        assert!(st.queue.is_some());
        st.select_next();
        assert!(st.queue.is_none(), "stale queue must not leak across rows");
        assert!(
            !st.capabilities.contains_key("h1"),
            "the selected host must not inherit another host's capabilities"
        );
    }

    #[test]
    fn lifecycle_actions_require_the_selected_hosts_device_and_capability() {
        let mut st = state_with_hosts(2);
        st.select_next();
        st.apply_devices(
            "h0".into(),
            Some(mold_core::DeviceState {
                devices: vec![test_device("cuda:first", 0)],
                plan_version: 1,
            }),
        );
        let mut capabilities = mold_core::ServerCapabilities::default();
        st.apply_capabilities("h0".into(), Some(capabilities.clone()));
        assert!(!st.can_mutate_selected_device());

        capabilities.devices.lifecycle = true;
        capabilities.dispatch.v2_authoritative = true;
        st.apply_capabilities("h1".into(), Some(capabilities.clone()));
        assert!(
            !st.can_mutate_selected_device(),
            "a stale response for another host cannot expose controls"
        );
        st.apply_capabilities("h0".into(), Some(capabilities));
        assert!(st.can_mutate_selected_device());
    }

    #[test]
    fn device_inventory_supports_sixty_four_entries_and_lifecycle_shapes() {
        let mut st = state_with_hosts(1);
        st.select_next();
        let mut devices = (0..64)
            .map(|ordinal| test_device(&format!("cuda:{ordinal:032x}"), ordinal))
            .collect::<Vec<_>>();
        devices[1].admin_state = mold_core::DeviceAdminState::Disabled;
        devices[2].admin_state = mold_core::DeviceAdminState::Draining;
        devices[3].health = mold_core::DeviceHealth::Unavailable;
        devices[4].device_kind = mold_core::DeviceKind::Mig;
        devices[4].mig_profile = Some("1g.10gb".into());
        st.apply_devices(
            "h0".into(),
            Some(mold_core::DeviceState {
                devices,
                plan_version: 1,
            }),
        );
        assert_eq!(st.devices.get("h0").unwrap().devices.len(), 64);
        st.device_selected = 63;
        assert_eq!(st.selected_device().unwrap().ordinal, Some(63));
    }

    #[test]
    fn apply_queue_ignores_non_selected_host() {
        let mut st = state_with_hosts(2);
        st.select_next(); // h0 selected
        st.apply_queue("h1".into(), Some(mold_core::QueueListingWire::default()));
        assert!(st.queue.is_none());
    }

    #[test]
    fn target_for_selected_toggles_back_to_auto() {
        let mut st = state_with_hosts(1);
        assert_eq!(st.target_for_selected(&GenTarget::Auto), GenTarget::Local);
        assert_eq!(st.target_for_selected(&GenTarget::Local), GenTarget::Auto);
        st.select_next();
        assert_eq!(
            st.target_for_selected(&GenTarget::Auto),
            GenTarget::Host("h0".into())
        );
        assert_eq!(
            st.target_for_selected(&GenTarget::Host("h0".into())),
            GenTarget::Auto
        );
    }

    #[test]
    fn offline_status_keeps_row_and_self_heals() {
        let mut st = state_with_hosts(1);
        st.apply_status("h0".into(), None);
        assert!(matches!(
            st.statuses.get("h0").unwrap().health,
            HostHealth::Offline(_)
        ));
        // Self-heal on the next successful poll.
        st.apply_status("h0".into(), Some(status_with(Some("h0"), None)));
        assert_eq!(st.statuses.get("h0").unwrap().health, HostHealth::Ready);
    }

    #[test]
    fn poll_plan_machines_view_covers_all_hosts_and_selected_queue() {
        let mut st = state_with_hosts(2);
        st.select_next(); // h0
        let plan = st.poll_plan(true, &GenTarget::Auto);
        assert_eq!(plan.hosts.len(), 2);
        assert!(plan
            .hosts
            .iter()
            .any(|request| request.entry.id == "h0" && request.include_queue));
        assert!(plan
            .hosts
            .iter()
            .any(|request| request.entry.id == "h1" && !request.include_queue));
        // Unknown hosts are marked Connecting while the fetch runs.
        assert_eq!(
            st.statuses.get("h0").unwrap().health,
            HostHealth::Connecting
        );
        assert!(
            st.poll_plan(true, &GenTarget::Auto).hosts.is_empty(),
            "in-flight hosts remain single-flight"
        );
    }

    #[test]
    fn poll_plan_background_polls_only_the_target_host() {
        let mut st = state_with_hosts(2);
        let plan = st.poll_plan(false, &GenTarget::Host("h1".into()));
        assert_eq!(
            plan.hosts
                .iter()
                .map(|request| request.entry.id.as_str())
                .collect::<Vec<_>>(),
            vec!["h1"]
        );
        assert!(plan.hosts.iter().all(|request| !request.include_queue));

        let plan = st.poll_plan(false, &GenTarget::Auto);
        assert!(plan.hosts.is_empty());
    }

    #[test]
    fn stalled_host_poll_is_single_flight_and_completion_relative() {
        let mut st = state_with_hosts(1);
        st.select_next();

        let first = st.poll_plan(true, &GenTarget::Auto);
        assert_eq!(first.hosts.len(), 1);
        assert!(first.hosts[0].include_queue);
        for _ in 0..32 {
            assert!(
                st.poll_plan(true, &GenTarget::Auto).hosts.is_empty(),
                "a stalled host must never acquire another polling lease"
            );
        }

        assert!(!st.finish_poll("h0"));
        assert!(
            st.poll_plan(true, &GenTarget::Auto).hosts.is_empty(),
            "the next interval starts when the prior poll finishes"
        );
        st.force_poll();
        assert_eq!(st.poll_plan(true, &GenTarget::Auto).hosts.len(), 1);
    }

    fn queue_job(id: &str, state: &str) -> mold_core::QueueJobEntryWire {
        mold_core::QueueJobEntryWire {
            id: id.into(),
            model: "flux-dev:q8".into(),
            state: state.into(),
            started_at_unix_ms: 0,
            position: 0,
            gpu: None,
            target_gpu: None,
            seed_pinned: None,
            metadata: None,
            durable: Some(true),
            held_reason: None,
            ..Default::default()
        }
    }

    fn queue_page(
        entries: Vec<mold_core::QueueJobEntryWire>,
        cursor: Option<&str>,
    ) -> mold_core::QueueListingWire {
        let returned = entries.len();
        mold_core::QueueListingWire {
            entries,
            page: Some(mold_core::QueuePage {
                limit: 2,
                offset: 0,
                returned,
                next_cursor: cursor.map(str::to_string),
            }),
            ..Default::default()
        }
    }

    #[test]
    fn continuation_and_running_rows_survive_head_refresh_without_duplicate_request() {
        let mut st = state_with_hosts(1);
        st.select_next();
        st.apply_queue(
            "h0".into(),
            Some(queue_page(
                vec![
                    queue_job("running", "running"),
                    queue_job("selected", "queued"),
                ],
                Some("cursor-1"),
            )),
        );
        st.queue_selected = 1;
        assert_eq!(st.begin_queue_continuation().unwrap().2, "cursor-1");
        assert!(
            st.begin_queue_continuation().is_none(),
            "the same continuation cannot be requested twice"
        );

        st.apply_queue(
            "h0".into(),
            Some(queue_page(
                vec![queue_job("new-head", "queued")],
                Some("new-cursor"),
            )),
        );
        let listing = &st.queue.as_ref().unwrap().1;
        assert_eq!(
            listing
                .entries
                .iter()
                .map(|job| job.id.as_str())
                .collect::<Vec<_>>(),
            vec!["new-head", "running", "selected"],
            "head refresh keeps the running row and the previously loaded window"
        );
        assert_eq!(listing.entries[st.queue_selected].id, "selected");
        assert!(st.queue_loading_more);

        st.apply_queue_continuation(
            "h0".into(),
            "cursor-1",
            Some(queue_page(
                vec![queue_job("continued", "queued")],
                Some("cursor-2"),
            )),
        );
        assert_eq!(
            st.queue
                .as_ref()
                .unwrap()
                .1
                .page
                .as_ref()
                .unwrap()
                .next_cursor
                .as_deref(),
            Some("cursor-2")
        );
        assert!(!st.queue_loading_more);

        st.apply_queue(
            "h0".into(),
            Some(queue_page(
                vec![queue_job("new-head", "queued")],
                Some("head-again"),
            )),
        );
        let listing = &st.queue.as_ref().unwrap().1;
        assert!(listing.entries.iter().any(|job| job.id == "continued"));
        assert_eq!(listing.entries[st.queue_selected].id, "selected");
        assert_eq!(
            listing.page.as_ref().unwrap().next_cursor.as_deref(),
            Some("cursor-2")
        );
        assert_eq!(st.begin_queue_continuation().unwrap().2, "cursor-2");
    }

    #[test]
    fn routine_head_refresh_keeps_only_missing_in_flight_rows_before_load_more() {
        let mut st = state_with_hosts(1);
        st.select_next();
        st.apply_queue(
            "h0".into(),
            Some(queue_page(
                vec![
                    queue_job("running", "running"),
                    queue_job("old-queued", "queued"),
                ],
                Some("cursor-1"),
            )),
        );
        st.apply_queue(
            "h0".into(),
            Some(queue_page(
                vec![queue_job("head", "queued")],
                Some("cursor-2"),
            )),
        );

        assert_eq!(
            st.queue
                .as_ref()
                .unwrap()
                .1
                .entries
                .iter()
                .map(|job| job.id.as_str())
                .collect::<Vec<_>>(),
            vec!["head", "running"]
        );
    }

    #[test]
    #[serial(mold_env)]
    fn forget_clamps_selection_and_clears_queue() {
        with_isolated_env(|_home| {
            let mut st = state_with_hosts(1);
            st.select_next();
            st.apply_queue("h0".into(), Some(mold_core::QueueListingWire::default()));
            st.forget("h0");
            assert_eq!(st.registry.hosts.len(), 0);
            assert_eq!(st.selected, 0);
            assert!(st.queue.is_none());
        });
    }

    #[test]
    #[serial(mold_env)]
    fn complete_connect_registers_backfills_name_and_selects_row() {
        with_isolated_env(|_home| {
            let mut st = MachinesState::default();
            let id = st
                .complete_connect(
                    "http://hal9000:7680",
                    Some("sekrit"),
                    status_with(Some("hal9000"), Some("uuid-a")),
                )
                .unwrap();
            assert_eq!(id, "hal9000-7680");
            let host = st.registry.get(&id).unwrap();
            assert_eq!(host.name.as_deref(), Some("hal9000"));
            assert_eq!(host.instance_id.as_deref(), Some("uuid-a"));
            assert_eq!(api_key_for(&id).as_deref(), Some("sekrit"));
            assert_eq!(st.selected_row(), MachineRowId::Host(id.clone()));
            assert_eq!(st.statuses.get(&id).unwrap().health, HostHealth::Ready);

            // Registry persisted.
            assert_eq!(HostRegistry::load().hosts.len(), 1);
        });
    }

    #[test]
    #[serial(mold_env)]
    fn connect_rejects_duplicate_instance_id() {
        with_isolated_env(|_home| {
            let mut st = MachinesState::default();
            st.complete_connect(
                "http://hal9000:7680",
                None,
                status_with(Some("hal9000"), Some("uuid-a")),
            )
            .unwrap();
            let err = st
                .complete_connect(
                    "http://192.168.1.5:7680",
                    None,
                    status_with(Some("hal9000"), Some("uuid-a")),
                )
                .unwrap_err();
            assert_eq!(
                err,
                AddHostError::AlreadyKnown {
                    name: "hal9000".into()
                }
            );
            assert_eq!(st.registry.hosts.len(), 1);
        });
    }

    // ── connect state machine ───────────────────────────────────

    #[test]
    fn connect_normalizes_url_on_enter() {
        let mut form = ConnectForm::default();
        for c in "hal9000".chars() {
            connect_advance(&mut form, ConnectInput::Char(c));
        }
        let fx = connect_advance(&mut form, ConnectInput::Enter);
        assert_eq!(fx, ConnectEffect::None);
        assert_eq!(form.step, ConnectStep::ApiKey);
        assert_eq!(form.url, "http://hal9000:7680");
    }

    #[test]
    fn connect_empty_url_enter_is_a_noop() {
        let mut form = ConnectForm::default();
        assert_eq!(
            connect_advance(&mut form, ConnectInput::Enter),
            ConnectEffect::None
        );
        assert_eq!(form.step, ConnectStep::Url);
    }

    #[test]
    fn connect_optional_key_skipped_with_enter() {
        let mut form = ConnectForm {
            step: ConnectStep::ApiKey,
            url: "http://hal9000:7680".into(),
            ..Default::default()
        };
        let fx = connect_advance(&mut form, ConnectInput::Enter);
        assert_eq!(fx, ConnectEffect::StartTest);
        assert_eq!(form.step, ConnectStep::Testing);
        assert!(form.api_key.is_empty());
    }

    #[test]
    fn connect_success_backfills_name_and_saves() {
        let mut form = ConnectForm {
            step: ConnectStep::Testing,
            url: "http://hal9000:7680".into(),
            ..Default::default()
        };
        let fx = connect_advance(
            &mut form,
            ConnectInput::TestOk {
                hostname: Some("hal9000"),
            },
        );
        assert_eq!(fx, ConnectEffect::Save);
        assert_eq!(form.name, "hal9000");

        // No hostname → fall back to host:port.
        let mut anon = ConnectForm {
            step: ConnectStep::Testing,
            url: "http://10.0.0.2:7680".into(),
            ..Default::default()
        };
        connect_advance(&mut anon, ConnectInput::TestOk { hostname: None });
        assert_eq!(anon.name, "10.0.0.2:7680");
    }

    #[test]
    fn connect_failure_offers_retry_and_edit() {
        let mut form = ConnectForm {
            step: ConnectStep::Testing,
            url: "http://down:7680".into(),
            ..Default::default()
        };
        connect_advance(&mut form, ConnectInput::TestErr("connection refused"));
        assert_eq!(form.step, ConnectStep::Failed);
        assert_eq!(form.error.as_deref(), Some("connection refused"));

        // Enter retries the test.
        let fx = connect_advance(&mut form, ConnectInput::Enter);
        assert_eq!(fx, ConnectEffect::StartTest);
        assert_eq!(form.step, ConnectStep::Testing);

        // …and after another failure, `e` goes back to editing the URL.
        connect_advance(&mut form, ConnectInput::TestErr("still down"));
        connect_advance(&mut form, ConnectInput::Char('e'));
        assert_eq!(form.step, ConnectStep::Url);
        assert!(form.error.is_none());
    }

    #[test]
    fn connect_esc_closes_from_url_and_testing() {
        let mut form = ConnectForm::default();
        assert_eq!(
            connect_advance(&mut form, ConnectInput::Esc),
            ConnectEffect::Close
        );
        let mut testing = ConnectForm {
            step: ConnectStep::Testing,
            ..Default::default()
        };
        assert_eq!(
            connect_advance(&mut testing, ConnectInput::Esc),
            ConnectEffect::Close
        );
        // Esc from the ApiKey step steps back to the URL instead.
        let mut keyed = ConnectForm {
            step: ConnectStep::ApiKey,
            ..Default::default()
        };
        assert_eq!(
            connect_advance(&mut keyed, ConnectInput::Esc),
            ConnectEffect::None
        );
        assert_eq!(keyed.step, ConnectStep::Url);
    }

    // ── client construction ─────────────────────────────────────

    #[test]
    fn client_for_builds_for_both_key_variants() {
        // Constructor smoke test — the api-key header behavior itself is
        // covered by mold-core's wiremock tests.
        let _ = client_for("http://hal9000:7680", None);
        let _ = client_for("http://hal9000:7680", Some("key"));
        let _ = client_for("http://hal9000:7680", Some(""));
    }
}
