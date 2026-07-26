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
use std::collections::HashMap;
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

    /// Every machine this instance knows about, local first. The local
    /// entry is synthetic (empty URL — it is not an HTTP target); the
    /// future multi-host Library merge iterates this list directly.
    pub fn all(&self) -> Vec<HostEntry> {
        let mut out = Vec::with_capacity(self.hosts.len() + 1);
        out.push(HostEntry {
            id: LOCAL_HOST_ID.to_string(),
            url: String::new(),
            name: Some(local_display_name().to_string()),
            instance_id: None,
        });
        out.extend(self.hosts.iter().cloned());
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

/// Read the saved API key for a host, if any.
pub(crate) fn api_key_for(id: &str) -> Option<String> {
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
const POLL_INTERVAL: std::time::Duration = std::time::Duration::from_secs(4);

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
    /// Stable-ID device inventory for the selected remote host.
    pub devices: Option<(String, mold_core::DeviceState)>,
    /// Job selection inside the detail pane's queue lanes.
    pub queue_selected: usize,
    /// Stable device selection inside the detail pane.
    pub device_selected: usize,
    /// Last host-poll instant (None = poll immediately).
    pub last_poll: Option<std::time::Instant>,
}

/// What [`MachinesState::poll_plan`] decided to fetch this tick.
#[derive(Debug, Default)]
pub(crate) struct PollPlan {
    pub status_hosts: Vec<HostEntry>,
    pub queue_host: Option<HostEntry>,
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
        self.devices = None;
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

    /// The queued (cancelable) job under the queue selection, if any.
    pub fn selected_queued_job(&self) -> Option<(String, mold_core::QueueJobEntryWire)> {
        let (host_id, listing) = self.queue.as_ref()?;
        let job = listing.entries.get(self.queue_selected)?;
        if job.state == "queued" {
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
            MachineRowId::Host(id) => GenTarget::Host(id),
        };
        if picked == *current {
            GenTarget::Auto
        } else {
            picked
        }
    }

    /// Whether the host-poll interval has elapsed.
    pub fn poll_due(&self) -> bool {
        self.last_poll.is_none_or(|t| t.elapsed() >= POLL_INTERVAL)
    }

    /// Force the next [`Self::poll_due`] to fire immediately.
    pub fn force_poll(&mut self) {
        self.last_poll = None;
    }

    /// Decide which hosts to fetch this tick: every registered host (and
    /// the selected host's queue) while Machines is active, only the
    /// generation-target host otherwise. Marks not-yet-seen hosts as
    /// Connecting so their rows render honestly while the fetch runs.
    pub fn poll_plan(&mut self, machines_active: bool, target: &GenTarget) -> PollPlan {
        self.last_poll = Some(std::time::Instant::now());
        let mut plan = PollPlan::default();
        if machines_active {
            plan.status_hosts = self.registry.hosts.clone();
            plan.queue_host = self.selected_host().cloned();
        } else if let GenTarget::Host(id) = target {
            if let Some(entry) = self.registry.get(id) {
                plan.status_hosts.push(entry.clone());
            }
        }
        for entry in &plan.status_hosts {
            self.statuses.entry(entry.id.clone()).or_insert(HostStatus {
                health: HostHealth::Connecting,
                status: None,
            });
        }
        plan
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
        match queue {
            Some(listing) => {
                if self.queue_selected >= listing.entries.len() {
                    self.queue_selected = listing.entries.len().saturating_sub(1);
                }
                self.queue = Some((host_id, listing));
            }
            None => self.queue = None,
        }
    }

    pub fn apply_devices(&mut self, host_id: String, devices: Option<mold_core::DeviceState>) {
        let selected_id = match self.selected_row() {
            MachineRowId::Host(id) => id,
            MachineRowId::Local => return,
        };
        if selected_id == host_id {
            if let Some(state) = &devices {
                if self.device_selected >= state.devices.len() {
                    self.device_selected = state.devices.len().saturating_sub(1);
                }
            }
            self.devices = devices.map(|state| (host_id, state));
        }
    }

    pub fn selected_device(&self) -> Option<&mold_core::DeviceInfo> {
        let MachineRowId::Host(selected_id) = self.selected_row() else {
            return None;
        };
        let (host_id, state) = self.devices.as_ref()?;
        (host_id == &selected_id)
            .then(|| state.devices.get(self.device_selected))
            .flatten()
    }

    pub fn device_select_prev(&mut self) {
        self.device_selected = self.device_selected.saturating_sub(1);
    }

    pub fn device_select_next(&mut self) {
        let count = self
            .devices
            .as_ref()
            .map_or(0, |(_, state)| state.devices.len());
        if self.device_selected + 1 < count {
            self.device_selected += 1;
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
        if self
            .devices
            .as_ref()
            .is_some_and(|(host_id, _)| host_id == id)
        {
            self.devices = None;
        }
        if self.selected >= self.row_count() {
            self.selected = self.row_count() - 1;
        }
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

/// Fetch `/api/status` for one host and report back. `None` status marks
/// the row Offline; it stays listed and self-heals on the next poll.
pub(crate) async fn fetch_host_status(
    entry: HostEntry,
    tx: mpsc::UnboundedSender<BackgroundEvent>,
) {
    let client = client_for_host(&entry);
    let status = client.server_status().await.ok().map(Box::new);
    let devices = client.devices().await.ok();
    let capabilities = client.server_capabilities().await.ok();
    let _ = tx.send(BackgroundEvent::HostStatusUpdate {
        host_id: entry.id.clone(),
        status,
    });
    let _ = tx.send(BackgroundEvent::HostDevicesUpdate {
        host_id: entry.id.clone(),
        devices,
    });
    let _ = tx.send(BackgroundEvent::HostCapabilitiesUpdate {
        host_id: entry.id,
        capabilities,
    });
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
    let queue = client.list_queue().await.ok();
    let devices = client.devices().await.ok();
    let host_id = entry.id;
    let _ = tx.send(BackgroundEvent::HostQueueUpdate {
        host_id: host_id.clone(),
        queue,
    });
    let _ = tx.send(BackgroundEvent::HostDevicesUpdate { host_id, devices });
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

/// Update one stable device and publish the server's authoritative state.
pub(crate) async fn set_host_device_enabled(
    entry: HostEntry,
    device_id: String,
    enabled: bool,
    tx: mpsc::UnboundedSender<BackgroundEvent>,
) {
    let client = client_for_host(&entry);
    match client.set_device_enabled(&device_id, enabled).await {
        Ok(devices) => {
            let _ = tx.send(BackgroundEvent::HostDevicesUpdate {
                host_id: entry.id,
                devices: Some(devices),
            });
        }
        Err(error) => {
            let _ = tx.send(BackgroundEvent::Error(format!(
                "Device update failed for {device_id}: {error:#}"
            )));
        }
    }
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
        })
        .unwrap();

        // Same instance id via a different URL — one box, two routes.
        let err = reg
            .add(HostEntry {
                id: "192-168-1-5-7680".into(),
                url: "http://192.168.1.5:7680".into(),
                name: None,
                instance_id: Some("uuid-a".into()),
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
        st.device_select_next();
        assert_eq!(st.selected_device().unwrap().id, "cuda:second");
        st.device_select_next();
        assert_eq!(st.device_selected, 1);
        st.device_select_prev();
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
        assert_eq!(plan.status_hosts.len(), 2);
        assert_eq!(plan.queue_host.as_ref().map(|h| h.id.as_str()), Some("h0"));
        // Unknown hosts are marked Connecting while the fetch runs.
        assert_eq!(
            st.statuses.get("h0").unwrap().health,
            HostHealth::Connecting
        );
        assert!(!st.poll_due(), "poll_plan stamps last_poll");
    }

    #[test]
    fn poll_plan_background_polls_only_the_target_host() {
        let mut st = state_with_hosts(2);
        let plan = st.poll_plan(false, &GenTarget::Host("h1".into()));
        assert_eq!(
            plan.status_hosts
                .iter()
                .map(|h| h.id.as_str())
                .collect::<Vec<_>>(),
            vec!["h1"]
        );
        assert!(plan.queue_host.is_none());

        let plan = st.poll_plan(false, &GenTarget::Auto);
        assert!(plan.status_hosts.is_empty());
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
