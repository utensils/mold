use mold_core::{GenerateResponse, ModelInfoExtended, ServerStatus, SseProgressEvent};

/// Structured embed data that the handler converts to serenity `CreateEmbed`.
#[derive(Debug, Clone)]
pub struct EmbedData {
    pub title: String,
    pub description: String,
    pub fields: Vec<(String, String, bool)>,
    pub color: u32,
}

// Colors for embed theming
const COLOR_SUCCESS: u32 = 0x57F287; // green
const COLOR_INFO: u32 = 0x5865F2; // blurple
const COLOR_WARNING: u32 = 0xFEE75C; // yellow
const COLOR_ERROR: u32 = 0xED4245; // red
const DISCORD_EMBED_TITLE_MAX: usize = 256;
const DISCORD_EMBED_DESCRIPTION_MAX: usize = 4096;
const DISCORD_EMBED_FIELD_NAME_MAX: usize = 256;
const DISCORD_EMBED_FIELD_VALUE_MAX: usize = 1024;
const DISCORD_EMBED_FIELDS_MAX: usize = 25;
const DISCORD_EMBED_TOTAL_MAX: usize = 6000;

/// Format an SSE progress event into a human-readable status line.
pub fn format_progress(event: &SseProgressEvent) -> String {
    match event {
        SseProgressEvent::DependencyWait { dependency, reason } => {
            format!("Waiting for {dependency}: {reason}")
        }
        SseProgressEvent::Queued { position, .. } => {
            if *position == 0 {
                "Starting generation...".to_string()
            } else {
                format!("Queued — position #{position}")
            }
        }
        SseProgressEvent::StageStart { name } => {
            format!("Loading: {name}...")
        }
        SseProgressEvent::StageDone { name, elapsed_ms } => {
            format!("{name} ({elapsed_ms}ms)")
        }
        SseProgressEvent::StageProgress {
            name,
            current,
            total,
        } => format!("{name} {current}/{total}"),
        SseProgressEvent::Info { message } => message.clone(),
        SseProgressEvent::CacheHit { resource } => {
            format!("Cache hit: {resource}")
        }
        // Latent previews are for canvas clients; Discord shows the step bar.
        SseProgressEvent::Preview { step, total, .. } => {
            format!("Denoising {step}/{total}")
        }
        SseProgressEvent::DenoiseStep {
            step,
            total,
            elapsed_ms,
        } => {
            let bar = progress_bar(*step, *total);
            let ms_per_step = if *step > 0 {
                *elapsed_ms as f64 / *step as f64
            } else {
                0.0
            };
            format!("Denoising {step}/{total} {bar} ({ms_per_step:.0}ms/step)")
        }
        SseProgressEvent::DownloadProgress {
            filename,
            file_index,
            total_files,
            bytes_downloaded,
            bytes_total,
            ..
        } => {
            let pct = if *bytes_total > 0 {
                (*bytes_downloaded as f64 / *bytes_total as f64 * 100.0) as u64
            } else {
                0
            };
            format!(
                "Downloading {filename} [{}/{total_files}] {pct}%",
                file_index + 1
            )
        }
        SseProgressEvent::DownloadDone {
            filename,
            file_index,
            total_files,
            ..
        } => {
            format!("Downloaded {filename} [{}/{total_files}]", file_index + 1)
        }
        SseProgressEvent::PullComplete { model } => {
            format!("Model {model} downloaded")
        }
        SseProgressEvent::WeightLoad {
            bytes_loaded,
            bytes_total,
            component,
        } => {
            let pct = if *bytes_total > 0 {
                (*bytes_loaded as f64 / *bytes_total as f64 * 100.0) as u64
            } else {
                0
            };
            format!("Loading {component} ({pct}%)")
        }
    }
}

/// Build a text progress bar: `[=========>          ]`
fn progress_bar(current: usize, total: usize) -> String {
    let width = 20;
    let filled = (current * width).checked_div(total).unwrap_or(0);
    let fill_len = if filled > 0 && filled < width {
        filled - 1
    } else {
        filled
    };
    let has_arrow = filled > 0 && filled < width;
    let space_len = width - fill_len - if has_arrow { 1 } else { 0 };
    format!(
        "[{}{}{}]",
        "=".repeat(fill_len),
        if has_arrow { ">" } else { "" },
        " ".repeat(space_len)
    )
}

/// One-line identity note for the result embed, built from the request the
/// bot actually sent — `GenerateResponse` carries no metadata, and the bot is
/// the only place that knows which photo went out.
///
/// `None` for every ordinary render, so nothing changes for a request that
/// carried no face reference. The defaults are `mold_core::identity`'s, which
/// is what the server applied for an omitted knob.
pub fn identity_note(req: &mold_core::GenerateRequest) -> Option<String> {
    req.id_image.as_ref()?;
    let name = req
        .id_image_name
        .as_deref()
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .unwrap_or("attached photo");
    Some(format!(
        "{name} \u{00b7} strength {:.2} \u{00b7} from step {}",
        mold_core::identity::effective_id_weight(req),
        mold_core::identity::effective_id_start_step(req)
    ))
}

/// Format a completed generation result into embed data. Video responses get
/// a distinct title and include frame/fps metadata alongside the usual fields.
/// `identity` is the [`identity_note`] line for a print that carried a face
/// reference, and `None` for every ordinary render.
pub fn format_generation_result(
    resp: &GenerateResponse,
    prompt: &str,
    identity: Option<&str>,
) -> EmbedData {
    let time_secs = resp.generation_time_ms as f64 / 1000.0;

    let truncated_prompt = if prompt.chars().count() > 256 {
        let truncated: String = prompt.chars().take(253).collect();
        format!("{truncated}...")
    } else {
        prompt.to_string()
    };

    let (title, mut fields) = if let Some(mesh) = resp.mesh.as_ref() {
        // Probed before video for the reason `plan_delivery` gives: a
        // mesh response has no video and no images, so the image arm below
        // would caption it with an empty size.
        let extent = |axis: usize| (mesh.bounds_max[axis] - mesh.bounds_min[axis]).abs();
        let fields = vec![
            ("Model".to_string(), resp.model.clone(), true),
            (
                "Triangles".to_string(),
                mold_core::format::group_thousands(mesh.face_count),
                true,
            ),
            (
                "Vertices".to_string(),
                mold_core::format::group_thousands(mesh.vertex_count),
                true,
            ),
            (
                "Bounds".to_string(),
                format!("{:.2} × {:.2} × {:.2}", extent(0), extent(1), extent(2)),
                true,
            ),
            ("Time".to_string(), format!("{time_secs:.1}s"), true),
            (
                "Format".to_string(),
                mesh.format.extension().to_ascii_uppercase(),
                true,
            ),
            ("Seed".to_string(), resp.seed_used.to_string(), false),
        ];
        ("Mesh Generated".to_string(), fields)
    } else if let Some(video) = resp.video.as_ref() {
        let format_label = video.format.extension().to_ascii_uppercase();
        let mut fields = vec![
            ("Model".to_string(), resp.model.clone(), true),
            (
                "Size".to_string(),
                format!("{}x{}", video.width, video.height),
                true,
            ),
            ("Time".to_string(), format!("{time_secs:.1}s"), true),
            (
                "Frames".to_string(),
                format!("{} @ {} fps", video.frames, video.fps),
                true,
            ),
            ("Format".to_string(), format_label, true),
        ];
        if video.has_audio {
            fields.push(("Audio".to_string(), "yes".to_string(), true));
        }
        fields.push(("Seed".to_string(), resp.seed_used.to_string(), false));
        ("Video Generated".to_string(), fields)
    } else {
        let dims = resp
            .images
            .first()
            .map(|img| format!("{}x{}", img.width, img.height))
            .unwrap_or_default();
        let fields = vec![
            ("Model".to_string(), resp.model.clone(), true),
            ("Size".to_string(), dims, true),
            ("Time".to_string(), format!("{time_secs:.1}s"), true),
            ("Seed".to_string(), resp.seed_used.to_string(), false),
        ];
        ("Image Generated".to_string(), fields)
    };

    // Preserve insertion order so tests/UI stay stable.
    fields.retain(|(_, v, _)| !v.is_empty());
    // Identity provenance rides on its own full-width row: the photo name is
    // long, and the conditioning is the point of the print.
    if let Some(identity) = identity.map(str::trim).filter(|note| !note.is_empty()) {
        fields.push(("Identity".to_string(), identity.to_string(), false));
    }
    if let Some(gpu) = resp.gpu {
        fields.push(("Device".to_string(), format!("GPU {gpu}"), true));
    }

    EmbedData {
        title,
        description: truncated_prompt,
        fields,
        color: COLOR_SUCCESS,
    }
}

/// Format the model list into embed data, grouped by family.
/// Downloaded/loaded models are shown in bold; undownloaded models are shown in
/// plain text. All models in every family are listed individually.
/// Alpha families are tagged with `(alpha)` in the section header.
pub fn format_model_list(models: &[ModelInfoExtended]) -> EmbedData {
    if models.is_empty() {
        return EmbedData {
            title: "Available Models".to_string(),
            description: "No models configured on the server.".to_string(),
            fields: vec![],
            color: COLOR_WARNING,
        };
    }

    // Group models by family, preserving insertion order
    let mut families: Vec<String> = Vec::new();
    let mut groups: std::collections::HashMap<String, Vec<&ModelInfoExtended>> =
        std::collections::HashMap::new();
    for m in models {
        let family = m.info.family.clone();
        groups.entry(family.clone()).or_default().push(m);
        if !families.contains(&family) {
            families.push(family);
        }
    }

    let mut sections = Vec::new();
    for family in &families {
        let members = &groups[family];
        let is_alpha = members
            .iter()
            .any(|m| m.defaults.description.starts_with("[alpha]"));

        let header = if is_alpha {
            format!("**{}** (alpha)", family.to_uppercase())
        } else {
            format!("**{}**", family.to_uppercase())
        };

        let mut lines = vec![header];
        for m in members {
            if m.info.is_loaded {
                lines.push(format!(
                    "**{}** — `{:.1}GB` [loaded]",
                    m.info.name, m.info.size_gb
                ));
            } else if m.downloaded {
                lines.push(format!(
                    "**{}** — `{:.1}GB` [ready]",
                    m.info.name, m.info.size_gb
                ));
            } else {
                lines.push(format!("{} — `{:.1}GB`", m.info.name, m.info.size_gb));
            }
        }
        sections.push(lines.join("\n"));
    }

    let description = sections.join("\n\n");
    // Truncate if too long for Discord (4096 char limit)
    let description = if description.chars().count() > 4000 {
        let truncated: String = description.chars().take(3997).collect();
        format!("{truncated}...")
    } else {
        description
    };

    EmbedData {
        title: "Available Models".to_string(),
        description,
        fields: vec![],
        color: COLOR_INFO,
    }
}

/// Format server status into embed data.
pub fn format_server_status(status: &ServerStatus) -> EmbedData {
    format_server_status_with_devices(status, None, None)
}

pub fn format_server_status_with_devices(
    status: &ServerStatus,
    devices: Option<&mold_core::DeviceState>,
    queue: Option<&mold_core::QueueListingWire>,
) -> EmbedData {
    format_server_status_pages(status, devices, queue)
        .into_iter()
        .next()
        .expect("status formatting always emits at least one page")
}

/// Discord caps every field and every embed independently. A 64-device host
/// cannot fit in one legal embed, so status is deterministically paginated as
/// one embed per follow-up message.
pub fn format_server_status_pages(
    status: &ServerStatus,
    devices: Option<&mold_core::DeviceState>,
    queue: Option<&mold_core::QueueListingWire>,
) -> Vec<EmbedData> {
    let uptime = format_duration_secs(status.uptime_secs);
    let version = match (&status.git_sha, &status.build_date) {
        (Some(sha), Some(date)) => format!("{} ({sha}, {date})", status.version),
        (Some(sha), None) => format!("{} ({sha})", status.version),
        _ => status.version.clone(),
    };

    let loaded = if status.models_loaded.is_empty() {
        "None".to_string()
    } else {
        status.models_loaded.join(", ")
    };

    let busy_status = if status.busy { "Generating..." } else { "Idle" };

    let mut fields = vec![
        ("Version".to_string(), version, true),
        ("Status".to_string(), busy_status.to_string(), true),
        ("Uptime".to_string(), uptime, true),
        ("Models Loaded".to_string(), loaded, false),
    ];

    if let Some(devices) = devices.filter(|state| !state.devices.is_empty()) {
        let lines = devices
            .devices
            .iter()
            .map(|device| {
                let ordinal = device
                    .ordinal
                    .map_or_else(|| "—".into(), |value| value.to_string());
                let memory = match (device.memory.used_bytes, device.memory.total_bytes) {
                    (Some(used), Some(total)) => {
                        format!("{}/{}MB", used / 1024_u64.pow(2), total / 1024_u64.pow(2))
                    }
                    (_, Some(total)) => {
                        format!("{}MB total, usage unknown", total / 1024_u64.pow(2))
                    }
                    _ => "VRAM unknown".to_string(),
                };
                let state = if device.admin_state == mold_core::DeviceAdminState::Draining {
                    "finishing current work".to_string()
                } else if device.health != mold_core::DeviceHealth::Healthy {
                    device.health.as_str().to_string()
                } else {
                    device.admin_state.as_str().to_string()
                };
                let kind = if device.device_kind == mold_core::DeviceKind::Mig {
                    format!(
                        "MIG {}",
                        device.mig_profile.as_deref().unwrap_or("profile unknown")
                    )
                } else {
                    device.backend.as_str().to_ascii_uppercase()
                };
                truncate_chars(
                    &format!(
                        "GPU {ordinal} · {} · {kind} · `{}` · {state} · {memory}",
                        device.name, device.id
                    ),
                    360,
                )
            })
            .collect::<Vec<_>>();
        for (index, chunk) in chunk_lines(&lines, DISCORD_EMBED_FIELD_VALUE_MAX)
            .into_iter()
            .enumerate()
        {
            fields.push((format!("Devices {}", index + 1), chunk, false));
        }
    } else if let Some(gpus) = status.gpus.as_ref().filter(|gpus| !gpus.is_empty()) {
        let mut groups: Vec<(&str, usize, u64, u64)> = Vec::new();
        for gpu in gpus {
            if let Some((_, count, used, total)) =
                groups.iter_mut().find(|(name, _, _, _)| *name == gpu.name)
            {
                *count += 1;
                *used += gpu.vram_used_bytes;
                *total += gpu.vram_total_bytes;
            } else {
                groups.push((&gpu.name, 1, gpu.vram_used_bytes, gpu.vram_total_bytes));
            }
        }
        let summary = groups
            .into_iter()
            .map(|(name, count, used, total)| {
                let fleet = if count > 1 {
                    format!("{count}× {name}")
                } else {
                    name.to_string()
                };
                format!(
                    "{fleet} ({}MB / {}MB VRAM)",
                    used / 1024_u64.pow(2),
                    total / 1024_u64.pow(2)
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        fields.push(("GPUs".to_string(), summary, false));
    } else if let Some(gpu) = &status.gpu_info {
        fields.push((
            "GPU".to_string(),
            format!(
                "{} ({}MB / {}MB VRAM)",
                gpu.name, gpu.vram_used_mb, gpu.vram_total_mb
            ),
            false,
        ));
    }
    if let Some(queue) = queue {
        let blocked = queue
            .plan
            .as_ref()
            .map(|plan| {
                plan.work_items
                    .iter()
                    .filter(|work| {
                        work.blocked_reason.as_ref().is_some_and(|reason| {
                            !mold_core::queue_wait::is_benign_queue_reason(Some(reason))
                        })
                    })
                    .count()
            })
            .unwrap_or(0);
        let planned = queue.plan.as_ref().map_or(0, |plan| plan.work_items.len());
        let queue_scope = match (status.queue_depth, queue.page.as_ref()) {
            (Some(depth), Some(_)) => format!(
                "{depth} total backlog · {} hydrated page rows",
                queue.entries.len()
            ),
            (Some(depth), None) => format!(
                "{depth} server-reported queue load · {} response rows",
                queue.entries.len()
            ),
            (None, Some(_)) => format!("{} hydrated page rows", queue.entries.len()),
            (None, None) => format!("{} response rows", queue.entries.len()),
        };
        let runtime = status
            .queue_capacity
            .map(|capacity| format!(" · runtime capacity {capacity}"))
            .unwrap_or_default();
        fields.push((
            "Queue".to_string(),
            format!("{queue_scope}{runtime} · {planned} planned · {blocked} blocked"),
            false,
        ));
        if let Some(plan) = queue.plan.as_ref() {
            let lines = plan
                .work_items
                .iter()
                .map(|work| {
                    let order = work
                        .lane_order
                        .map(|order| order.to_string())
                        .unwrap_or_else(|| "—".to_string());
                    let lane = match work.planned_lane_kind.as_ref() {
                        Some(mold_core::QueuePlannedLaneKind::HostUtility) => {
                            format!("host utility/{order}")
                        }
                        Some(mold_core::QueuePlannedLaneKind::Device) => work
                            .planned_device_id
                            .as_deref()
                            .map(|device| format!("{device}/{order}"))
                            .unwrap_or_else(|| format!("device/{order}")),
                        Some(mold_core::QueuePlannedLaneKind::Unknown(_)) => {
                            format!("assigned/{order}")
                        }
                        None if work.is_host_utility_lane()
                            || work.activity_phase == mold_core::QueueActivityPhase::Cpu =>
                        {
                            format!("host utility/{order}")
                        }
                        None => work
                            .planned_device_id
                            .as_deref()
                            .map(|device| format!("{device}/{order}"))
                            .unwrap_or_else(|| "unassigned".to_string()),
                    };
                    let finish = work
                        .estimated_finish_unix_ms
                        .map(|value| value.to_string())
                        .unwrap_or_else(|| "unknown".to_string());
                    let reason = mold_core::queue_wait::queue_work_item_reason(work)
                        .unwrap_or("none");
                    let preparation = work
                        .preparation_label()
                        .map(|label| format!(" · {label}"))
                        .unwrap_or_default();
                    let runtime = work
                        .runtime_label()
                        .map(|label| format!(" · {label}"))
                        .unwrap_or_default();
                    truncate_chars(
                        &format!(
                            "`{}` · {}{preparation}{runtime} · lane {lane} · finish {finish} · {} confidence · {reason}",
                            work.work_id,
                            work.presentation_phase(),
                            work.estimate_confidence
                        ),
                        360,
                    )
                })
                .collect::<Vec<_>>();
            for (index, chunk) in chunk_lines(&lines, DISCORD_EMBED_FIELD_VALUE_MAX)
                .into_iter()
                .enumerate()
            {
                fields.push((format!("Queue lanes {}", index + 1), chunk, false));
            }
        }
    }

    paginate_embed_fields(
        "Server Status",
        "",
        fields,
        if status.busy {
            COLOR_WARNING
        } else {
            COLOR_INFO
        },
    )
}

fn truncate_chars(value: &str, max: usize) -> String {
    if value.chars().count() <= max {
        return value.to_string();
    }
    if max <= 3 {
        return value.chars().take(max).collect();
    }
    format!("{}...", value.chars().take(max - 3).collect::<String>())
}

fn chunk_lines(lines: &[String], max_chars: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut current = String::new();
    for line in lines {
        for line_chunk in split_chars(line, max_chars) {
            let separator = usize::from(!current.is_empty());
            if current.chars().count() + separator + line_chunk.chars().count() > max_chars {
                chunks.push(std::mem::take(&mut current));
            }
            if !current.is_empty() {
                current.push('\n');
            }
            current.push_str(&line_chunk);
        }
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    chunks
}

fn split_chars(value: &str, max_chars: usize) -> Vec<String> {
    if value.is_empty() {
        return vec!["—".into()];
    }
    let chars = value.chars().collect::<Vec<_>>();
    chars
        .chunks(max_chars)
        .map(|chunk| chunk.iter().collect())
        .collect()
}

fn paginate_embed_fields(
    title: &str,
    description: &str,
    fields: Vec<(String, String, bool)>,
    color: u32,
) -> Vec<EmbedData> {
    let title = truncate_chars(title, DISCORD_EMBED_TITLE_MAX);
    let description = truncate_chars(description, DISCORD_EMBED_DESCRIPTION_MAX);
    let mut normalized = Vec::new();
    for (name, value, inline) in fields {
        let name = truncate_chars(&name, DISCORD_EMBED_FIELD_NAME_MAX);
        let chunks = split_chars(&value, DISCORD_EMBED_FIELD_VALUE_MAX);
        let chunk_count = chunks.len();
        for (index, chunk) in chunks.into_iter().enumerate() {
            let chunk_name = if chunk_count == 1 {
                name.clone()
            } else {
                truncate_chars(
                    &format!("{name} ({}/{chunk_count})", index + 1),
                    DISCORD_EMBED_FIELD_NAME_MAX,
                )
            };
            normalized.push((chunk_name, chunk, inline && chunk_count == 1));
        }
    }

    let mut pages: Vec<Vec<(String, String, bool)>> = vec![Vec::new()];
    // Reserve the maximum title width so adding "(N/M)" after pagination
    // cannot push an otherwise-full page over Discord's aggregate cap.
    let mut page_chars = DISCORD_EMBED_TITLE_MAX + description.chars().count();
    for field in normalized {
        let field_chars = field.0.chars().count() + field.1.chars().count();
        let current = pages.last_mut().expect("status page exists");
        if !current.is_empty()
            && (current.len() >= DISCORD_EMBED_FIELDS_MAX
                || page_chars + field_chars > DISCORD_EMBED_TOTAL_MAX)
        {
            pages.push(Vec::new());
            page_chars = DISCORD_EMBED_TITLE_MAX;
        }
        page_chars += field_chars;
        pages.last_mut().expect("status page exists").push(field);
    }

    let page_count = pages.len();
    pages
        .into_iter()
        .enumerate()
        .map(|(index, fields)| EmbedData {
            title: if page_count == 1 {
                title.clone()
            } else {
                truncate_chars(
                    &format!("{title} ({}/{page_count})", index + 1),
                    DISCORD_EMBED_TITLE_MAX,
                )
            },
            description: if index == 0 {
                description.clone()
            } else {
                String::new()
            },
            fields,
            color,
        })
        .collect()
}

/// Format an expand result into embed data.
pub fn format_expand_result(
    resp: &mold_core::ExpandResponse,
    original_prompt: &str,
    model_family: &str,
) -> EmbedData {
    let description = if resp.expanded.len() == 1 {
        let expanded = &resp.expanded[0];
        if expanded.chars().count() > 4000 {
            let truncated: String = expanded.chars().take(3997).collect();
            format!("{truncated}...")
        } else {
            expanded.clone()
        }
    } else {
        let mut parts = Vec::new();
        for (i, expanded) in resp.expanded.iter().enumerate() {
            let display = if expanded.chars().count() > 800 {
                let truncated: String = expanded.chars().take(797).collect();
                format!("{truncated}...")
            } else {
                expanded.clone()
            };
            parts.push(format!("**Variation {}:**\n{}", i + 1, display));
        }
        let joined = parts.join("\n\n");
        if joined.chars().count() > 4000 {
            let truncated: String = joined.chars().take(3997).collect();
            format!("{truncated}...")
        } else {
            joined
        }
    };

    EmbedData {
        title: "Prompt Expanded".to_string(),
        description,
        fields: vec![
            (
                "Original".to_string(),
                bounded_field(original_prompt),
                false,
            ),
            ("Family".to_string(), model_family.to_uppercase(), true),
            (
                "Variations".to_string(),
                resp.expanded.len().to_string(),
                true,
            ),
        ],
        color: COLOR_SUCCESS,
    }
}

/// Discord rejects an embed whose field value exceeds 1,024 characters.
fn bounded_field(value: &str) -> String {
    const LIMIT: usize = 1_024;
    if value.chars().count() > LIMIT {
        let truncated: String = value.chars().take(LIMIT - 3).collect();
        format!("{truncated}...")
    } else {
        value.to_string()
    }
}

/// Format a remix result into embed data. Every variant names the dimension
/// it varied so the reader can pick by intent, not by reading each prompt.
pub fn format_remix_result(resp: &mold_core::RemixResponse, model_family: &str) -> EmbedData {
    let mut parts = Vec::new();
    for (i, variant) in resp.variants.iter().enumerate() {
        let display = if variant.prompt.chars().count() > 800 {
            let truncated: String = variant.prompt.chars().take(797).collect();
            format!("{truncated}...")
        } else {
            variant.prompt.clone()
        };
        let labels = variant
            .dimensions
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        let heading = if labels.is_empty() {
            format!("**Variation {}:**", i + 1)
        } else {
            format!("**Variation {} ({labels}):**", i + 1)
        };
        parts.push(format!("{heading}\n{display}"));
    }
    let joined = parts.join("\n\n");
    let description = if joined.chars().count() > 4000 {
        let truncated: String = joined.chars().take(3997).collect();
        format!("{truncated}...")
    } else {
        joined
    };
    EmbedData {
        title: "Prompt Remixed".to_string(),
        description,
        fields: vec![
            (
                "Source".to_string(),
                bounded_field(&resp.source_prompt),
                false,
            ),
            ("Family".to_string(), model_family.to_uppercase(), true),
            (
                "Alternatives".to_string(),
                resp.variants.len().to_string(),
                true,
            ),
        ],
        color: COLOR_SUCCESS,
    }
}

/// Format a user's daily quota status into embed data.
pub fn format_quota(used: u32, max: Option<u32>) -> EmbedData {
    match max {
        Some(max) => {
            let remaining = max.saturating_sub(used);
            EmbedData {
                title: "Daily Quota".to_string(),
                description: format!(
                    "You have **{remaining}** of **{max}** generations remaining today."
                ),
                fields: vec![
                    ("Used Today".to_string(), used.to_string(), true),
                    ("Limit".to_string(), max.to_string(), true),
                    ("Remaining".to_string(), remaining.to_string(), true),
                ],
                color: if remaining > 0 {
                    COLOR_INFO
                } else {
                    COLOR_WARNING
                },
            }
        }
        None => EmbedData {
            title: "Daily Quota".to_string(),
            description: "No daily limit is configured. Generate freely!".to_string(),
            fields: if used > 0 {
                vec![("Used Today".to_string(), used.to_string(), true)]
            } else {
                vec![]
            },
            color: COLOR_INFO,
        },
    }
}

/// Format an error message into embed data.
pub fn format_error(msg: &str) -> EmbedData {
    EmbedData {
        title: "Error".to_string(),
        description: msg.to_string(),
        fields: vec![],
        color: COLOR_ERROR,
    }
}

/// Format seconds into a human-readable duration string.
fn format_duration_secs(secs: u64) -> String {
    let days = secs / 86400;
    let hours = (secs % 86400) / 3600;
    let minutes = (secs % 3600) / 60;

    if days > 0 {
        format!("{days}d {hours}h {minutes}m")
    } else if hours > 0 {
        format!("{hours}h {minutes}m")
    } else {
        format!("{minutes}m")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mold_core::{GpuInfo, ImageData, OutputFormat};

    #[test]
    fn format_progress_queued_zero() {
        let event = SseProgressEvent::Queued {
            position: 0,
            id: String::new(),
        };
        assert_eq!(format_progress(&event), "Starting generation...");
    }

    #[test]
    fn format_progress_dependency_wait() {
        let event = SseProgressEvent::DependencyWait {
            dependency: "ltx2.3-vae".to_string(),
            reason: "download in progress".to_string(),
        };
        assert_eq!(
            format_progress(&event),
            "Waiting for ltx2.3-vae: download in progress"
        );
    }

    #[test]
    fn format_progress_queued_nonzero() {
        let event = SseProgressEvent::Queued {
            position: 3,
            id: String::new(),
        };
        assert_eq!(format_progress(&event), "Queued — position #3");
    }

    #[test]
    fn format_progress_stage_start() {
        let event = SseProgressEvent::StageStart {
            name: "Loading T5 encoder".to_string(),
        };
        assert_eq!(format_progress(&event), "Loading: Loading T5 encoder...");
    }

    #[test]
    fn format_progress_stage_done() {
        let event = SseProgressEvent::StageDone {
            name: "T5 encoder".to_string(),
            elapsed_ms: 1234,
        };
        assert_eq!(format_progress(&event), "T5 encoder (1234ms)");
    }

    #[test]
    fn format_progress_info() {
        let event = SseProgressEvent::Info {
            message: "Hello world".to_string(),
        };
        assert_eq!(format_progress(&event), "Hello world");
    }

    #[test]
    fn format_progress_cache_hit() {
        let event = SseProgressEvent::CacheHit {
            resource: "prompt conditioning".to_string(),
        };
        assert_eq!(format_progress(&event), "Cache hit: prompt conditioning");
    }

    #[test]
    fn format_progress_denoise_step() {
        let event = SseProgressEvent::DenoiseStep {
            step: 5,
            total: 20,
            elapsed_ms: 500,
        };
        let text = format_progress(&event);
        assert!(text.contains("5/20"));
        assert!(text.contains("100ms/step"));
        assert!(text.contains('['));
    }

    #[test]
    fn format_progress_download() {
        let event = SseProgressEvent::DownloadProgress {
            filename: "model.safetensors".to_string(),
            file_index: 0,
            total_files: 3,
            bytes_downloaded: 500_000_000,
            bytes_total: 1_000_000_000,
            batch_bytes_downloaded: 500_000_000,
            batch_bytes_total: 3_000_000_000,
            batch_elapsed_ms: 10_000,
        };
        let text = format_progress(&event);
        assert!(text.contains("model.safetensors"));
        assert!(text.contains("[1/3]"));
        assert!(text.contains("50%"));
    }

    #[test]
    fn format_progress_download_done() {
        let event = SseProgressEvent::DownloadDone {
            filename: "model.safetensors".to_string(),
            file_index: 1,
            total_files: 3,
            batch_bytes_downloaded: 2_000_000_000,
            batch_bytes_total: 3_000_000_000,
            batch_elapsed_ms: 20_000,
        };
        let text = format_progress(&event);
        assert!(text.contains("Downloaded"));
        assert!(text.contains("[2/3]"));
    }

    #[test]
    fn format_progress_pull_complete() {
        let event = SseProgressEvent::PullComplete {
            model: "flux-schnell:q8".to_string(),
        };
        assert_eq!(format_progress(&event), "Model flux-schnell:q8 downloaded");
    }

    /// The identity note names the photo, the strength, and the start step —
    /// resolving both knobs through `mold_core::identity`, so an omitted one
    /// reports what the server actually applied rather than nothing.
    #[test]
    fn identity_note_reports_the_photo_and_both_knobs() {
        let mut req = mold_core::GenerateRequest {
            prompt: "a portrait".into(),
            model: "flux-dev:q8".into(),
            ..crate::commands::generate::build_generate_request(
                crate::commands::generate::BuildParams {
                    prompt: "a portrait",
                    model: "flux-dev:q8",
                    family: Some("flux"),
                    ..Default::default()
                },
            )
        };
        assert_eq!(identity_note(&req), None, "an ordinary print has no note");

        req.id_image = Some(vec![0x89, 0x50, 0x4E, 0x47]);
        req.id_image_name = Some("ada.png".into());
        assert_eq!(
            identity_note(&req).as_deref(),
            Some("ada.png \u{00b7} strength 1.00 \u{00b7} from step 0"),
            "absent knobs report mold-core's defaults, which is what rendered"
        );

        req.id_weight = Some(1.4);
        req.id_start_step = Some(3);
        assert_eq!(
            identity_note(&req).as_deref(),
            Some("ada.png \u{00b7} strength 1.40 \u{00b7} from step 3")
        );

        // A photo with no label still gets a note — the conditioning happened.
        req.id_image_name = None;
        assert!(identity_note(&req).unwrap().starts_with("attached photo"));
    }

    #[test]
    fn identity_note_becomes_a_full_width_embed_row() {
        let resp = GenerateResponse {
            mesh: None,
            audio: None,
            images: vec![ImageData {
                data: vec![],
                format: OutputFormat::Png,
                width: 1024,
                height: 1024,
                index: 0,
            }],
            generation_time_ms: 5500,
            model: "flux-dev:q8".to_string(),
            seed_used: 42,
            video: None,
            gpu: None,
            request_warnings: Vec::new(),
        };
        let plain = format_generation_result(&resp, "a portrait", None);
        assert!(
            !plain.fields.iter().any(|(name, ..)| name == "Identity"),
            "an ordinary print gains no Identity row"
        );

        let embed = format_generation_result(
            &resp,
            "a portrait",
            Some("ada.png \u{00b7} strength 1.40 \u{00b7} from step 3"),
        );
        assert_eq!(
            embed.fields.last().unwrap(),
            &(
                "Identity".to_string(),
                "ada.png \u{00b7} strength 1.40 \u{00b7} from step 3".to_string(),
                false
            )
        );
        // Blank notes never manufacture an empty row.
        let blank = format_generation_result(&resp, "a portrait", Some("   "));
        assert!(!blank.fields.iter().any(|(name, ..)| name == "Identity"));
    }

    #[test]
    fn generation_result_basic() {
        let resp = GenerateResponse {
            mesh: None,
            request_warnings: Vec::new(),
            audio: None,
            images: vec![ImageData {
                data: vec![],
                format: OutputFormat::Png,
                width: 1024,
                height: 1024,
                index: 0,
            }],
            generation_time_ms: 5500,
            model: "flux-schnell:q8".to_string(),
            seed_used: 42,
            video: None,
            gpu: None,
        };
        let embed = format_generation_result(&resp, "a cat on mars", None);
        assert_eq!(embed.title, "Image Generated");
        assert_eq!(embed.description, "a cat on mars");
        assert_eq!(embed.color, COLOR_SUCCESS);
        assert_eq!(
            embed.fields[0],
            ("Model".to_string(), "flux-schnell:q8".to_string(), true)
        );
        assert_eq!(
            embed.fields[1],
            ("Size".to_string(), "1024x1024".to_string(), true)
        );
        assert_eq!(
            embed.fields[2],
            ("Time".to_string(), "5.5s".to_string(), true)
        );
        // Seed is full-width (inline=false) to prevent word-wrap on long values
        assert_eq!(
            embed.fields[3],
            ("Seed".to_string(), "42".to_string(), false)
        );
    }

    #[test]
    fn generation_result_truncates_long_prompt() {
        let long_prompt = "a".repeat(300);
        let resp = GenerateResponse {
            mesh: None,
            request_warnings: Vec::new(),
            audio: None,
            images: vec![ImageData {
                data: vec![],
                format: OutputFormat::Png,
                width: 512,
                height: 512,
                index: 0,
            }],
            generation_time_ms: 1000,
            model: "test".to_string(),
            seed_used: 1,
            video: None,
            gpu: None,
        };
        let embed = format_generation_result(&resp, &long_prompt, None);
        assert!(embed.description.chars().count() <= 260);
        assert!(embed.description.ends_with("..."));
    }

    #[test]
    fn generation_result_video_has_frame_and_format_fields() {
        let resp = GenerateResponse {
            mesh: None,
            request_warnings: Vec::new(),
            audio: None,
            images: vec![],
            video: Some(mold_core::VideoData {
                video_only: None,
                attention_path: None,
                int8_arm: None,
                data: vec![0u8; 16],
                format: OutputFormat::Mp4,
                width: 768,
                height: 512,
                frames: 25,
                fps: 24,
                pipeline: None,
                pipeline_provenance_sha256: None,
                source_preprocessing: None,
                thumbnail: vec![],
                gif_preview: vec![],
                has_audio: true,
                duration_ms: Some(1000),
                audio_sample_rate: Some(44_100),
                audio_channels: Some(2),
            }),
            generation_time_ms: 10_000,
            model: "ltx-2-19b-distilled:fp8".to_string(),
            seed_used: 7,
            gpu: None,
        };
        let embed = format_generation_result(&resp, "a drone shot", None);
        assert_eq!(embed.title, "Video Generated");
        assert!(embed
            .fields
            .iter()
            .any(|(k, v, _)| k == "Frames" && v == "25 @ 24 fps"));
        assert!(embed
            .fields
            .iter()
            .any(|(k, v, _)| k == "Format" && v == "MP4"));
        assert!(embed
            .fields
            .iter()
            .any(|(k, v, _)| k == "Audio" && v == "yes"));
        assert!(embed
            .fields
            .iter()
            .any(|(k, v, _)| k == "Size" && v == "768x512"));
    }

    /// A mesh gets its own summary: geometry counts, extent, and the GLB
    /// container, never an empty "Size" row from the image arm.
    #[test]
    fn generation_result_mesh_lists_tris_verts_bounds_and_seed() {
        let resp = GenerateResponse {
            mesh: Some(mold_core::MeshData {
                data: vec![0u8; 16],
                format: OutputFormat::Glb,
                vertex_count: 24_576,
                face_count: 49_152,
                bounds_min: [-0.5, -0.4, -0.3],
                bounds_max: [0.5, 0.4, 0.3],
                textured: false,
                poster: vec![],
                poster_width: 512,
                poster_height: 512,
            }),
            request_warnings: Vec::new(),
            audio: None,
            images: vec![],
            video: None,
            generation_time_ms: 4_000,
            model: mold_core::manifest::HUNYUAN3D_DEFAULT_MODEL.to_string(),
            seed_used: 9,
            gpu: Some(1),
        };
        let embed = format_generation_result(&resp, "", None);
        assert_eq!(embed.title, "Mesh Generated");
        let field = |name: &str| {
            embed
                .fields
                .iter()
                .find(|(k, _, _)| k == name)
                .map(|(_, v, _)| v.as_str())
        };
        assert_eq!(field("Triangles"), Some("49,152"));
        assert_eq!(field("Vertices"), Some("24,576"));
        assert_eq!(field("Bounds"), Some("1.00 × 0.80 × 0.60"));
        assert_eq!(field("Format"), Some("GLB"));
        assert_eq!(field("Seed"), Some("9"));
        assert_eq!(field("Time"), Some("4.0s"));
        assert_eq!(field("Device"), Some("GPU 1"));
        assert!(field("Size").is_none(), "a mesh has no raster size");
        assert!(
            embed.description.is_empty(),
            "an empty prompt is not padded with placeholder text"
        );
    }

    #[test]
    fn generation_result_video_gif_shows_gif_label() {
        let resp = GenerateResponse {
            mesh: None,
            request_warnings: Vec::new(),
            audio: None,
            images: vec![],
            video: Some(mold_core::VideoData {
                video_only: None,
                attention_path: None,
                int8_arm: None,
                data: vec![0u8; 16],
                format: OutputFormat::Gif,
                width: 512,
                height: 512,
                frames: 17,
                fps: 12,
                pipeline: None,
                pipeline_provenance_sha256: None,
                source_preprocessing: None,
                thumbnail: vec![],
                gif_preview: vec![],
                has_audio: false,
                duration_ms: None,
                audio_sample_rate: None,
                audio_channels: None,
            }),
            generation_time_ms: 500,
            model: "ltx-video-0.9.6-distilled:bf16".to_string(),
            seed_used: 3,
            gpu: None,
        };
        let embed = format_generation_result(&resp, "loop", None);
        assert_eq!(embed.title, "Video Generated");
        assert!(embed
            .fields
            .iter()
            .any(|(k, v, _)| k == "Format" && v == "GIF"));
        assert!(!embed.fields.iter().any(|(k, _, _)| k == "Audio"));
    }

    #[test]
    fn generation_result_truncates_non_ascii_prompt_safely() {
        // Multi-byte characters: each is 4 bytes in UTF-8
        let long_prompt = "\u{1F600}".repeat(300); // 300 emoji characters
        let resp = GenerateResponse {
            mesh: None,
            request_warnings: Vec::new(),
            audio: None,
            images: vec![ImageData {
                data: vec![],
                format: OutputFormat::Png,
                width: 512,
                height: 512,
                index: 0,
            }],
            generation_time_ms: 1000,
            model: "test".to_string(),
            seed_used: 1,
            video: None,
            gpu: None,
        };
        let embed = format_generation_result(&resp, &long_prompt, None);
        assert!(embed.description.chars().count() <= 260);
        assert!(embed.description.ends_with("..."));
    }

    #[test]
    fn model_list_empty() {
        let embed = format_model_list(&[]);
        assert!(embed.description.contains("No models"));
        assert_eq!(embed.color, COLOR_WARNING);
    }

    #[test]
    fn model_list_with_models() {
        let models = vec![ModelInfoExtended {
            runtime_available: None,
            runtime_unavailable_reason: None,
            info: mold_core::ModelInfo {
                name: "flux-schnell:q8".to_string(),
                family: "flux".to_string(),
                size_gb: 4.5,
                is_loaded: true,
                last_used: None,
                hf_repo: "test/repo".to_string(),
            },
            defaults: mold_core::ModelDefaults {
                default_steps: 4,
                default_guidance: 0.0,
                default_width: 1024,
                default_height: 1024,
                description: "Fast flux".to_string(),
                ..Default::default()
            },
            downloaded: true,
            disk_usage_bytes: None,
            remaining_download_bytes: None,
            display_name: None,
            kind: None,
            modality: None,
            nsfw: None,
            supports_audio: None,
            supports_extend: None,
            supports_sequence: None,
            extend_default_overlap_frames: None,
            guidance_capabilities: None,
            source_image: None,
            generation_profile: None,
            supports_identity: None,
            supports_duration_prediction: None,
            runtime_ready: None,
            runtime_readiness_error: None,
        }];
        let embed = format_model_list(&models);
        assert!(embed.description.contains("**FLUX**"));
        assert!(embed.description.contains("**flux-schnell:q8**"));
        assert!(embed.description.contains("[loaded]"));
        assert!(embed.description.contains("4.5GB"));
    }

    #[test]
    fn model_list_groups_families_and_shows_all_variants() {
        let models = vec![
            ModelInfoExtended {
                runtime_available: None,
                runtime_unavailable_reason: None,
                info: mold_core::ModelInfo {
                    name: "flux-schnell:q8".to_string(),
                    family: "flux".to_string(),
                    size_gb: 11.8,
                    is_loaded: true,
                    last_used: None,
                    hf_repo: "test/repo".to_string(),
                },
                defaults: mold_core::ModelDefaults {
                    default_steps: 4,
                    default_guidance: 0.0,
                    default_width: 1024,
                    default_height: 1024,
                    description: "FLUX Schnell Q8".to_string(),
                    ..Default::default()
                },
                downloaded: true,
                disk_usage_bytes: None,
                remaining_download_bytes: None,
                display_name: None,
                kind: None,
                modality: None,
                nsfw: None,
                supports_audio: None,
                supports_extend: None,
                supports_sequence: None,
                extend_default_overlap_frames: None,
                guidance_capabilities: None,
                source_image: None,
                generation_profile: None,
                supports_identity: None,
                supports_duration_prediction: None,
                runtime_ready: None,
                runtime_readiness_error: None,
            },
            ModelInfoExtended {
                runtime_available: None,
                runtime_unavailable_reason: None,
                info: mold_core::ModelInfo {
                    name: "flux-dev:q4".to_string(),
                    family: "flux".to_string(),
                    size_gb: 7.0,
                    is_loaded: false,
                    last_used: None,
                    hf_repo: "test/repo".to_string(),
                },
                defaults: mold_core::ModelDefaults {
                    default_steps: 20,
                    default_guidance: 3.5,
                    default_width: 1024,
                    default_height: 1024,
                    description: "FLUX Dev Q4".to_string(),
                    ..Default::default()
                },
                downloaded: false,
                disk_usage_bytes: None,
                remaining_download_bytes: None,
                display_name: None,
                kind: None,
                modality: None,
                nsfw: None,
                supports_audio: None,
                supports_extend: None,
                supports_sequence: None,
                extend_default_overlap_frames: None,
                guidance_capabilities: None,
                source_image: None,
                generation_profile: None,
                supports_identity: None,
                supports_duration_prediction: None,
                runtime_ready: None,
                runtime_readiness_error: None,
            },
            ModelInfoExtended {
                runtime_available: None,
                runtime_unavailable_reason: None,
                info: mold_core::ModelInfo {
                    name: "wuerstchen-v2:fp16".to_string(),
                    family: "wuerstchen".to_string(),
                    size_gb: 5.0,
                    is_loaded: false,
                    last_used: None,
                    hf_repo: "test/repo".to_string(),
                },
                defaults: mold_core::ModelDefaults {
                    default_steps: 30,
                    default_guidance: 4.0,
                    default_width: 1024,
                    default_height: 1024,
                    description: "Wuerstchen v2".to_string(),
                    ..Default::default()
                },
                downloaded: false,
                disk_usage_bytes: None,
                remaining_download_bytes: None,
                display_name: None,
                kind: None,
                modality: None,
                nsfw: None,
                supports_audio: None,
                supports_extend: None,
                supports_sequence: None,
                extend_default_overlap_frames: None,
                guidance_capabilities: None,
                source_image: None,
                generation_profile: None,
                supports_identity: None,
                supports_duration_prediction: None,
                runtime_ready: None,
                runtime_readiness_error: None,
            },
        ];
        let embed = format_model_list(&models);
        // FLUX section: loaded model is bold, undownloaded is plain
        assert!(embed.description.contains("**FLUX**"));
        assert!(embed.description.contains("**flux-schnell:q8**"));
        assert!(embed.description.contains("flux-dev:q4"));
        assert!(!embed.description.contains("**flux-dev:q4**"));
        // Wuerstchen section, variant name visible
        assert!(embed.description.contains("**WUERSTCHEN**"));
        assert!(!embed.description.contains("(alpha)"));
        assert!(embed.description.contains("wuerstchen-v2:fp16"));
        assert!(!embed.description.contains("**wuerstchen-v2:fp16**"));
    }

    #[test]
    fn server_status_idle() {
        let status = ServerStatus {
            version: "0.2.0".to_string(),
            git_sha: Some("abc1234".to_string()),
            build_date: Some("2026-03-25".to_string()),
            models_loaded: vec!["flux-schnell:q8".to_string()],
            busy: false,
            current_generation: None,
            gpu_info: Some(GpuInfo {
                name: "RTX 4090".to_string(),
                vram_total_mb: 24564,
                vram_used_mb: 8192,
                backend: None,
            }),
            uptime_secs: 3661,
            hostname: Some("hal9000".to_string()),
            memory_status: Some("VRAM: 16.0 GB free".to_string()),
            gpus: Some(vec![
                mold_core::GpuWorkerStatus {
                    ordinal: 0,
                    name: "RTX 4090".to_string(),
                    vram_total_bytes: 24 * 1024_u64.pow(3),
                    vram_used_bytes: 8 * 1024_u64.pow(3),
                    loaded_model: Some("flux-schnell:q8".to_string()),
                    state: mold_core::GpuWorkerState::Generating,
                },
                mold_core::GpuWorkerStatus {
                    ordinal: 1,
                    name: "RTX 4090".to_string(),
                    vram_total_bytes: 24 * 1024_u64.pow(3),
                    vram_used_bytes: 4 * 1024_u64.pow(3),
                    loaded_model: None,
                    state: mold_core::GpuWorkerState::Idle,
                },
            ]),
            queue_depth: None,
            queue_capacity: None,
            queue_paused: None,
            instance_id: None,
            models_disk: None,
            host_memory: None,
            durable_media: None,
        };
        let embed = format_server_status(&status);
        assert_eq!(embed.title, "Server Status");
        assert!(embed
            .fields
            .iter()
            .any(|(k, v, _)| k == "Status" && v == "Idle"));
        assert!(embed
            .fields
            .iter()
            .any(|(k, v, _)| k == "Uptime" && v == "1h 1m"));
        assert!(embed
            .fields
            .iter()
            .any(|(k, v, _)| { k == "GPUs" && v == "2× RTX 4090 (12288MB / 49152MB VRAM)" }));
        assert_eq!(embed.color, COLOR_INFO);
    }

    #[test]
    fn server_status_busy() {
        let status = ServerStatus {
            version: "0.2.0".to_string(),
            git_sha: None,
            build_date: None,
            models_loaded: vec![],
            busy: true,
            current_generation: None,
            gpu_info: None,
            uptime_secs: 60,
            hostname: None,
            memory_status: None,
            gpus: None,
            queue_depth: None,
            queue_capacity: None,
            queue_paused: None,
            instance_id: None,
            models_disk: None,
            host_memory: None,
            durable_media: None,
        };
        let embed = format_server_status(&status);
        assert!(embed
            .fields
            .iter()
            .any(|(k, v, _)| k == "Status" && v == "Generating..."));
        assert!(embed
            .fields
            .iter()
            .any(|(k, v, _)| k == "Models Loaded" && v == "None"));
        assert_eq!(embed.color, COLOR_WARNING);
    }

    #[test]
    fn sixty_four_device_status_paginates_within_every_discord_embed_limit() {
        let status = ServerStatus {
            version: "0.20.2".into(),
            git_sha: None,
            build_date: None,
            models_loaded: vec!["flux-dev:q8".into()],
            busy: false,
            current_generation: None,
            gpu_info: None,
            uptime_secs: 60,
            hostname: Some("fleet".into()),
            memory_status: None,
            gpus: None,
            queue_depth: Some(64),
            queue_capacity: Some(256),
            queue_paused: Some(false),
            instance_id: None,
            models_disk: None,
            host_memory: None,
            durable_media: None,
        };
        let mut inventory = mold_core::DeviceState {
            devices: (0..64)
                .map(|ordinal| {
                    serde_json::from_value(serde_json::json!({
                        "id": format!("cuda:{ordinal:032x}"),
                        "backend": "cuda",
                        "ordinal": ordinal,
                        "device_kind": "full_gpu",
                        "nvml_uuid": null,
                        "physical_uuid": null,
                        "mig_uuid": null,
                        "mig_parent_uuid": null,
                        "mig_profile": null,
                        "name": format!("Synthetic accelerator {ordinal}"),
                        "pci_bus_id": null,
                        "compute_capability": "10.0",
                        "memory": {
                            "total_bytes": 192_u64 * 1024 * 1024 * 1024,
                            "used_bytes": ordinal as u64 * 1024 * 1024,
                            "mold_used_bytes": null,
                            "other_used_bytes": null
                        },
                        "telemetry": {
                            "utilization_percent": ordinal,
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
                })
                .collect(),
            plan_version: 7,
        };
        inventory.devices[1].desired_enabled = false;
        inventory.devices[1].admin_state = mold_core::DeviceAdminState::Disabled;
        inventory.devices[2].desired_enabled = false;
        inventory.devices[2].admin_state = mold_core::DeviceAdminState::Draining;
        inventory.devices[3].health = mold_core::DeviceHealth::Unavailable;
        inventory.devices[3].schedulable = false;
        inventory.devices[4].device_kind = mold_core::DeviceKind::Mig;
        inventory.devices[4].mig_profile = Some("1g.23gb".into());
        inventory.devices[5].memory = mold_core::DeviceMemoryInfo {
            total_bytes: None,
            used_bytes: None,
            mold_used_bytes: None,
            other_used_bytes: None,
        };

        let pages = format_server_status_pages(&status, Some(&inventory), None);
        assert!(pages.len() > 1, "64 detailed devices must paginate");
        let all_values = pages
            .iter()
            .flat_map(|page| page.fields.iter().map(|(_, value, _)| value.as_str()))
            .collect::<Vec<_>>()
            .join("\n");
        for ordinal in 0..64 {
            assert!(
                all_values.contains(&format!("GPU {ordinal} ·")),
                "GPU {ordinal} missing from Discord pages"
            );
        }
        assert!(all_values.contains("disabled"));
        assert!(all_values.contains("finishing current work"));
        assert!(all_values.contains("unavailable"));
        assert!(all_values.contains("MIG 1g.23gb"));
        assert!(all_values.contains("VRAM unknown"));
        assert!(!all_values.contains("0/0MB"));

        for page in &pages {
            assert!(page.title.chars().count() <= DISCORD_EMBED_TITLE_MAX);
            assert!(page.description.chars().count() <= DISCORD_EMBED_DESCRIPTION_MAX);
            assert!(page.fields.len() <= DISCORD_EMBED_FIELDS_MAX);
            let total = page.title.chars().count()
                + page.description.chars().count()
                + page
                    .fields
                    .iter()
                    .map(|(name, value, _)| {
                        assert!(name.chars().count() <= DISCORD_EMBED_FIELD_NAME_MAX);
                        assert!(value.chars().count() <= DISCORD_EMBED_FIELD_VALUE_MAX);
                        name.chars().count() + value.chars().count()
                    })
                    .sum::<usize>();
            assert!(total <= DISCORD_EMBED_TOTAL_MAX, "{total}");
        }
    }

    #[test]
    fn queue_summary_counts_only_typed_blocks_and_keeps_lane_observability() {
        let status = ServerStatus {
            version: "0.20.2".into(),
            git_sha: None,
            build_date: None,
            models_loaded: vec![],
            busy: true,
            current_generation: None,
            gpu_info: None,
            uptime_secs: 1,
            hostname: None,
            memory_status: None,
            gpus: None,
            queue_depth: Some(2),
            queue_capacity: Some(8),
            queue_paused: Some(false),
            instance_id: None,
            models_disk: None,
            host_memory: None,
            durable_media: None,
        };
        let queue = mold_core::QueueListingWire {
            entries: vec![],
            live_only_entries: vec![],
            plan: Some(mold_core::QueuePlan {
                work_items: vec![
                    mold_core::QueueWorkItem {
                        work_id: "assigned".into(),
                        activity_phase: mold_core::QueueActivityPhase::Active,
                        planned_lane_kind: Some(mold_core::QueuePlannedLaneKind::Device),
                        planned_device_id: Some("cuda:stable-a".into()),
                        lane_order: Some(0),
                        estimate_confidence: mold_core::QueueEstimateConfidence::High,
                        assignment_reason: Some("warm_model".into()),
                        ..Default::default()
                    },
                    mold_core::QueueWorkItem {
                        work_id: "blocked".into(),
                        activity_phase: mold_core::QueueActivityPhase::Blocked,
                        blocked_reason: Some(mold_core::QueueBlockedReason::InsufficientVram),
                        estimate_confidence: mold_core::QueueEstimateConfidence::Low,
                        ..Default::default()
                    },
                    mold_core::QueueWorkItem {
                        work_id: "ordinary-wait".into(),
                        activity_phase: mold_core::QueueActivityPhase::Queued,
                        blocked_reason: Some(mold_core::QueueBlockedReason::NoSchedulableDevice),
                        estimate_confidence: mold_core::QueueEstimateConfidence::Low,
                        ..Default::default()
                    },
                ],
                ..Default::default()
            }),
            page: None,
        };

        let fields = format_server_status_pages(&status, None, Some(&queue))
            .iter()
            .flat_map(|page| page.fields.iter())
            .map(|(name, value, _)| format!("{name}: {value}"))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(fields.contains("3 planned · 1 blocked"));
        assert!(fields.contains("cuda:stable-a/0"));
        assert!(fields.contains("high confidence"));
        assert!(fields.contains("warm_model"));
        assert!(fields.contains("insufficient_vram"));
        assert!(!fields.contains("no_schedulable_device"));
    }

    #[test]
    fn queue_summary_hides_legacy_queued_host_utility_identity() {
        let status = ServerStatus {
            version: "0.20.2".into(),
            git_sha: None,
            build_date: None,
            models_loaded: vec![],
            busy: true,
            current_generation: None,
            gpu_info: None,
            uptime_secs: 1,
            hostname: None,
            memory_status: None,
            gpus: None,
            queue_depth: Some(3),
            queue_capacity: Some(8),
            queue_paused: Some(false),
            instance_id: None,
            models_disk: None,
            host_memory: None,
            durable_media: None,
        };
        let queue = mold_core::QueueListingWire {
            entries: vec![],
            live_only_entries: vec![],
            plan: Some(mold_core::QueuePlan {
                work_items: vec![
                    mold_core::QueueWorkItem {
                        work_id: "legacy-upscale".into(),
                        activity_phase: mold_core::QueueActivityPhase::Queued,
                        planned_lane_kind: None,
                        planned_device_id: Some("cpu:utility:0".into()),
                        lane_order: Some(1),
                        ..Default::default()
                    },
                    mold_core::QueueWorkItem {
                        work_id: "typed-host".into(),
                        planned_lane_kind: Some(mold_core::QueuePlannedLaneKind::HostUtility),
                        planned_device_id: Some("typed-host-internal".into()),
                        lane_order: Some(2),
                        ..Default::default()
                    },
                    mold_core::QueueWorkItem {
                        work_id: "future".into(),
                        planned_lane_kind: Some(mold_core::QueuePlannedLaneKind::Unknown(
                            "remote_utility".into(),
                        )),
                        planned_device_id: Some("future-internal-id".into()),
                        lane_order: Some(3),
                        ..Default::default()
                    },
                ],
                ..Default::default()
            }),
            page: None,
        };

        let fields = format_server_status_pages(&status, None, Some(&queue))
            .iter()
            .flat_map(|page| page.fields.iter())
            .map(|(_, value, _)| value.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(fields.contains("host utility/1"));
        assert!(fields.contains("host utility/2"));
        assert!(fields.contains("assigned/3"));
        assert!(!fields.contains("cpu:utility:0"));
        assert!(!fields.contains("typed-host-internal"));
        assert!(!fields.contains("future-internal-id"));
    }

    #[test]
    fn queue_summary_separates_total_backlog_hydrated_page_and_runtime_capacity() {
        let status = ServerStatus {
            version: "test".into(),
            git_sha: None,
            build_date: None,
            models_loaded: vec![],
            busy: true,
            current_generation: None,
            gpu_info: None,
            uptime_secs: 1,
            hostname: None,
            memory_status: None,
            gpus: None,
            queue_depth: Some(120),
            queue_capacity: Some(8),
            queue_paused: Some(false),
            instance_id: None,
            models_disk: None,
            host_memory: None,
            durable_media: None,
        };
        let row = mold_core::QueueJobEntryWire {
            id: "job".into(),
            model: "flux-dev:q8".into(),
            state: "queued".into(),
            started_at_unix_ms: 0,
            position: 0,
            gpu: None,
            target_gpu: None,
            seed_pinned: None,
            metadata: None,
            durable: Some(true),
            held_reason: None,
            ..Default::default()
        };
        let queue = mold_core::QueueListingWire {
            entries: vec![row; 6],
            page: Some(mold_core::QueuePage {
                limit: 6,
                offset: 0,
                returned: 6,
                next_cursor: Some("opaque".into()),
            }),
            ..Default::default()
        };

        let summary = format_server_status_pages(&status, None, Some(&queue))
            .into_iter()
            .flat_map(|page| page.fields)
            .find(|(name, _, _)| name == "Queue")
            .unwrap()
            .1;
        assert!(summary.contains("120 total backlog"));
        assert!(summary.contains("6 hydrated page rows"));
        assert!(summary.contains("runtime capacity 8"));
        assert!(!summary.contains("active/queued"));
        assert!(!summary.contains("120/8"));
    }

    #[test]
    fn error_embed() {
        let embed = format_error("Something went wrong");
        assert_eq!(embed.title, "Error");
        assert_eq!(embed.description, "Something went wrong");
        assert_eq!(embed.color, COLOR_ERROR);
    }

    #[test]
    fn duration_formatting() {
        assert_eq!(format_duration_secs(0), "0m");
        assert_eq!(format_duration_secs(59), "0m");
        assert_eq!(format_duration_secs(60), "1m");
        assert_eq!(format_duration_secs(3600), "1h 0m");
        assert_eq!(format_duration_secs(3661), "1h 1m");
        assert_eq!(format_duration_secs(86400), "1d 0h 0m");
        assert_eq!(format_duration_secs(90061), "1d 1h 1m");
    }

    #[test]
    fn progress_bar_boundaries() {
        let bar_start = progress_bar(0, 20);
        assert!(bar_start.contains('['));
        assert!(bar_start.contains(']'));

        let bar_mid = progress_bar(10, 20);
        assert!(bar_mid.contains('='));
        assert!(bar_mid.contains('>'));

        let bar_end = progress_bar(20, 20);
        assert!(bar_end.contains('='));
        assert!(!bar_end.contains('>'));
    }

    #[test]
    fn progress_bar_zero_total() {
        let bar = progress_bar(0, 0);
        assert!(bar.contains('['));
        assert!(bar.contains(']'));
    }

    // --- format_expand_result tests ---

    #[test]
    fn expand_result_single_variation() {
        let resp = mold_core::ExpandResponse {
            original: "a cat".to_string(),
            expanded: vec!["a fluffy orange tabby cat sitting on a windowsill".to_string()],
        };
        let embed = format_expand_result(&resp, "a cat", "flux");
        assert_eq!(embed.title, "Prompt Expanded");
        assert!(embed.description.contains("fluffy orange tabby"));
        assert!(embed
            .fields
            .iter()
            .any(|(k, v, _)| k == "Original" && v == "a cat"));
        assert!(embed
            .fields
            .iter()
            .any(|(k, v, _)| k == "Family" && v == "FLUX"));
        assert!(embed
            .fields
            .iter()
            .any(|(k, v, _)| k == "Variations" && v == "1"));
        assert_eq!(embed.color, COLOR_SUCCESS);
    }

    #[test]
    fn expand_result_multiple_variations() {
        let resp = mold_core::ExpandResponse {
            original: "sunset".to_string(),
            expanded: vec![
                "golden sunset over the ocean".to_string(),
                "dramatic red sunset behind mountains".to_string(),
                "pastel sunset with silhouetted trees".to_string(),
            ],
        };
        let embed = format_expand_result(&resp, "sunset", "sdxl");
        assert!(embed.description.contains("Variation 1:"));
        assert!(embed.description.contains("Variation 2:"));
        assert!(embed.description.contains("Variation 3:"));
        assert!(embed.description.contains("golden sunset"));
        assert!(embed
            .fields
            .iter()
            .any(|(k, v, _)| k == "Variations" && v == "3"));
        assert!(embed
            .fields
            .iter()
            .any(|(k, v, _)| k == "Family" && v == "SDXL"));
    }

    #[test]
    fn expand_result_truncates_long_single_prompt() {
        let long_prompt = "a".repeat(5000);
        let resp = mold_core::ExpandResponse {
            original: "test".to_string(),
            expanded: vec![long_prompt],
        };
        let embed = format_expand_result(&resp, "test", "flux");
        assert!(embed.description.chars().count() <= 4003);
        assert!(embed.description.ends_with("..."));
    }

    // --- format_quota tests ---

    #[test]
    fn quota_with_limit_and_remaining() {
        let embed = format_quota(3, Some(10));
        assert_eq!(embed.title, "Daily Quota");
        assert!(embed.description.contains("**7**"));
        assert!(embed.description.contains("**10**"));
        assert!(embed
            .fields
            .iter()
            .any(|(k, v, _)| k == "Used Today" && v == "3"));
        assert!(embed
            .fields
            .iter()
            .any(|(k, v, _)| k == "Remaining" && v == "7"));
        assert_eq!(embed.color, COLOR_INFO);
    }

    #[test]
    fn quota_exhausted() {
        let embed = format_quota(10, Some(10));
        assert!(embed.description.contains("**0**"));
        assert_eq!(embed.color, COLOR_WARNING);
    }

    #[test]
    fn quota_unlimited_no_usage() {
        let embed = format_quota(0, None);
        assert!(embed.description.contains("No daily limit"));
        assert!(embed.fields.is_empty());
        assert_eq!(embed.color, COLOR_INFO);
    }

    #[test]
    fn quota_unlimited_with_usage() {
        let embed = format_quota(5, None);
        assert!(embed.description.contains("No daily limit"));
        assert!(embed
            .fields
            .iter()
            .any(|(k, v, _)| k == "Used Today" && v == "5"));
    }

    #[test]
    fn expand_result_sd15_family() {
        let resp = mold_core::ExpandResponse {
            original: "dog".to_string(),
            expanded: vec!["cute dog, photorealistic, 8k".to_string()],
        };
        let embed = format_expand_result(&resp, "dog", "sd15");
        assert!(embed
            .fields
            .iter()
            .any(|(k, v, _)| k == "Family" && v == "SD15"));
    }
}
