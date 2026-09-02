//! The prompting corpus: one set of per-family, per-task, and per-model
//! prompting guides that every surface reads.
//!
//! The markdown files beside this module are the single source of truth for
//! how to write a prompt for each model family. Two consumers render them:
//!
//! - the agent skill bundle (`mold skill install`) copies every file under
//!   `references/prompting/` and links them from `SKILL.md`;
//! - prompt expansion and remix (`mold expand`, `mold remix`, `--expand`,
//!   `/api/expand`, `/api/remix`) inject the [`expansion_excerpt`] of the
//!   resolved route into the LLM system prompt.
//!
//! Numbers live in Rust (word limits, identity matchers); prose lives in
//! markdown. A guide writes `{{word_limit}}` wherever it states its budget so
//! the figure is never typed twice. Sections titled `CLI` and `Sources` are
//! agent-only and never reach the expander; everything else is shared.

use crate::generation_profile::canonical_family;
use crate::{
    ExpandContext, ExpandReference, ExpandReferenceRole, ExpandTask, GenerationReferenceKind,
};

/// The always-read practice shared by every family.
pub const SHARED_PATH: &str = "shared.md";
const SHARED_MD: &str = include_str!("shared.md");

/// Section titles (H2) that are rendered for agents but excluded from the
/// expansion excerpt handed to the LLM.
pub const AGENT_ONLY_SECTIONS: &[&str] = &["CLI", "Sources"];

/// Upper bound on the words an expansion excerpt may carry so the default
/// local expander (Qwen3 1.7B) and OpenAI-compatible hosts at a 2,048-token
/// default context still see the user prompt after the guide.
pub const EXCERPT_WORD_BUDGET: usize = 700;

/// One base guide per manifest family.
#[derive(Clone, Copy, Debug)]
pub struct FamilyGuide {
    /// Manifest family id (`ModelManifest.family`).
    pub family: &'static str,
    /// Additional spellings accepted on the wire beyond
    /// [`canonical_family`]'s normalisation.
    pub aliases: &'static [&'static str],
    /// Corpus-relative path (`families/<family>.md`).
    pub path: &'static str,
    pub contents: &'static str,
    /// Default word budget for an expanded prompt.
    pub word_limit: u32,
}

/// How a task leaf binds to a model identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IdentityMatch {
    /// Any identity in the family.
    Any,
    /// The model name contains this fragment (for example `-t2v-`).
    Contains(&'static str),
    /// The model name starts with this prefix (for example `minimax-h3-ref2va:`).
    Prefix(&'static str),
}

impl IdentityMatch {
    fn matches(self, model: &str) -> bool {
        match self {
            Self::Any => true,
            Self::Contains(fragment) => model.contains(fragment),
            Self::Prefix(prefix) => model.starts_with(prefix),
        }
    }

    fn is_specific(self) -> bool {
        !matches!(self, Self::Any)
    }
}

/// A task-specific leaf added below the family base.
#[derive(Clone, Copy, Debug)]
pub struct TaskLeaf {
    pub family: &'static str,
    /// Human label used in `SKILL.md`'s route list.
    pub label: &'static str,
    /// Corpus-relative path (`<family>/<leaf>.md`).
    pub path: &'static str,
    pub contents: &'static str,
    /// Expansion tasks this leaf governs. Empty means the leaf is selected
    /// only explicitly (a workflow with no expansion contract of its own).
    pub tasks: &'static [ExpandTask],
    /// Identity binding used when no task is supplied.
    pub identity: IdentityMatch,
    /// Word budget override for this task.
    pub word_limit: Option<u32>,
    /// The leaf replaces the family base in the expansion excerpt (an
    /// audio-only task must not inherit the family's camera language).
    pub standalone: bool,
    /// The leaf applies only when the request carries a source video (Dub-It
    /// re-voices an existing clip; plain audio-to-video does not).
    pub needs_source_video: bool,
}

/// Request facts that refine leaf selection beyond family, identity, and task.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RouteHints {
    /// A source video is attached (video-to-video, retake, lip dub).
    pub source_video: bool,
}

/// A per-checkpoint quirk leaf added after the task leaf.
#[derive(Clone, Copy, Debug)]
pub struct ModelLeaf {
    pub family: &'static str,
    /// Base model names (identity with the `:tag` removed) this leaf covers.
    pub models: &'static [&'static str],
    pub label: &'static str,
    /// Corpus-relative path (`models/<name>.md`).
    pub path: &'static str,
    pub contents: &'static str,
}

macro_rules! family {
    ($family:literal, $file:literal, $limit:expr) => {
        family!($family, $file, $limit, &[])
    };
    ($family:literal, $file:literal, $limit:expr, $aliases:expr) => {
        FamilyGuide {
            family: $family,
            aliases: $aliases,
            path: concat!("families/", $file),
            contents: include_str!(concat!("families/", $file)),
            word_limit: $limit,
        }
    };
}

/// Every family base guide, in `SKILL.md` order.
pub const FAMILY_GUIDES: &[FamilyGuide] = &[
    family!("flux", "flux.md", 150, &["flux.1", "flux-1"]),
    family!("flux2", "flux2.md", 120),
    family!("sd15", "sd15.md", 50, &["sd1.5", "stable-diffusion-1.5"]),
    family!("sdxl", "sdxl.md", 60),
    family!("sd3", "sd3.md", 150, &["sd3.5", "sd35"]),
    family!("z-image", "z-image.md", 150, &["zimage", "z_image"]),
    family!("hunyuan3d", "hunyuan3d.md", 40),
    family!("wuerstchen", "wuerstchen.md", 50, &["wuerstchen-v2"]),
    family!("qwen-image", "qwen-image.md", 180, &["qwen_image"]),
    family!(
        "qwen-image-edit",
        "qwen-image-edit.md",
        100,
        &["qwen_image_edit"]
    ),
    family!("ltx-video", "ltx-video.md", 150, &["ltx_video", "ltxvideo"]),
    family!(
        "ltx2",
        "ltx2.md",
        200,
        &["ltx-2.3", "ltx-2.5", "ltx2.3", "ltx2.5"]
    ),
    family!(
        "wan",
        "wan.md",
        100,
        &["wan2.1", "wan2.2", "wan21", "wan22"]
    ),
    family!("minimax-h3", "minimax-h3.md", 250),
    family!("upscaler", "upscaler.md", 20),
];

macro_rules! leaf {
    ($family:literal, $label:literal, $file:literal, $tasks:expr, $identity:expr, $limit:expr) => {
        leaf!($family, $label, $file, $tasks, $identity, $limit, false, false)
    };
    ($family:literal, $label:literal, $file:literal, $tasks:expr, $identity:expr, $limit:expr, $standalone:expr) => {
        leaf!(
            $family,
            $label,
            $file,
            $tasks,
            $identity,
            $limit,
            $standalone,
            false
        )
    };
    ($family:literal, $label:literal, $file:literal, $tasks:expr, $identity:expr, $limit:expr, $standalone:expr, $source_video:expr) => {
        TaskLeaf {
            family: $family,
            label: $label,
            path: $file,
            contents: include_str!($file),
            tasks: $tasks,
            identity: $identity,
            word_limit: $limit,
            standalone: $standalone,
            needs_source_video: $source_video,
        }
    };
}

/// Every task leaf, in `SKILL.md` order.
pub const TASK_LEAVES: &[TaskLeaf] = &[
    leaf!(
        "minimax-h3",
        "MiniMax H3 base modes",
        "minimax-h3/base-modes.md",
        &[
            ExpandTask::TextToVideo,
            ExpandTask::ImageToVideo,
            ExpandTask::KeyframeInterpolation,
        ],
        IdentityMatch::Prefix("minimax-h3-fl2va"),
        Some(250)
    ),
    leaf!(
        "minimax-h3",
        "MiniMax H3 Ref2VA",
        "minimax-h3/ref2va.md",
        &[ExpandTask::ReferenceToAudioVideo],
        IdentityMatch::Prefix("minimax-h3-ref2va"),
        Some(300)
    ),
    leaf!(
        "wan",
        "Wan text-to-video",
        "wan/text-to-video.md",
        &[ExpandTask::TextToVideo],
        IdentityMatch::Contains("-t2v-"),
        Some(100)
    ),
    leaf!(
        "wan",
        "Wan image-conditioned",
        "wan/image-conditioned.md",
        &[ExpandTask::ImageToVideo, ExpandTask::KeyframeInterpolation],
        IdentityMatch::Contains("i2v-"),
        Some(80)
    ),
    leaf!(
        "ltx2",
        "LTX-2 Dub-It",
        "ltx2/dub-it.md",
        &[ExpandTask::AudioDrivenVideo],
        IdentityMatch::Any,
        Some(120),
        false,
        true
    ),
    leaf!(
        "ltx2",
        "LTX-2 text-to-audio",
        "ltx2/text-to-audio.md",
        &[ExpandTask::TextToAudio],
        IdentityMatch::Any,
        Some(120),
        true
    ),
];

macro_rules! model_leaf {
    ($family:literal, $label:literal, $file:literal, $models:expr) => {
        ModelLeaf {
            family: $family,
            models: $models,
            label: $label,
            path: concat!("models/", $file),
            contents: include_str!(concat!("models/", $file)),
        }
    };
}

/// Per-checkpoint quirk leaves. A model appears in at most one leaf.
pub const MODEL_LEAVES: &[ModelLeaf] = &[
    model_leaf!(
        "flux",
        "FLUX.1 Schnell",
        "flux-schnell.md",
        &["flux-schnell"]
    ),
    model_leaf!(
        "flux2",
        "FLUX.2 Klein base (undistilled)",
        "flux2-klein-base.md",
        &["flux2-klein-base", "flux2-klein-base-9b"]
    ),
    model_leaf!("sdxl", "SDXL Turbo", "sdxl-turbo.md", &["sdxl-turbo"]),
    model_leaf!(
        "sdxl",
        "Pony Diffusion V6 XL",
        "pony-v6.md",
        &["pony-v6", "cyberrealistic-pony"]
    ),
    model_leaf!(
        "sdxl",
        "Playground v2.5",
        "playground-v2.5.md",
        &["playground-v2.5"]
    ),
    model_leaf!(
        "sd3",
        "SD 3.5 Large Turbo",
        "sd3.5-large-turbo.md",
        &["sd3.5-large-turbo"]
    ),
    model_leaf!(
        "qwen-image",
        "Qwen-Image few-step distills",
        "qwen-image-flash.md",
        &[
            "qwen-image-flash",
            "qwen-image-distill",
            "qwen-image-lightning"
        ]
    ),
    model_leaf!(
        "qwen-image-edit",
        "Qwen-Image-Edit Lightning",
        "qwen-image-edit-lightning.md",
        &["qwen-image-edit-lightning"]
    ),
    model_leaf!(
        "ltx2",
        "LTX-2.5",
        "ltx-2.5.md",
        &["ltx-2.5-22b-dev", "ltx-2.5-22b-distilled"]
    ),
    model_leaf!(
        "wan",
        "Wan 2.2 TI2V-5B",
        "wan22-ti2v-5b.md",
        &["wan22-ti2v-5b"]
    ),
];

/// A resolved route: shared practice, one family base, at most one task
/// leaf, at most one model leaf.
#[derive(Clone, Copy, Debug)]
pub struct PromptingRoute {
    pub family: &'static FamilyGuide,
    pub task: Option<&'static TaskLeaf>,
    pub model: Option<&'static ModelLeaf>,
}

/// Why a route could not be resolved.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RouteError {
    UnknownFamily(String),
    LeafFamilyMismatch { leaf: String, family: String },
    LeafConflictsWithIdentity { leaf: String, model: String },
}

impl std::fmt::Display for RouteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownFamily(family) => {
                write!(
                    f,
                    "no canonical prompting guide for manifest family {family}"
                )
            }
            Self::LeafFamilyMismatch { leaf, family } => {
                write!(f, "{leaf} task guide cannot route family {family}")
            }
            Self::LeafConflictsWithIdentity { leaf, model } => {
                write!(f, "{leaf} task guide conflicts with model identity {model}")
            }
        }
    }
}

impl std::error::Error for RouteError {}

/// Normalise a family spelling to the manifest family id, or `None` when no
/// guide exists for it.
pub fn family_guide(family: &str) -> Option<&'static FamilyGuide> {
    let canonical = canonical_family(&family.trim().to_ascii_lowercase()).to_string();
    FAMILY_GUIDES.iter().find(|guide| {
        guide.family == canonical || guide.aliases.iter().any(|alias| *alias == canonical)
    })
}

/// Whether the family has a prompting guide.
pub fn is_known_family(family: &str) -> bool {
    family_guide(family).is_some()
}

fn identity_leaf(family: &str, model: &str) -> Option<&'static TaskLeaf> {
    TASK_LEAVES.iter().find(|leaf| {
        leaf.family == family && leaf.identity.is_specific() && leaf.identity.matches(model)
    })
}

fn task_leaf(
    family: &str,
    model: &str,
    task: ExpandTask,
    hints: RouteHints,
) -> Option<&'static TaskLeaf> {
    let candidates = || {
        TASK_LEAVES.iter().filter(|leaf| {
            leaf.family == family
                && leaf.tasks.contains(&task)
                && (!leaf.needs_source_video || hints.source_video)
        })
    };
    candidates()
        .find(|leaf| leaf.identity.matches(model))
        .or_else(|| candidates().next())
}

fn model_leaf(family: &str, model: &str) -> Option<&'static ModelLeaf> {
    let base = crate::manifest::model_base_name(model);
    MODEL_LEAVES
        .iter()
        .find(|leaf| leaf.family == family && leaf.models.contains(&base))
}

/// Resolve the route for a family, an optional exact model identity, and an
/// optional expansion task.
///
/// A supplied task wins over the identity binding (a TI2V checkpoint asked
/// for text-to-video reads the text-to-video leaf). Without a task, the
/// identity binding decides, which is what the skill bundle documents.
pub fn route(
    family: &str,
    model: Option<&str>,
    task: Option<ExpandTask>,
) -> Result<PromptingRoute, RouteError> {
    route_with_hints(family, model, task, RouteHints::default())
}

/// [`route`] with request facts that refine leaf selection: an LTX-2
/// audio-driven expansion reads the Dub-It leaf only when a source video is
/// attached.
pub fn route_with_hints(
    family: &str,
    model: Option<&str>,
    task: Option<ExpandTask>,
    hints: RouteHints,
) -> Result<PromptingRoute, RouteError> {
    let family_guide =
        family_guide(family).ok_or_else(|| RouteError::UnknownFamily(family.to_string()))?;
    let model = model.unwrap_or("");
    let leaf = match task {
        Some(task) => task_leaf(family_guide.family, model, task, hints),
        None => identity_leaf(family_guide.family, model),
    };
    Ok(PromptingRoute {
        family: family_guide,
        task: leaf,
        model: model_leaf(family_guide.family, model),
    })
}

/// Resolve a route with an explicitly selected task leaf, validating that the
/// leaf belongs to the family and does not contradict the model identity.
pub fn route_with_leaf(
    family: &str,
    model: &str,
    leaf_path: &str,
) -> Result<PromptingRoute, RouteError> {
    let mut resolved = route(family, Some(model), None)?;
    let leaf = TASK_LEAVES
        .iter()
        .find(|leaf| leaf.path == leaf_path)
        .ok_or_else(|| RouteError::LeafFamilyMismatch {
            leaf: leaf_path.to_string(),
            family: family.to_string(),
        })?;
    if leaf.family != resolved.family.family {
        return Err(RouteError::LeafFamilyMismatch {
            leaf: leaf.path.to_string(),
            family: family.to_string(),
        });
    }
    if let Some(identity) = identity_leaf(resolved.family.family, model) {
        if identity.path != leaf.path {
            return Err(RouteError::LeafConflictsWithIdentity {
                leaf: leaf.path.to_string(),
                model: model.to_string(),
            });
        }
    }
    resolved.task = Some(leaf);
    Ok(resolved)
}

impl PromptingRoute {
    /// Corpus-relative paths in read order.
    pub fn paths(&self) -> Vec<&'static str> {
        let mut paths = vec![SHARED_PATH, self.family.path];
        if let Some(leaf) = self.task {
            paths.push(leaf.path);
        }
        if let Some(leaf) = self.model {
            paths.push(leaf.path);
        }
        paths
    }

    /// Effective word budget: the task leaf's override, else the family's.
    pub fn word_limit(&self) -> u32 {
        self.task
            .and_then(|leaf| leaf.word_limit)
            .unwrap_or(self.family.word_limit)
    }

    /// The text handed to the expansion LLM: every routed file with the
    /// agent-only sections and shell examples removed.
    pub fn expansion_excerpt(&self) -> String {
        self.expansion_excerpt_with_limit(self.word_limit())
    }

    /// [`Self::expansion_excerpt`] with an explicit word budget (a user
    /// override from `expand.families.<family>.word_limit`).
    pub fn expansion_excerpt_with_limit(&self, word_limit: u32) -> String {
        let mut parts = vec![excerpt(SHARED_MD, word_limit)];
        if !self.task.is_some_and(|leaf| leaf.standalone) {
            parts.push(excerpt(self.family.contents, word_limit));
        }
        if let Some(leaf) = self.task {
            parts.push(excerpt(leaf.contents, word_limit));
        }
        if let Some(leaf) = self.model {
            parts.push(excerpt(leaf.contents, word_limit));
        }
        parts
            .into_iter()
            .filter(|part| !part.trim().is_empty())
            .collect::<Vec<_>>()
            .join("\n\n")
    }
}

/// Substitute the corpus placeholders for a rendered (agent-facing) file.
pub fn substitute(contents: &str, word_limit: u32) -> String {
    contents.replace("{{word_limit}}", &word_limit.to_string())
}

/// Word budget used when rendering a corpus file outside a route (the skill
/// bundle renders each file once, so a leaf uses its own override and a
/// family its own default).
pub fn file_word_limit(path: &str) -> u32 {
    if let Some(leaf) = TASK_LEAVES.iter().find(|leaf| leaf.path == path) {
        if let Some(limit) = leaf.word_limit {
            return limit;
        }
        return family_guide(leaf.family).map_or(150, |guide| guide.word_limit);
    }
    if let Some(leaf) = MODEL_LEAVES.iter().find(|leaf| leaf.path == path) {
        return family_guide(leaf.family).map_or(150, |guide| guide.word_limit);
    }
    FAMILY_GUIDES
        .iter()
        .find(|guide| guide.path == path)
        .map_or(150, |guide| guide.word_limit)
}

/// Every corpus file as `(corpus-relative path, rendered contents)`, in
/// `SKILL.md` order: shared, families, task leaves, model leaves.
pub fn rendered_files() -> Vec<(&'static str, String)> {
    let mut files = vec![(SHARED_PATH, substitute(SHARED_MD, 150))];
    for guide in FAMILY_GUIDES {
        files.push((guide.path, substitute(guide.contents, guide.word_limit)));
    }
    for leaf in TASK_LEAVES {
        files.push((
            leaf.path,
            substitute(leaf.contents, file_word_limit(leaf.path)),
        ));
    }
    for leaf in MODEL_LEAVES {
        files.push((
            leaf.path,
            substitute(leaf.contents, file_word_limit(leaf.path)),
        ));
    }
    files
}

/// Reduce one corpus file to its expansion excerpt: drop agent-only H2
/// sections, drop fenced `bash` blocks, keep every other fence (H3's
/// Context-IR sample is a `text` fence), drop the `Manifest family:` line and
/// the top-level title, then substitute placeholders.
pub fn excerpt(markdown: &str, word_limit: u32) -> String {
    let mut out = String::new();
    let mut skipping_section = false;
    let mut fence: Option<(String, bool)> = None;
    for line in markdown.lines() {
        if let Some((marker, dropped)) = &fence {
            let closes = line.trim_start().starts_with(marker.as_str());
            if !dropped {
                out.push_str(line);
                out.push('\n');
            }
            if closes {
                fence = None;
            }
            continue;
        }
        let trimmed = line.trim_start();
        if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
            let marker = trimmed[..3].to_string();
            let info = trimmed[3..].trim().to_ascii_lowercase();
            let dropped = skipping_section
                || info == "bash"
                || info == "sh"
                || info == "shell"
                || info == "console";
            if !dropped {
                out.push_str(line);
                out.push('\n');
            }
            fence = Some((marker, dropped));
            continue;
        }
        if let Some(title) = trimmed.strip_prefix("## ") {
            let title = title.trim();
            skipping_section = AGENT_ONLY_SECTIONS
                .iter()
                .any(|agent_only| title.eq_ignore_ascii_case(agent_only));
            if skipping_section {
                continue;
            }
        } else if trimmed.starts_with("# ") {
            continue;
        }
        if skipping_section {
            continue;
        }
        if trimmed.starts_with("Manifest family:") {
            continue;
        }
        out.push_str(line);
        out.push('\n');
    }
    let collapsed = collapse_blank_lines(&out);
    substitute(collapsed.trim(), word_limit)
}

/// Render ONE H2 section of a corpus file the way [`excerpt`] renders a whole
/// file: the section body without its heading, shell fences dropped, and the
/// placeholders substituted. `None` when the file has no such section (title
/// match is case-insensitive) or the body is empty.
///
/// This is how a surface quotes a single piece of a guide — a family that
/// reads no prompt answers `mold expand` with its `Generation context`
/// section — without a second copy of that prose living in Rust.
pub fn section_excerpt(markdown: &str, title: &str, word_limit: u32) -> Option<String> {
    let mut body = String::new();
    let mut inside = false;
    for line in markdown.lines() {
        if let Some(heading) = line.trim_start().strip_prefix("## ") {
            if inside {
                break;
            }
            inside = heading.trim().eq_ignore_ascii_case(title.trim());
            continue;
        }
        if inside {
            body.push_str(line);
            body.push('\n');
        }
    }
    let rendered = excerpt(&body, word_limit);
    (!rendered.is_empty()).then_some(rendered)
}

fn collapse_blank_lines(text: &str) -> String {
    let mut out = String::new();
    let mut blank = false;
    for line in text.lines() {
        if line.trim().is_empty() {
            if !blank {
                out.push('\n');
            }
            blank = true;
        } else {
            out.push_str(line);
            out.push('\n');
            blank = false;
        }
    }
    out
}

/// MiniMax H3 reference labels in presentation order. Mirrors the
/// conditioner's own counter (`mold-candle` `minimax_h3::presentation`):
/// independent `<Picture n>`, `<Video n>`, and `<Audio n>` counters in the
/// supplied order, and a video that carries a soundtrack takes the next
/// `<Audio n>` label before its own `<Video n>` label.
pub fn h3_reference_labels(references: &[ExpandReference]) -> Vec<String> {
    let (mut picture, mut video, mut audio) = (0u32, 0u32, 0u32);
    let mut labels = Vec::new();
    for reference in references {
        if reference.has_audio && reference.kind == GenerationReferenceKind::Video {
            audio += 1;
            labels.push(format!("<Audio {audio}>"));
        }
        match reference.kind {
            GenerationReferenceKind::Image => {
                picture += 1;
                labels.push(format!("<Picture {picture}>"));
            }
            GenerationReferenceKind::Video => {
                video += 1;
                labels.push(format!("<Video {video}>"));
            }
            GenerationReferenceKind::Audio => {
                audio += 1;
                labels.push(format!("<Audio {audio}>"));
            }
        }
    }
    labels
}

fn role_phrase(role: Option<ExpandReferenceRole>) -> &'static str {
    match role {
        Some(ExpandReferenceRole::FirstFrame) => "the opening frame",
        Some(ExpandReferenceRole::LastFrame) => "the closing frame",
        Some(ExpandReferenceRole::Keyframe) => "a keyframe anchor",
        Some(ExpandReferenceRole::Source) => "the source to transform",
        Some(ExpandReferenceRole::Identity) => "a face identity reference",
        Some(ExpandReferenceRole::Edit) => "an edit reference",
        Some(ExpandReferenceRole::Reference) | None => "a reference",
    }
}

/// Render the generation facts as plain lines for the LLM. Labels follow the
/// family's own addressing grammar: H3 uses `<Picture n>` labels, FLUX.2 and
/// Qwen-Image-Edit use ordinals ("image 1"), everything else names the role.
pub fn render_generation_context(
    family: &str,
    task: ExpandTask,
    context: &ExpandContext,
) -> String {
    let family = family_guide(family).map_or(family, |guide| guide.family);
    let mut lines = Vec::new();
    let output = match task {
        ExpandTask::TextToImage => "image",
        ExpandTask::TextToAudio => "audio",
        _ => "video",
    };
    let mut target = format!("Target: {output}");
    if let Some(model) = context
        .model
        .as_deref()
        .filter(|model| !model.trim().is_empty())
    {
        target.push_str(&format!(" on {model}"));
    }
    if let (Some(width), Some(height)) = (context.width, context.height) {
        let orientation = if width > height {
            "landscape"
        } else if height > width {
            "portrait"
        } else {
            "square"
        };
        target.push_str(&format!(", {width}x{height} {orientation}"));
    }
    if output != "image" {
        if let Some(frames) = context.frames {
            if frames <= 1 {
                target.push_str(", a single still frame");
            } else if let Some(fps) = context.fps.filter(|fps| *fps > 0) {
                let seconds = f64::from(frames) / f64::from(fps);
                target.push_str(&format!(", {frames} frames at {fps} fps ({seconds:.1} s)"));
            } else {
                target.push_str(&format!(", {frames} frames"));
            }
        }
        if let Some(clip) = context
            .clip_frames
            .filter(|clip| *clip > 0 && context.frames.is_some_and(|frames| frames > *clip))
        {
            let clips = context
                .frames
                .map(|frames| frames.div_ceil(clip))
                .unwrap_or(1);
            target.push_str(&format!(
                "; rendered as {clips} chained clips of {clip} frames, so describe one continuous shot whose motion can carry across seams"
            ));
        }
        if context.audio == Some(true) {
            target.push_str("; audio is generated with the video");
        } else if context.audio == Some(false) {
            target.push_str("; silent, no audio track");
        }
    }
    target.push('.');
    lines.push(target);
    if !context.references.is_empty() {
        let labels = if family == "minimax-h3" {
            h3_reference_labels(&context.references)
        } else {
            Vec::new()
        };
        let mut described = Vec::new();
        let mut label_index = 0usize;
        for (index, reference) in context.references.iter().enumerate() {
            let kind = match reference.kind {
                GenerationReferenceKind::Image => "image",
                GenerationReferenceKind::Video => "video",
                GenerationReferenceKind::Audio => "audio",
            };
            let role = role_phrase(reference.role);
            if family == "minimax-h3" {
                let mut own = Vec::new();
                if reference.has_audio && reference.kind == GenerationReferenceKind::Video {
                    own.push(labels[label_index].clone());
                    label_index += 1;
                }
                own.push(labels[label_index].clone());
                label_index += 1;
                described.push(format!("{} = {kind}, {role}", own.join(" and ")));
            } else if matches!(family, "flux2" | "qwen-image-edit") {
                described.push(format!("image {} = {role}", index + 1));
            } else {
                described.push(format!("{kind} {} = {role}", index + 1));
            }
        }
        lines.push(format!(
            "References, in order: {}. Refer to them with exactly these names and never describe their pixels from imagination.",
            described.join("; ")
        ));
    }
    match context.prompt_mode {
        Some(crate::generation_profile::PromptRequirement::Ignored) => lines.push(
            "Prompt: not read by this model; the attached image is the whole conditioning, so there is nothing for the text to change."
                .to_string(),
        ),
        Some(crate::generation_profile::PromptRequirement::Optional) => lines.push(
            "Prompt: optional; the attached media conditions the render and the text refines it."
                .to_string(),
        ),
        Some(crate::generation_profile::PromptRequirement::Required) | None => {}
    }
    match context.negative_prompt_supported {
        Some(true) => lines.push(
            "Negative prompt: supported; keep unwanted traits out of the positive prompt."
                .to_string(),
        ),
        Some(false) => lines.push(
            "Negative prompt: not honoured by this model; avoid things only by describing what should be there instead."
                .to_string(),
        ),
        None => {}
    }
    if !context.loras.is_empty() {
        lines.push(format!(
            "LoRA adapters active: {}. Keep any trigger words the user wrote.",
            context.loras.join(", ")
        ));
    }
    lines.join("\n")
}

/// Count words the way the budget tests do.
pub fn word_count(text: &str) -> usize {
    text.split_whitespace().count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    fn generation_manifests() -> Vec<&'static crate::manifest::ModelManifest> {
        crate::manifest::known_manifests()
            .iter()
            .filter(|manifest| manifest.is_generation_model() || manifest.is_upscaler())
            .collect()
    }

    #[test]
    fn every_manifest_family_has_exactly_one_base_guide() {
        let manifest_families = generation_manifests()
            .iter()
            .map(|manifest| manifest.family.as_str())
            .collect::<BTreeSet<_>>();
        let documented = FAMILY_GUIDES
            .iter()
            .map(|guide| guide.family)
            .collect::<BTreeSet<_>>();
        assert_eq!(manifest_families, documented);
        assert_eq!(
            FAMILY_GUIDES.len(),
            documented.len(),
            "duplicate family base"
        );
    }

    #[test]
    fn every_manifest_routes_to_shared_one_base_and_the_expected_leaves() {
        for manifest in generation_manifests() {
            let route = route(&manifest.family, Some(&manifest.name), None).unwrap();
            let paths = route.paths();
            assert_eq!(paths[0], SHARED_PATH, "{}", manifest.name);
            assert_eq!(
                paths
                    .iter()
                    .filter(|path| path.starts_with("families/"))
                    .count(),
                1,
                "{}",
                manifest.name
            );
            let leaves = route.task.iter().count();
            match manifest.family.as_str() {
                "minimax-h3" | "wan" => assert_eq!(leaves, 1, "{}", manifest.name),
                _ => assert_eq!(leaves, 0, "{}", manifest.name),
            }
        }
    }

    #[test]
    fn section_excerpt_renders_exactly_one_titled_section() {
        let guide = family_guide("hunyuan3d").unwrap();
        let section =
            section_excerpt(guide.contents, "Generation context", guide.word_limit).unwrap();
        assert!(
            section.starts_with("Three properties of the image"),
            "{section}"
        );
        assert!(section.contains("three-quarter"), "{section}");
        assert!(!section.contains("## "), "{section}");
        assert!(!section.contains("Write no prompt"), "{section}");
        assert_eq!(
            section_excerpt(guide.contents, "generation CONTEXT", guide.word_limit).as_deref(),
            Some(section.as_str()),
            "title match is case-insensitive"
        );
        assert!(section_excerpt(guide.contents, "No such section", 40).is_none());
        // An agent-only section asked for by name is rendered without its
        // shell fences, the same way the excerpt treats every other file.
        let cli = section_excerpt(guide.contents, "CLI", guide.word_limit).unwrap();
        assert!(!cli.contains("```"), "{cli}");
        assert!(cli.contains("--octree"), "{cli}");
    }

    #[test]
    fn identity_and_task_routing_pick_the_documented_leaves() {
        let leaf = |family: &str, model: &str, task: Option<ExpandTask>| {
            route(family, Some(model), task)
                .unwrap()
                .task
                .map(|leaf| leaf.path)
        };
        assert_eq!(
            leaf("minimax-h3", "minimax-h3-fl2va:q8", None),
            Some("minimax-h3/base-modes.md")
        );
        assert_eq!(
            leaf("minimax-h3", "minimax-h3-ref2va:q8", None),
            Some("minimax-h3/ref2va.md")
        );
        assert_eq!(
            leaf("wan", "wan22-t2v-a14b:q4", None),
            Some("wan/text-to-video.md")
        );
        assert_eq!(
            leaf("wan", "wan22-ti2v-5b:q8", None),
            Some("wan/image-conditioned.md")
        );
        assert_eq!(
            leaf("wan", "wan22-ti2v-5b:q8", Some(ExpandTask::TextToVideo)),
            Some("wan/text-to-video.md"),
            "a supplied task wins over the identity binding"
        );
        assert_eq!(
            leaf("wan", "wan22-t2v-a14b:q4", Some(ExpandTask::ImageToVideo)),
            Some("wan/image-conditioned.md")
        );
        assert_eq!(
            leaf("ltx2", "ltx-2.5-22b:q6", Some(ExpandTask::TextToAudio)),
            Some("ltx2/text-to-audio.md")
        );
        assert_eq!(leaf("ltx2", "ltx-2.5-22b:q6", None), None);
        assert_eq!(
            leaf("ltx2", "ltx-2.5-22b:q6", Some(ExpandTask::TextToVideo)),
            None
        );
        assert_eq!(
            leaf("ltx2", "ltx-2.5-22b:q6", Some(ExpandTask::AudioDrivenVideo)),
            None,
            "audio-to-video without a clip is not a dub"
        );
        assert_eq!(
            route_with_hints(
                "ltx2",
                Some("ltx-2.5-22b:q6"),
                Some(ExpandTask::AudioDrivenVideo),
                RouteHints { source_video: true },
            )
            .unwrap()
            .task
            .map(|leaf| leaf.path),
            Some("ltx2/dub-it.md"),
            "lip dub re-voices an attached clip through the Dub-It leaf"
        );
        assert_eq!(
            route_with_leaf("ltx2", "ltx-2.5-22b:q6", "ltx2/dub-it.md")
                .unwrap()
                .task
                .map(|leaf| leaf.path),
            Some("ltx2/dub-it.md")
        );
        assert!(route_with_leaf("flux", "flux-dev:q8", "ltx2/dub-it.md").is_err());
        assert!(route_with_leaf(
            "minimax-h3",
            "minimax-h3-ref2va:q8",
            "minimax-h3/base-modes.md"
        )
        .is_err());
        assert!(route("not-a-family", None, None).is_err());
    }

    #[test]
    fn family_lookup_accepts_wire_aliases() {
        for (alias, family) in [
            ("ltx-2", "ltx2"),
            ("LTX2", "ltx2"),
            ("flux.2", "flux2"),
            ("minimax_h3", "minimax-h3"),
            ("minimaxh3", "minimax-h3"),
            ("sd1.5", "sd15"),
            ("stable-diffusion-1.5", "sd15"),
            ("wan2.2", "wan"),
            ("wuerstchen-v2", "wuerstchen"),
            (" sdxl ", "sdxl"),
        ] {
            assert_eq!(
                family_guide(alias).map(|guide| guide.family),
                Some(family),
                "{alias}"
            );
        }
        assert!(family_guide("hailuo").is_none());
    }

    #[test]
    fn every_task_leaf_and_model_leaf_belongs_to_a_documented_family() {
        for leaf in TASK_LEAVES {
            assert!(family_guide(leaf.family).is_some(), "{}", leaf.path);
            assert!(
                leaf.path.starts_with(&format!("{}/", leaf.family)),
                "{}",
                leaf.path
            );
        }
        let mut seen = BTreeSet::new();
        for leaf in MODEL_LEAVES {
            assert!(family_guide(leaf.family).is_some(), "{}", leaf.path);
            for model in leaf.models {
                assert!(seen.insert(*model), "{model} appears in two model leaves");
                let manifest = crate::manifest::known_manifests()
                    .iter()
                    .find(|manifest| crate::manifest::model_base_name(&manifest.name) == *model)
                    .unwrap_or_else(|| panic!("{model} is not a known manifest"));
                assert_eq!(manifest.family, leaf.family, "{model}");
            }
        }
    }

    #[test]
    fn rendered_files_carry_no_placeholders_and_cover_every_registry_entry() {
        let files = rendered_files();
        let paths = files.iter().map(|(path, _)| *path).collect::<BTreeSet<_>>();
        assert_eq!(
            files.len(),
            1 + FAMILY_GUIDES.len() + TASK_LEAVES.len() + MODEL_LEAVES.len()
        );
        assert_eq!(paths.len(), files.len(), "duplicate corpus path");
        for (path, contents) in &files {
            assert!(
                !contents.contains("{{"),
                "{path} has an unrendered placeholder"
            );
            assert!(!path.contains(':'), "identity tag leaked into {path}");
            if path.starts_with("models/") {
                // A model leaf is named after the checkpoint base name, which
                // may legitimately be `sdxl-turbo`; only the `:tag` is banned.
                continue;
            }
            for tag in [
                "q4",
                "q5",
                "q6",
                "q8",
                "fp8",
                "fp16",
                "bf16",
                "nvfp4",
                "turbo",
                "lightning",
            ] {
                assert!(
                    !path.to_ascii_lowercase().contains(tag),
                    "identity tier leaked into {path}"
                );
            }
        }
    }

    #[test]
    fn excerpt_drops_agent_only_sections_and_shell_blocks() {
        let guide = "# Demo prompting\n\nManifest family: `demo`.\n\n## Prompt style\n\nKeep it under {{word_limit}} words.\n\n```text\nkeep this sample\n```\n\n## CLI\n\n```bash\nmold run demo \"x\"\n```\n\nagent prose\n\n## Pitfalls\n\n```bash\nmold run demo \"y\"\n```\n\nstill shared\n\n## Sources\n\n- https://example.invalid\n";
        let text = excerpt(guide, 42);
        assert!(text.starts_with("## Prompt style"), "{text}");
        assert!(text.contains("under 42 words"));
        assert!(text.contains("keep this sample"));
        assert!(text.contains("still shared"));
        assert!(!text.contains("mold run"));
        assert!(!text.contains("agent prose"));
        assert!(!text.contains("example.invalid"));
        assert!(!text.contains("Manifest family"));
        assert!(!text.contains("# Demo"));
    }

    #[test]
    fn every_route_excerpt_stays_inside_the_expander_budget() {
        let mut routes = Vec::new();
        for manifest in generation_manifests() {
            routes.push(route(&manifest.family, Some(&manifest.name), None).unwrap());
            for task in [
                ExpandTask::TextToImage,
                ExpandTask::TextToVideo,
                ExpandTask::ImageToVideo,
                ExpandTask::VideoToVideo,
                ExpandTask::Retake,
                ExpandTask::KeyframeInterpolation,
                ExpandTask::AudioDrivenVideo,
                ExpandTask::ReferenceToAudioVideo,
                ExpandTask::TextToAudio,
            ] {
                routes.push(route(&manifest.family, Some(&manifest.name), Some(task)).unwrap());
            }
        }
        for leaf in TASK_LEAVES {
            routes.push(route_with_leaf(leaf.family, "", leaf.path).unwrap());
        }
        for route in routes {
            let text = route.expansion_excerpt();
            let words = word_count(&text);
            assert!(
                words <= EXCERPT_WORD_BUDGET,
                "{:?} excerpt is {words} words (budget {EXCERPT_WORD_BUDGET})",
                route.paths()
            );
            assert!(!text.contains("```bash"), "{:?}", route.paths());
            assert!(!text.contains("{{"), "{:?}", route.paths());
        }
    }

    #[test]
    fn h3_labels_follow_the_conditioner_counter() {
        let refs = vec![
            ExpandReference {
                kind: GenerationReferenceKind::Video,
                has_audio: true,
                role: None,
            },
            ExpandReference::image(ExpandReferenceRole::Reference),
            ExpandReference {
                kind: GenerationReferenceKind::Audio,
                has_audio: false,
                role: None,
            },
        ];
        assert_eq!(
            h3_reference_labels(&refs),
            vec!["<Audio 1>", "<Video 1>", "<Picture 1>", "<Audio 2>"]
        );
        let text = render_generation_context(
            "minimax-h3",
            ExpandTask::ReferenceToAudioVideo,
            &ExpandContext {
                references: refs,
                frames: Some(125),
                fps: Some(24),
                ..ExpandContext::default()
            },
        );
        assert!(text.contains("<Audio 1> and <Video 1> = video"), "{text}");
        assert!(text.contains("<Picture 1> = image"), "{text}");
        assert!(text.contains("<Audio 2> = audio"), "{text}");
        assert!(text.contains("125 frames at 24 fps (5.2 s)"), "{text}");
    }

    #[test]
    fn generation_context_names_canvas_duration_chain_and_ordinals() {
        let text = render_generation_context(
            "wan",
            ExpandTask::ImageToVideo,
            &ExpandContext {
                model: Some("wan22-ti2v-5b:q8".into()),
                width: Some(1280),
                height: Some(704),
                frames: Some(100),
                fps: Some(24),
                clip_frames: Some(49),
                negative_prompt_supported: Some(true),
                audio: Some(false),
                references: vec![ExpandReference::image(ExpandReferenceRole::FirstFrame)],
                loras: vec!["paper-boat".into()],
                prompt_mode: None,
            },
        );
        assert!(
            text.contains("Target: video on wan22-ti2v-5b:q8, 1280x704 landscape, 100 frames at 24 fps (4.2 s); rendered as 3 chained clips of 49 frames"),
            "{text}"
        );
        assert!(text.contains("silent, no audio track"), "{text}");
        assert!(text.contains("image 1 = the opening frame"), "{text}");
        assert!(text.contains("Negative prompt: supported"), "{text}");
        assert!(text.contains("LoRA adapters active: paper-boat"), "{text}");
        let edit = render_generation_context(
            "qwen-image-edit",
            ExpandTask::TextToImage,
            &ExpandContext {
                references: vec![
                    ExpandReference::image(ExpandReferenceRole::Edit),
                    ExpandReference::image(ExpandReferenceRole::Edit),
                ],
                ..ExpandContext::default()
            },
        );
        assert!(
            edit.contains("image 1 = an edit reference; image 2 = an edit reference"),
            "{edit}"
        );
        assert!(edit.starts_with("Target: image."), "{edit}");
    }
}
