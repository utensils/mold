//! Model-family-aware system prompt templates for LLM prompt expansion.
//!
//! Different diffusion models have different prompt styles and token limits.
//! This module provides tailored system prompts for each model family.

use crate::expand::FamilyOverride;
use crate::prompting;
use crate::{expand::remix_dimensions_for_position, ExpandContext, ExpandTask, RemixDimension};

/// Fallback notes for a family that has no guide (a future or catalog-only
/// family). Every manifest family has a guide, so this is a wire-only path.
const GENERIC_MODEL_NOTES: &str =
    "This model uses a large text encoder that understands natural language well. \
Write descriptive, vivid natural language. Include composition, lighting, color palette, \
textures, atmosphere, and camera angle.";

/// Word budget used when a family has no guide.
const GENERIC_WORD_LIMIT: u32 = 150;

/// Return the word limit and the prompting-guide excerpt for a model family.
///
/// The excerpt is the family's route in the prompting corpus
/// ([`crate::prompting`]): shared practice, the family base, and the task or
/// model leaves selected by `task` and `model`.
pub fn family_config(family: &str) -> (u32, String) {
    guide_config(family, None, ExpandTask::for_family(family))
}

fn guide_config(family: &str, model: Option<&str>, task: ExpandTask) -> (u32, String) {
    match prompting::route(family, model, Some(task)) {
        Ok(route) => {
            let limit = route.word_limit();
            (limit, guide_notes(&route, limit))
        }
        Err(_) => (GENERIC_WORD_LIMIT, GENERIC_MODEL_NOTES.to_string()),
    }
}

/// The answer every prompt-transform surface gives for a family whose
/// prompt is [`PromptRequirement::Ignored`]: there is no text encoder, so no
/// language model is called and the "expansion" is the guide's own advice on
/// preparing the image.
///
/// [`PromptRequirement::Ignored`]: crate::generation_profile::PromptRequirement::Ignored
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IgnoredPromptAdvice {
    /// The manifest family that ignores its prompt.
    pub family: &'static str,
    /// The one-line explanation a user sees first.
    pub headline: String,
    /// The family guide's `Generation context` section, rendered from the
    /// corpus exactly as the skill bundle and the website render it.
    pub preparation: String,
}

impl IgnoredPromptAdvice {
    /// Headline and preparation advice as one text, which is what the wire
    /// `expanded` array and the MCP tools carry.
    pub fn text(&self) -> String {
        if self.preparation.is_empty() {
            self.headline.clone()
        } else {
            format!("{}\n\n{}", self.headline, self.preparation)
        }
    }
}

/// The ONE decision behind every expand and remix door: whether `family`
/// reads its prompt at all, answered by
/// [`crate::generation_profile::prompt_requirement_for_family`] (the same
/// rule admission and every generation profile use), and if not, what the
/// user is told instead of an LLM rewrite.
///
/// A conditioned request is the case that can differ, and it is the only
/// case where an `Ignored` answer exists, so the rule is asked for one. The
/// CLI, `/api/expand`, `/api/remix`, the MCP tools, the TUI, and the shared
/// expansion driver all ask this function and nothing else, so a family
/// that ignores its prompt is never handed to a language model anywhere.
pub fn ignored_prompt_advice(family: &str) -> Option<IgnoredPromptAdvice> {
    // The guide registry normalises spelling and aliases; the rule is then
    // asked about the manifest family id, exactly as admission asks it.
    let route = prompting::route(family, None, None).ok()?;
    let family = route.family.family;
    if !crate::generation_profile::prompt_requirement_for_family(Some(family), true).is_ignored() {
        return None;
    }
    let preparation = prompting::section_excerpt(
        route.family.contents,
        "Generation context",
        route.word_limit(),
    )
    .unwrap_or_default();
    Some(IgnoredPromptAdvice {
        family,
        headline: format!(
            "{family} reads no prompt: the image is the whole conditioning, so there is nothing to expand or remix. Prepare the image instead."
        ),
        preparation,
    })
}

fn guide_notes(route: &prompting::PromptingRoute, word_limit: u32) -> String {
    format!(
        "MODEL PROMPTING GUIDE ({} family):\n{}",
        route.family.family,
        route.expansion_excerpt_with_limit(word_limit)
    )
}

const SINGLE_SYSTEM_TEMPLATE: &str = "\
You are an image generation prompt writer. Expand the user's brief description into a detailed, vivid image prompt.

Rules:
1. PRESERVE the user's core subject and intent exactly.
2. ADD: composition, lighting, color palette, textures, atmosphere, camera angle.
3. Use comma-separated descriptive phrases.
4. Keep under {WORD_LIMIT} words.
5. Output ONLY the expanded prompt, nothing else. No preamble, no explanation.

{MODEL_NOTES}";

const BATCH_SYSTEM_TEMPLATE: &str = "\
You are an image generation prompt writer. Generate {N} distinct image prompts based on the user's concept.

Each prompt must:
- Keep the core concept but explore a DIFFERENT angle
- Vary at least 2 of: time of day, weather, camera angle, color palette, mood, artistic style
- Be self-contained and under {WORD_LIMIT} words
- Use comma-separated descriptive phrases

Output as a JSON array of {N} strings, nothing else. No preamble, no explanation.
Example format: [\"prompt one\", \"prompt two\"]

{MODEL_NOTES}";

const TEXT_TO_VIDEO_SYSTEM_TEMPLATE: &str = "\
You are a video generation prompt writer. Expand the user's brief direction into a literal, chronological shot description.

Rules:
1. PRESERVE the user's subject, action, and intent exactly.
2. Describe the principal action and ordered motion in chronological order.
3. State camera behavior, environment response, and continuity constraints. Keep the camera stable unless the user requests movement.
4. Describe visible motion rather than abstract themes; do not invent dialogue or scene changes.
5. Keep under {WORD_LIMIT} words.
6. Output ONLY the expanded prompt, nothing else. No preamble, no explanation.

{MODEL_NOTES}";

const IMAGE_TO_VIDEO_SYSTEM_TEMPLATE: &str = "\
You are a video generation prompt writer expanding motion from an attached opening frame. The source image is the visual authority.

Rules:
1. PRESERVE the user's requested action and intent exactly.
2. Expand only the requested temporal change: ordered subject motion, environment response, and requested camera behavior.
3. Do not re-invent or restate visible identity, composition, clothing, setting, lighting, or style. Inaccurate restatement can compete with the frame.
4. Do not add scene cuts or camera movement unless the user asks for them.
5. Keep continuity with the opening frame and stay under {WORD_LIMIT} words.
6. Output ONLY the expanded prompt, nothing else. No preamble, no explanation.

{MODEL_NOTES}";

const VIDEO_TO_VIDEO_SYSTEM_TEMPLATE: &str = "\
You are a video generation prompt writer directing a transformation or continuation of an attached source video. The source video is the visual and temporal authority.

Rules:
1. PRESERVE existing subject identity, layout, motion direction, timing, camera path, and continuity unless the user explicitly changes one.
2. Expand only the requested transformation or next chronological action.
3. Avoid a competing scene description, new cut, or unrelated camera move.
4. Keep under {WORD_LIMIT} words and output ONLY the expanded prompt.

{MODEL_NOTES}";

const RETAKE_SYSTEM_TEMPLATE: &str = "\
You are a video generation prompt writer directing a bounded retake inside an existing clip. The surrounding source video is authoritative continuity.

Rules:
1. Expand only the requested correction or action inside the selected range.
2. Preserve subject identity, framing, lighting, camera path, motion direction, and entry/exit continuity.
3. Do not rewrite the whole scene or introduce a cut.
4. Keep under {WORD_LIMIT} words and output ONLY the expanded prompt.

{MODEL_NOTES}";

const KEYFRAME_SYSTEM_TEMPLATE: &str = "\
You are a video generation prompt writer directing motion between attached keyframes. The keyframes are fixed visual anchors.

Rules:
1. Expand only the ordered connective motion and requested camera behavior between anchors.
2. Preserve identity, composition, and continuity established by every keyframe.
3. Do not replace, contradict, or independently redesign an anchor.
4. Keep under {WORD_LIMIT} words and output ONLY the expanded prompt.

{MODEL_NOTES}";

const AUDIO_DRIVEN_VIDEO_SYSTEM_TEMPLATE: &str = "\
You are a video generation prompt writer directing visuals from attached conditioning audio. The audio timing and events are authoritative.

Rules:
1. Expand the user's visual intent into chronological action synchronized to the supplied audio.
2. Preserve the audio's timing, cadence, speech, music, and event order; do not invent competing dialogue or sounds.
3. When a source clip is attached, preserve its identity, framing, camera path, motion, and timing unless the user explicitly requests a change.
4. State only the requested subject motion, environment response, and continuity without changing the soundtrack or an authoritative source clip.
5. Keep under {WORD_LIMIT} words and output ONLY the expanded prompt.

{MODEL_NOTES}";

const REFERENCE_TO_AUDIO_VIDEO_SYSTEM_TEMPLATE: &str = "\
You are an audio-video generation prompt writer directing semantic resynthesis from ordered image, video, and audio references.

Rules:
1. PRESERVE referenced identities, relationships, chronology, motion cues, and audible events unless the user explicitly requests a change.
2. Expand the requested target action as one synchronized chronological shot; do not describe this as a pixel-aligned edit or denoise operation.
3. Respect reference order and do not merge, omit, reorder, or invent referenced people, objects, voices, music, or events.
4. Describe only target composition, action, camera behavior, environment response, and synchronized sound needed by the request.
5. Keep under {WORD_LIMIT} words and output ONLY the expanded prompt.

{MODEL_NOTES}";

const TEXT_TO_AUDIO_SYSTEM_TEMPLATE: &str = "\
You are an audio generation prompt writer. Expand the user's brief into a chronological sound direction.

Rules:
1. PRESERVE the requested sound, speech, music, mood, and intent exactly.
2. Describe ordered audible events, timing, space, intensity, and continuity.
3. Do not add visual composition or camera language.
4. Keep under {WORD_LIMIT} words and output ONLY the expanded prompt.

{MODEL_NOTES}";

const VIDEO_BATCH_SYSTEM_TEMPLATE: &str = "\
You are a video generation prompt writer. Generate {N} distinct motion directions based on the user's concept.

Each prompt must preserve the subject and core intent, read as a literal chronological shot, and vary controlled motion, camera behavior, environment response, or pacing without inventing unrelated scenes. Keep each under {WORD_LIMIT} words.

Output as a JSON array of {N} strings, nothing else. No preamble, no explanation.

{TASK_NOTES}

{MODEL_NOTES}";

const CONDITIONED_VIDEO_BATCH_SYSTEM_TEMPLATE: &str = "\
You are a video generation prompt writer. Generate {N} distinct directions for the same conditioned video task.

Every prompt must obey the source authority below. Vary only a requested action's timing, pacing, or degree when the user's intent leaves room; never vary source identity, composition, camera behavior, timing anchors, or audio events unless the user explicitly requests that change. Keep each under {WORD_LIMIT} words.

Output as a JSON array of {N} strings, nothing else. No preamble, no explanation.

{TASK_NOTES}

{MODEL_NOTES}";

const AUDIO_BATCH_SYSTEM_TEMPLATE: &str = "\
You are an audio generation prompt writer. Generate {N} distinct chronological sound directions based on the user's concept.

Vary audible timing, space, intensity, or arrangement while preserving the requested sound, speech, music, mood, and intent. Do not use visual composition or camera language. Keep each under {WORD_LIMIT} words.

Output as a JSON array of {N} strings, nothing else. No preamble, no explanation.

{TASK_NOTES}

{MODEL_NOTES}";

const REMIX_SYSTEM_TEMPLATE: &str = "\
You are a subject-preserving prompt remix editor. Generate {N} intentionally varied prompt alternatives from the user's source prompt.

Non-negotiable rules:
1. Preserve the central subject, identity, requested action, named entities, quantities, relationships, and every explicit user constraint.
2. Do not replace the concept with a new scene and do not merely swap synonyms.
3. For each numbered variation, vary ONLY its assigned creative dimension. Preserve all unassigned dimensions.
4. Obey the conditioning authority in TASK NOTES; attached media is not present here but remains authoritative for generation.
5. Keep each prompt self-contained and under {WORD_LIMIT} words.
6. Output a JSON array of exactly {N} strings, in the assigned order, with no preamble or explanation.

ASSIGNED DIMENSIONS:
{DIMENSION_PLAN}

{TASK_NOTES}

{MODEL_NOTES}";

fn task_notes(task: ExpandTask) -> &'static str {
    match task {
        ExpandTask::TextToImage => "",
        ExpandTask::TextToVideo => {
            "The prompts are text-to-video shots: establish the scene, then describe ordered motion and continuity."
        }
        ExpandTask::ImageToVideo => {
            "The source image is the visual authority. Describe only temporal changes; do not re-invent visible identity or composition."
        }
        ExpandTask::VideoToVideo => {
            "The source video is the visual and temporal authority. Preserve its identity, camera path, timing, and continuity."
        }
        ExpandTask::Retake => {
            "The surrounding source video is authoritative. Change only the selected retake range and preserve its entry and exit continuity."
        }
        ExpandTask::KeyframeInterpolation => {
            "The keyframes are fixed visual anchors. Generate the requested number of distinct motion directions between them without redesigning any anchor."
        }
        ExpandTask::AudioDrivenVideo => {
            "The attached audio is authoritative. Synchronize distinct visual directions to its existing timing and events without inventing competing audio. When a source clip is attached, preserve its identity, framing, camera path, motion, and timing unless explicitly changed."
        }
        ExpandTask::ReferenceToAudioVideo => {
            "The ordered heterogeneous references are semantic audio-video authority. Preserve their identities, relationships, event order, motion cues, and sound cues while describing the requested new synchronized shot; this is resynthesis, not pixel-aligned editing."
        }
        ExpandTask::TextToAudio => {
            "These are audio-only directions. Vary audible timing, space, intensity, or arrangement without visual or camera language."
        }
    }
}

fn single_template(task: ExpandTask) -> &'static str {
    match task {
        ExpandTask::TextToImage => SINGLE_SYSTEM_TEMPLATE,
        ExpandTask::TextToVideo => TEXT_TO_VIDEO_SYSTEM_TEMPLATE,
        ExpandTask::ImageToVideo => IMAGE_TO_VIDEO_SYSTEM_TEMPLATE,
        ExpandTask::VideoToVideo => VIDEO_TO_VIDEO_SYSTEM_TEMPLATE,
        ExpandTask::Retake => RETAKE_SYSTEM_TEMPLATE,
        ExpandTask::KeyframeInterpolation => KEYFRAME_SYSTEM_TEMPLATE,
        ExpandTask::AudioDrivenVideo => AUDIO_DRIVEN_VIDEO_SYSTEM_TEMPLATE,
        ExpandTask::ReferenceToAudioVideo => REFERENCE_TO_AUDIO_VIDEO_SYSTEM_TEMPLATE,
        ExpandTask::TextToAudio => TEXT_TO_AUDIO_SYSTEM_TEMPLATE,
    }
}

fn batch_template(task: ExpandTask) -> &'static str {
    match task {
        ExpandTask::TextToImage => BATCH_SYSTEM_TEMPLATE,
        ExpandTask::TextToVideo => VIDEO_BATCH_SYSTEM_TEMPLATE,
        ExpandTask::TextToAudio => AUDIO_BATCH_SYSTEM_TEMPLATE,
        ExpandTask::ImageToVideo
        | ExpandTask::VideoToVideo
        | ExpandTask::Retake
        | ExpandTask::KeyframeInterpolation
        | ExpandTask::AudioDrivenVideo
        | ExpandTask::ReferenceToAudioVideo => CONDITIONED_VIDEO_BATCH_SYSTEM_TEMPLATE,
    }
}

/// Resolve the effective word limit and model notes for a route, applying
/// any user-provided overrides on top of the corpus defaults. A custom
/// `style_notes` override replaces the guide excerpt wholesale; a custom
/// `word_limit` re-renders the excerpt with that budget.
fn route_hints(context: Option<&ExpandContext>) -> prompting::RouteHints {
    prompting::RouteHints {
        source_video: context.is_some_and(|context| {
            context.references.iter().any(|reference| {
                reference.kind == crate::GenerationReferenceKind::Video
                    && reference.role == Some(crate::ExpandReferenceRole::Source)
            })
        }),
    }
}

#[cfg(test)]
fn resolve_task_family_config(
    family: &str,
    model: Option<&str>,
    task: ExpandTask,
    overrides: Option<&FamilyOverride>,
) -> (u32, String) {
    resolve_task_family_config_with_hints(
        family,
        model,
        task,
        overrides,
        prompting::RouteHints::default(),
    )
}

fn resolve_task_family_config_with_hints(
    family: &str,
    model: Option<&str>,
    task: ExpandTask,
    overrides: Option<&FamilyOverride>,
    hints: prompting::RouteHints,
) -> (u32, String) {
    let route = prompting::route_with_hints(family, model, Some(task), hints).ok();
    let default_limit = route
        .as_ref()
        .map_or(GENERIC_WORD_LIMIT, prompting::PromptingRoute::word_limit);
    let word_limit = overrides
        .and_then(|value| value.word_limit)
        .unwrap_or(default_limit);
    // A family with no text encoder gets no guide and no style notes: the
    // only thing worth saying is how to prepare the image. The surfaces
    // answer before any backend is created, so this is what a caller that
    // still builds the messages sees rather than a guide that says "write
    // no prompt" handed to a model asked to write one.
    if let Some(advice) = ignored_prompt_advice(family) {
        return (word_limit, advice.text());
    }
    if let Some(notes) = overrides.and_then(|value| value.style_notes.clone()) {
        return (word_limit, notes);
    }
    let notes = route.as_ref().map_or_else(
        || GENERIC_MODEL_NOTES.to_string(),
        |route| guide_notes(route, word_limit),
    );
    (word_limit, notes)
}

/// Append the generation facts (identity, canvas, clip, references) after the
/// guide so a custom template that omits a placeholder cannot drop them.
fn append_generation_context(
    system: &mut String,
    family: &str,
    task: ExpandTask,
    context: Option<&ExpandContext>,
) {
    if let Some(context) = context {
        let rendered = prompting::render_generation_context(family, task, context);
        if !rendered.trim().is_empty() {
            system.push_str("\n\nGENERATION CONTEXT:\n");
            system.push_str(&rendered);
        }
    }
}

/// Append the style directive to a rendered system prompt. Audio-only tasks
/// translate the same user-facing style into sonic language rather than
/// introducing contradictory visual scene direction.
fn apply_style_directive(system: &mut String, style: Option<&str>, task: ExpandTask) {
    if let Some(style) = style {
        let style = style.trim();
        if !style.is_empty() {
            if task == ExpandTask::TextToAudio {
                system.push_str(&format!(
                    "\n\nSTYLE DIRECTIVE: Express this sonic style — {style}. \
                     Weave the audible cues naturally into the direction; do not just list them."
                ));
            } else {
                system.push_str(&format!(
                    "\n\nSTYLE DIRECTIVE: Render the scene in this visual style — {style}. \
                     Weave these cues naturally into the description; do not just list them."
                ));
            }
        }
    }
}

/// Build chat messages for a single prompt expansion.
///
/// Accepts optional custom template, per-family overrides, and a visual style
/// to weave into the expansion. Returns `Vec<(role, content)>` tuples.
pub fn build_single_messages(
    prompt: &str,
    family: &str,
    custom_template: Option<&str>,
    family_override: Option<&FamilyOverride>,
    style: Option<&str>,
) -> Vec<(String, String)> {
    build_single_messages_for_task(
        prompt,
        family,
        ExpandTask::for_family(family),
        custom_template,
        family_override,
        style,
        None,
    )
}

/// Build single expansion messages for an explicitly resolved task.
///
/// `context` carries the exact identity (which selects a per-checkpoint
/// leaf) and the generation facts rendered after the guide.
pub fn build_single_messages_for_task(
    prompt: &str,
    family: &str,
    task: ExpandTask,
    custom_template: Option<&str>,
    family_override: Option<&FamilyOverride>,
    style: Option<&str>,
    context: Option<&ExpandContext>,
) -> Vec<(String, String)> {
    let model = context.and_then(|context| context.model.as_deref());
    let (word_limit, model_notes) = resolve_task_family_config_with_hints(
        family,
        model,
        task,
        family_override,
        route_hints(context),
    );
    let template = custom_template.unwrap_or_else(|| single_template(task));
    let mut system = template
        .replace("{WORD_LIMIT}", &word_limit.to_string())
        .replace("{MODEL_NOTES}", &model_notes);
    append_generation_context(&mut system, family, task, context);
    apply_style_directive(&mut system, style, task);

    vec![
        ("system".to_string(), system),
        ("user".to_string(), prompt.to_string()),
    ]
}

/// Build chat messages for batch variation generation.
///
/// Accepts optional custom template, per-family overrides, and a visual style
/// to weave into the expansion. Returns `Vec<(role, content)>` tuples.
pub fn build_batch_messages(
    prompt: &str,
    family: &str,
    variations: usize,
    custom_template: Option<&str>,
    family_override: Option<&FamilyOverride>,
    style: Option<&str>,
) -> Vec<(String, String)> {
    build_batch_messages_with_context(
        prompt,
        family,
        variations,
        None,
        custom_template,
        family_override,
        style,
    )
}

/// Build batch messages with the logical positions represented by this
/// bounded completion. Position context prevents later chunks from simply
/// repeating the first few concepts in a large logical batch.
pub fn build_batch_messages_with_context(
    prompt: &str,
    family: &str,
    variations: usize,
    logical_range: Option<(usize, usize)>,
    custom_template: Option<&str>,
    family_override: Option<&FamilyOverride>,
    style: Option<&str>,
) -> Vec<(String, String)> {
    build_batch_messages_with_context_for_task(
        prompt,
        family,
        variations,
        ExpandTask::for_family(family),
        logical_range,
        custom_template,
        family_override,
        style,
        None,
    )
}

/// Build batch expansion messages for an explicitly resolved task.
// This is the task-aware counterpart to the public builder above. Keeping the
// existing template inputs explicit preserves API compatibility at call sites.
#[allow(clippy::too_many_arguments)]
pub fn build_batch_messages_with_context_for_task(
    prompt: &str,
    family: &str,
    variations: usize,
    task: ExpandTask,
    logical_range: Option<(usize, usize)>,
    custom_template: Option<&str>,
    family_override: Option<&FamilyOverride>,
    style: Option<&str>,
    context: Option<&ExpandContext>,
) -> Vec<(String, String)> {
    let model = context.and_then(|context| context.model.as_deref());
    let (word_limit, model_notes) = resolve_task_family_config_with_hints(
        family,
        model,
        task,
        family_override,
        route_hints(context),
    );
    let template = custom_template.unwrap_or_else(|| batch_template(task));
    let mut system = template
        .replace("{N}", &variations.to_string())
        .replace("{WORD_LIMIT}", &word_limit.to_string())
        .replace("{MODEL_NOTES}", &model_notes)
        .replace("{TASK_NOTES}", task_notes(task));
    append_generation_context(&mut system, family, task, context);
    if let Some((start, total)) = logical_range {
        let end = start.saturating_add(variations).saturating_sub(1);
        system.push_str(&format!(
            "\n\nLARGE BATCH CONTEXT: Generate logical variations {start} through {end} \
             of {total}. Make them distinct from concepts likely used for every earlier \
             position in this batch; use the position numbers to drive new combinations."
        ));
    }
    apply_style_directive(&mut system, style, task);

    vec![
        ("system".to_string(), system),
        ("user".to_string(), prompt.to_string()),
    ]
}

/// Build subject-preserving Remix messages. Remix deliberately ignores custom
/// Expand templates so an expansion override cannot erase its preservation
/// contract. `logical_range` keeps dimension labels stable across chunks.
#[allow(clippy::too_many_arguments)]
pub fn build_remix_messages_with_context_for_task(
    prompt: &str,
    family: &str,
    variations: usize,
    task: ExpandTask,
    logical_range: Option<(usize, usize)>,
    family_override: Option<&FamilyOverride>,
    style: Option<&str>,
    dimensions: &[RemixDimension],
    context: Option<&ExpandContext>,
) -> Vec<(String, String)> {
    let model = context.and_then(|context| context.model.as_deref());
    let (word_limit, model_notes) = resolve_task_family_config_with_hints(
        family,
        model,
        task,
        family_override,
        route_hints(context),
    );
    let start = logical_range.map_or(1, |(start, _)| start);
    let dimension_plan = (0..variations)
        .map(|offset| {
            let position = start + offset;
            let labels = remix_dimensions_for_position(dimensions, position)
                .into_iter()
                .map(|dimension| dimension.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            format!("{position}. {labels}")
        })
        .collect::<Vec<_>>()
        .join("\n");
    let mut system = REMIX_SYSTEM_TEMPLATE
        .replace("{N}", &variations.to_string())
        .replace("{WORD_LIMIT}", &word_limit.to_string())
        .replace("{DIMENSION_PLAN}", &dimension_plan)
        .replace("{TASK_NOTES}", task_notes(task))
        .replace("{MODEL_NOTES}", &model_notes);
    append_generation_context(&mut system, family, task, context);
    if let Some((_, total)) = logical_range {
        system.push_str(&format!(
            "\n\nLOGICAL SET: These are positions {start} through {} of {total}; avoid concepts likely used by earlier positions.",
            start.saturating_add(variations).saturating_sub(1)
        ));
    }
    apply_style_directive(&mut system, style, task);
    if style.is_some_and(|value| !value.trim().is_empty()) {
        system.push_str("\nThe STYLE DIRECTIVE is locked: preserve it in every alternative.");
    }
    vec![
        ("system".to_string(), system),
        ("user".to_string(), prompt.to_string()),
    ]
}

/// Format a ChatML prompt string for local Qwen3 inference.
///
/// When `thinking` is false, appends `<think>\n\n</think>\n\n` after the
/// assistant prefix to disable thinking mode (same pattern as Flux.2 Klein).
pub fn format_chatml(messages: &[(String, String)], thinking: bool) -> String {
    let mut result = String::new();
    for (role, content) in messages {
        result.push_str(&format!("<|im_start|>{role}\n{content}<|im_end|>\n"));
    }
    result.push_str("<|im_start|>assistant\n");
    if !thinking {
        result.push_str("<think>\n\n</think>\n\n");
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── family_config ────────────────────────────────────────────────────

    #[test]
    fn family_config_reads_word_limits_from_the_prompting_registry() {
        for (family, expected) in [("sd15", 50), ("sdxl", 60), ("wuerstchen", 50)] {
            let guide = prompting::family_guide(family).unwrap();
            let (limit, notes) = family_config(family);
            assert_eq!(limit, guide.word_limit, "{family}");
            assert_eq!(limit, expected, "{family}");
            assert!(
                notes.starts_with(&format!("MODEL PROMPTING GUIDE ({family} family):")),
                "{family}: {notes}"
            );
        }
    }

    #[test]
    fn family_config_accepts_every_registered_alias() {
        for (alias, family) in [
            ("sd1.5", "sd15"),
            ("stable-diffusion-1.5", "sd15"),
            ("wuerstchen-v2", "wuerstchen"),
            ("ltx-2", "ltx2"),
            ("SD15", "sd15"),
            ("SDXL", "sdxl"),
        ] {
            let (limit, notes) = family_config(alias);
            let guide = prompting::family_guide(family).unwrap();
            assert_eq!(limit, guide.word_limit, "{alias}");
            assert!(
                notes.contains(&format!("({family} family)")),
                "{alias}: {notes}"
            );
        }
    }

    #[test]
    fn every_manifest_family_gets_its_own_guide_not_the_generic_notes() {
        for guide in prompting::FAMILY_GUIDES {
            let (_, notes) = family_config(guide.family);
            assert!(!notes.contains(GENERIC_MODEL_NOTES), "{}", guide.family);
            assert!(
                notes.contains(&format!("({} family)", guide.family)),
                "{}",
                guide.family
            );
        }
    }

    // ── prompt-Ignored families ──────────────────────────────────────────

    #[test]
    fn a_prompt_ignored_family_answers_with_the_guides_image_advice() {
        use crate::generation_profile::{prompt_requirement_for_family, PromptRequirement};
        for guide in prompting::FAMILY_GUIDES {
            let advice = ignored_prompt_advice(guide.family);
            let ignored = prompt_requirement_for_family(Some(guide.family), true)
                == PromptRequirement::Ignored;
            assert_eq!(advice.is_some(), ignored, "{}", guide.family);
        }
        let advice = ignored_prompt_advice("hunyuan3d").expect("hunyuan3d ignores the prompt");
        assert_eq!(advice.family, "hunyuan3d");
        assert!(
            advice.headline.contains("reads no prompt"),
            "{}",
            advice.headline
        );
        // The preparation advice is the guide's own Generation context
        // section, rendered from the corpus rather than typed again in Rust.
        let route = prompting::route("hunyuan3d", None, None).unwrap();
        let section = prompting::section_excerpt(
            route.family.contents,
            "Generation context",
            route.word_limit(),
        )
        .unwrap();
        assert_eq!(advice.preparation, section);
        assert!(advice.preparation.contains("three-quarter"));
        assert!(!advice.preparation.contains("## "));
        assert!(!advice.preparation.contains("```"));
        assert!(advice.text().starts_with(&advice.headline));
        assert!(advice.text().ends_with(&advice.preparation));
        assert!(
            ignored_prompt_advice("HUNYUAN3D").is_some(),
            "case-insensitive"
        );
        assert!(ignored_prompt_advice("flux").is_none());
        assert!(ignored_prompt_advice("some-future-model").is_none());
    }

    #[test]
    fn a_prompt_ignored_family_short_circuits_the_guide_notes() {
        let advice = ignored_prompt_advice("hunyuan3d").unwrap();
        let (limit, notes) =
            resolve_task_family_config("hunyuan3d", None, ExpandTask::TextToImage, None);
        assert_eq!(
            limit,
            prompting::family_guide("hunyuan3d").unwrap().word_limit
        );
        assert_eq!(notes, advice.text());
        assert!(!notes.contains("MODEL PROMPTING GUIDE"));
        let ov = FamilyOverride {
            word_limit: Some(12),
            style_notes: Some("ignored style notes".into()),
        };
        let (limit, notes) =
            resolve_task_family_config("hunyuan3d", None, ExpandTask::TextToImage, Some(&ov));
        assert_eq!(limit, 12, "the word limit override is still honoured");
        assert_eq!(
            notes,
            advice.text(),
            "style notes cannot revive a prompt nothing reads"
        );
    }

    #[test]
    fn generation_context_states_when_the_prompt_is_not_read() {
        use crate::generation_profile::PromptRequirement;
        let context = ExpandContext {
            model: Some("hunyuan3d-mini-turbo:fp16".into()),
            prompt_mode: Some(PromptRequirement::Ignored),
            references: vec![crate::ExpandReference::image(
                crate::ExpandReferenceRole::Source,
            )],
            ..ExpandContext::default()
        };
        let rendered =
            prompting::render_generation_context("hunyuan3d", ExpandTask::TextToImage, &context);
        assert!(
            rendered.contains("Prompt: not read by this model"),
            "{rendered}"
        );
        let optional = ExpandContext {
            prompt_mode: Some(PromptRequirement::Optional),
            ..ExpandContext::default()
        };
        let rendered =
            prompting::render_generation_context("ltx2", ExpandTask::ImageToVideo, &optional);
        assert!(rendered.contains("Prompt: optional"), "{rendered}");
        let required = ExpandContext {
            prompt_mode: Some(PromptRequirement::Required),
            ..ExpandContext::default()
        };
        let rendered =
            prompting::render_generation_context("flux", ExpandTask::TextToImage, &required);
        assert!(!rendered.contains("Prompt:"), "{rendered}");
    }

    #[test]
    fn family_config_unknown_uses_generic_notes() {
        let (limit, notes) = family_config("some-future-model");
        assert_eq!(limit, GENERIC_WORD_LIMIT);
        assert_eq!(notes, GENERIC_MODEL_NOTES);
    }

    #[test]
    fn text_to_audio_route_replaces_the_family_guide() {
        let route = prompting::route("ltx2", None, Some(ExpandTask::TextToAudio)).unwrap();
        assert!(route.task.is_some_and(|leaf| leaf.standalone));
        let (limit, notes) = guide_config("ltx2", None, ExpandTask::TextToAudio);
        assert_eq!(limit, route.word_limit());
        let family_only = prompting::excerpt(route.family.contents, limit);
        let first_family_line = family_only.lines().nth(2).unwrap_or("");
        assert!(
            first_family_line.is_empty() || !notes.contains(first_family_line),
            "family prose leaked into the audio route: {first_family_line}"
        );
    }

    #[test]
    fn exact_model_identity_selects_the_identity_leaf() {
        let context = ExpandContext {
            model: Some("wan22-ti2v-5b:q8".into()),
            ..ExpandContext::default()
        };
        let msgs = build_single_messages_for_task(
            "the boat drifts",
            "wan",
            ExpandTask::ImageToVideo,
            None,
            None,
            None,
            Some(&context),
        );
        let route = prompting::route(
            "wan",
            Some("wan22-ti2v-5b:q8"),
            Some(ExpandTask::ImageToVideo),
        )
        .unwrap();
        assert_eq!(
            route.task.map(|leaf| leaf.path),
            Some("wan/image-conditioned.md")
        );
        assert!(msgs[0]
            .1
            .contains(&format!("under {} words", route.word_limit())));
        assert!(msgs[0]
            .1
            .contains("GENERATION CONTEXT:\nTarget: video on wan22-ti2v-5b:q8"));
    }

    #[test]
    fn a_source_video_routes_ltx2_audio_driven_expansion_to_the_dub_it_leaf() {
        let context = ExpandContext {
            model: Some("ltx-2.3-22b-distilled:fp8".into()),
            references: vec![
                crate::ExpandReference {
                    kind: crate::GenerationReferenceKind::Video,
                    has_audio: false,
                    role: Some(crate::ExpandReferenceRole::Source),
                },
                crate::ExpandReference {
                    kind: crate::GenerationReferenceKind::Audio,
                    has_audio: false,
                    role: Some(crate::ExpandReferenceRole::Source),
                },
            ],
            ..ExpandContext::default()
        };
        let with_clip = build_single_messages_for_task(
            "she says: the harbour freezes every winter",
            "ltx2",
            ExpandTask::AudioDrivenVideo,
            None,
            None,
            None,
            Some(&context),
        );
        let dub = prompting::route_with_leaf("ltx2", "", "ltx2/dub-it.md").unwrap();
        let dub_excerpt = prompting::excerpt(dub.task.unwrap().contents, dub.word_limit());
        let first_line = dub_excerpt.lines().nth(2).unwrap_or("");
        assert!(!first_line.is_empty());
        assert!(with_clip[0].1.contains(first_line), "{}", with_clip[0].1);
        let without_clip = build_single_messages_for_task(
            "a paper sculpture reacting to music",
            "ltx2",
            ExpandTask::AudioDrivenVideo,
            None,
            None,
            None,
            None,
        );
        assert!(!without_clip[0].1.contains(first_line));
    }

    #[test]
    fn generation_context_is_appended_even_with_a_custom_template() {
        let context = ExpandContext {
            frames: Some(81),
            fps: Some(16),
            ..ExpandContext::default()
        };
        let msgs = build_single_messages_for_task(
            "a fox",
            "wan",
            ExpandTask::TextToVideo,
            Some("Custom: {WORD_LIMIT}"),
            None,
            None,
            Some(&context),
        );
        assert!(msgs[0].1.starts_with("Custom: "));
        assert!(msgs[0].1.contains("81 frames at 16 fps (5.1 s)"));
        let without = build_single_messages_for_task(
            "a fox",
            "wan",
            ExpandTask::TextToVideo,
            None,
            None,
            None,
            None,
        );
        assert!(!without[0].1.contains("GENERATION CONTEXT"));
    }

    #[test]
    fn text_to_video_uses_chronological_motion_instructions() {
        let msgs = build_single_messages_for_task(
            "a lighthouse in a storm",
            "ltx2",
            ExpandTask::TextToVideo,
            None,
            None,
            None,
            None,
        );
        let system = &msgs[0].1;
        assert!(system.contains("video generation prompt writer"));
        assert!(system.contains("chronological"));
        assert!(system.contains("ordered motion"));
        assert!(!system.contains("image generation prompt writer"));
    }

    #[test]
    fn image_to_video_treats_the_opening_frame_as_authority() {
        let msgs = build_single_messages_for_task(
            "she turns toward the window",
            "ltx-video",
            ExpandTask::ImageToVideo,
            None,
            None,
            None,
            None,
        );
        let system = &msgs[0].1;
        assert!(system.contains("source image is the visual authority"));
        assert!(system.contains("temporal change"));
        assert!(system.contains("Do not re-invent"));
    }

    #[test]
    fn every_specialized_video_and_audio_task_keeps_its_authority() {
        let cases = [
            (ExpandTask::VideoToVideo, "visual and temporal authority"),
            (ExpandTask::Retake, "selected range"),
            (ExpandTask::KeyframeInterpolation, "fixed visual anchors"),
            (
                ExpandTask::AudioDrivenVideo,
                "audio timing and events are authoritative",
            ),
            (
                ExpandTask::TextToAudio,
                "Do not add visual composition or camera language",
            ),
        ];

        for (task, expected) in cases {
            let msgs = build_single_messages_for_task(
                "preserve the source",
                "ltx2",
                task,
                None,
                None,
                None,
                None,
            );
            assert!(msgs[0].1.contains(expected), "{task} lost its task policy");
        }
    }

    #[test]
    fn conditioned_video_batch_preserves_authority_across_variations() {
        let msgs = build_batch_messages_with_context_for_task(
            "the dancer completes a turn",
            "ltx2",
            3,
            ExpandTask::KeyframeInterpolation,
            None,
            None,
            None,
            None,
            None,
        );
        let system = &msgs[0].1;
        assert!(system.contains("keyframes are fixed visual anchors"));
        assert!(system.contains("Generate 3 distinct directions"));
        assert!(system.contains("never vary source identity"));
        assert!(!system.contains("vary controlled motion, camera behavior"));
        assert!(!system.contains("distinct image prompts"));
    }

    #[test]
    fn remix_template_preserves_subject_and_assigns_dimensions() {
        let messages = build_remix_messages_with_context_for_task(
            "a red lighthouse with exactly three windows",
            "flux",
            3,
            ExpandTask::TextToImage,
            Some((1, 3)),
            None,
            Some("linocut"),
            &[RemixDimension::Camera, RemixDimension::Lighting],
            None,
        );
        let system = &messages[0].1;
        assert!(system.contains("central subject"));
        assert!(system.contains("every explicit user constraint"));
        assert!(system.contains("1. camera"));
        assert!(system.contains("2. lighting"));
        assert!(system.contains("3. camera"));
        assert!(system.contains("STYLE DIRECTIVE"));
        assert!(system.contains("locked"));
        assert_eq!(messages[1].1, "a red lighthouse with exactly three windows");
    }

    #[test]
    fn conditioned_remix_keeps_task_authority() {
        let messages = build_remix_messages_with_context_for_task(
            "she completes the turn",
            "ltx2",
            2,
            ExpandTask::KeyframeInterpolation,
            Some((1, 2)),
            None,
            None,
            &[RemixDimension::Movement],
            None,
        );
        let system = &messages[0].1;
        assert!(system.contains("keyframes are fixed visual anchors"));
        assert!(system.contains("vary ONLY its assigned creative dimension"));
    }

    #[test]
    fn text_to_audio_batch_never_receives_video_or_camera_instructions() {
        let msgs = build_batch_messages_with_context_for_task(
            "rain on a tin roof",
            "ltx2",
            3,
            ExpandTask::TextToAudio,
            None,
            None,
            None,
            None,
            None,
        );
        let system = &msgs[0].1;
        assert!(system.contains("audio generation prompt writer"));
        assert!(system.contains("Do not use visual composition or camera language"));
        assert!(!system.contains("video generation prompt writer"));
        assert!(!system.contains("camera behavior, environment response"));
        assert!(!system.contains("natural-language shot direction"));
    }

    #[test]
    fn text_to_audio_style_is_expressed_as_sonic_direction() {
        let msgs = build_single_messages_for_task(
            "rain on a tin roof",
            "ltx2",
            ExpandTask::TextToAudio,
            None,
            None,
            Some("noir"),
            None,
        );
        let system = &msgs[0].1;
        assert!(system.contains("Express this sonic style — noir"));
        assert!(!system.contains("Render the scene"));
        assert!(!system.contains("natural-language shot direction"));
    }

    #[test]
    fn audio_driven_video_preserves_an_attached_source_clip() {
        let msgs = build_single_messages_for_task(
            "match her mouth to the speech",
            "ltx2",
            ExpandTask::AudioDrivenVideo,
            None,
            None,
            None,
            None,
        );
        let system = &msgs[0].1;
        assert!(system.contains("When a source clip is attached"));
        assert!(system.contains("preserve its identity, framing, camera path, motion, and timing"));
    }

    // ── build_single_messages ────────────────────────────────────────────

    #[test]
    fn single_messages_flux() {
        let msgs = build_single_messages("a cat", "flux", None, None, None);
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].0, "system");
        assert!(msgs[0].1.contains("150 words"));
        assert_eq!(msgs[1].0, "user");
        assert_eq!(msgs[1].1, "a cat");
    }

    #[test]
    fn single_messages_sd15() {
        let msgs = build_single_messages("a cat", "sd15", None, None, None);
        assert_eq!(msgs.len(), 2);
        assert!(msgs[0].1.contains("50 words"));
        assert!(msgs[0].1.contains("MODEL PROMPTING GUIDE (sd15 family)"));
    }

    #[test]
    fn single_messages_preserves_user_prompt() {
        let prompt = "a cyberpunk city at night with neon reflections";
        let msgs = build_single_messages(prompt, "flux", None, None, None);
        assert_eq!(msgs[1].1, prompt);
    }

    // ── build_batch_messages ─────────────────────────────────────────────

    #[test]
    fn batch_messages_sdxl() {
        let msgs = build_batch_messages("sunset", "sdxl", 3, None, None, None);
        assert_eq!(msgs.len(), 2);
        assert!(msgs[0].1.contains("3 distinct"));
        assert!(msgs[0].1.contains("JSON array"));
        assert!(msgs[0].1.contains("60 words"));
    }

    #[test]
    fn batch_messages_count_substitution() {
        for n in [2, 5, 10] {
            let msgs = build_batch_messages("test", "flux", n, None, None, None);
            assert!(
                msgs[0].1.contains(&format!("{n} distinct")),
                "should contain '{n} distinct' for variations={n}"
            );
        }
    }

    #[test]
    fn batch_messages_preserves_user_prompt() {
        let prompt = "a dragon in a crystal cave";
        let msgs = build_batch_messages(prompt, "sdxl", 4, None, None, None);
        assert_eq!(msgs[1].1, prompt);
    }

    #[test]
    fn batch_messages_include_large_batch_position_context() {
        let msgs =
            build_batch_messages_with_context("sunset", "sdxl", 4, Some((5, 12)), None, None, None);
        assert!(msgs[0].1.contains("variations 5 through 8 of 12"));
        assert!(msgs[0].1.contains("distinct from concepts"));
    }

    #[test]
    fn batch_messages_keep_position_context_for_single_missing_retry() {
        let msgs =
            build_batch_messages_with_context("sunset", "sdxl", 1, Some((4, 8)), None, None, None);
        assert!(msgs[0].1.contains("Generate 1 distinct"));
        assert!(msgs[0].1.contains("variations 4 through 4 of 8"));
    }

    // ── custom templates and family overrides ────────────────────────────

    #[test]
    fn single_messages_custom_template() {
        let custom = "Custom system: limit {WORD_LIMIT}. Notes: {MODEL_NOTES}";
        let msgs = build_single_messages("a cat", "flux", Some(custom), None, None);
        assert!(msgs[0].1.contains("Custom system: limit 150"));
        assert!(msgs[0]
            .1
            .contains("Notes: MODEL PROMPTING GUIDE (flux family)"));
    }

    #[test]
    fn batch_messages_custom_template() {
        let custom = "Generate {N} prompts, max {WORD_LIMIT} words. {MODEL_NOTES}";
        let msgs = build_batch_messages("a cat", "flux", 3, Some(custom), None, None);
        assert!(msgs[0].1.contains("Generate 3 prompts"));
        assert!(msgs[0].1.contains("max 150 words"));
    }

    #[test]
    fn single_messages_family_override_word_limit() {
        let ov = FamilyOverride {
            word_limit: Some(200),
            style_notes: None,
        };
        let msgs = build_single_messages("a cat", "flux", None, Some(&ov), None);
        assert!(msgs[0].1.contains("200 words"));
        // Should still use the built-in guide
        assert!(msgs[0].1.contains("MODEL PROMPTING GUIDE (flux family)"));
    }

    #[test]
    fn single_messages_family_override_style_notes() {
        let ov = FamilyOverride {
            word_limit: None,
            style_notes: Some("Use haiku style.".to_string()),
        };
        let msgs = build_single_messages("a cat", "sd15", None, Some(&ov), None);
        // Word limit should still be the SD1.5 default (50)
        assert!(msgs[0].1.contains("50 words"));
        // But style notes should be overridden
        assert!(msgs[0].1.contains("Use haiku style."));
        assert!(!msgs[0].1.contains("keyword"));
    }

    #[test]
    fn batch_messages_family_override_both() {
        let ov = FamilyOverride {
            word_limit: Some(75),
            style_notes: Some("Cinematic descriptions only.".to_string()),
        };
        let msgs = build_batch_messages("a cat", "sdxl", 4, None, Some(&ov), None);
        assert!(msgs[0].1.contains("75 words"));
        assert!(msgs[0].1.contains("Cinematic descriptions only."));
    }

    #[test]
    fn custom_template_with_family_override() {
        let custom = "Limit: {WORD_LIMIT}. Style: {MODEL_NOTES}";
        let ov = FamilyOverride {
            word_limit: Some(300),
            style_notes: Some("Go wild.".to_string()),
        };
        let msgs = build_single_messages("test", "flux", Some(custom), Some(&ov), None);
        assert_eq!(msgs[0].1, "Limit: 300. Style: Go wild.");
    }

    #[test]
    fn resolve_family_config_defaults_preserved() {
        let (limit, notes) =
            resolve_task_family_config("sd15", None, ExpandTask::TextToImage, None);
        assert_eq!(limit, 50);
        assert!(notes.contains("(sd15 family)"));
    }

    #[test]
    fn resolve_family_config_partial_override() {
        let ov = FamilyOverride {
            word_limit: Some(100),
            style_notes: None,
        };
        let (limit, notes) =
            resolve_task_family_config("sd15", None, ExpandTask::TextToImage, Some(&ov));
        assert_eq!(limit, 100);
        // Notes still come from the guide, re-rendered with the new budget.
        assert!(notes.contains("(sd15 family)"));
        assert!(!notes.contains("{{word_limit}}"));
    }

    #[test]
    fn style_notes_override_replaces_the_guide_for_expand_and_remix() {
        let ov = FamilyOverride {
            word_limit: None,
            style_notes: Some("House style only.".to_string()),
        };
        let (_, notes) =
            resolve_task_family_config("sdxl", None, ExpandTask::TextToImage, Some(&ov));
        assert_eq!(notes, "House style only.");
        let remix = build_remix_messages_with_context_for_task(
            "a cat",
            "sdxl",
            2,
            ExpandTask::TextToImage,
            None,
            Some(&ov),
            None,
            &[RemixDimension::Lighting],
            None,
        );
        assert!(remix[0].1.contains("House style only."));
        assert!(!remix[0].1.contains("MODEL PROMPTING GUIDE"));
    }

    // ── style directive ──────────────────────────────────────────────────

    #[test]
    fn single_messages_style_appends_directive() {
        let msgs = build_single_messages("a cat", "flux", None, None, Some("gritty film noir"));
        assert_eq!(msgs[0].0, "system");
        assert!(msgs[0].1.contains(
            "STYLE DIRECTIVE: Render the scene in this visual style — gritty film noir."
        ));
        assert!(msgs[0].1.contains("Weave these cues naturally"));
        // The user message must stay the bare prompt — style never leaks there.
        assert_eq!(msgs[1].1, "a cat");
    }

    #[test]
    fn single_messages_no_style_no_directive() {
        let msgs = build_single_messages("a cat", "flux", None, None, None);
        assert!(!msgs[0].1.contains("STYLE DIRECTIVE"));
    }

    #[test]
    fn batch_messages_style_appends_directive() {
        let msgs = build_batch_messages("a cat", "sdxl", 3, None, None, Some("watercolor wash"));
        assert_eq!(msgs[0].0, "system");
        assert!(msgs[0]
            .1
            .contains("STYLE DIRECTIVE: Render the scene in this visual style — watercolor wash."));
        assert!(msgs[0].1.contains("do not just list them"));
        assert_eq!(msgs[1].1, "a cat");
    }

    #[test]
    fn batch_messages_no_style_no_directive() {
        let msgs = build_batch_messages("a cat", "sdxl", 3, None, None, None);
        assert!(!msgs[0].1.contains("STYLE DIRECTIVE"));
    }

    #[test]
    fn style_directive_applies_after_custom_template() {
        let custom = "Custom system: limit {WORD_LIMIT}. Notes: {MODEL_NOTES}";
        let msgs = build_single_messages("a cat", "flux", Some(custom), None, Some("pixel art"));
        assert!(msgs[0].1.starts_with("Custom system: limit 150"));
        assert!(msgs[0]
            .1
            .contains("STYLE DIRECTIVE: Render the scene in this visual style — pixel art."));
    }

    // ── format_chatml ────────────────────────────────────────────────────

    #[test]
    fn chatml_without_thinking() {
        let msgs = vec![
            ("system".to_string(), "You are helpful.".to_string()),
            ("user".to_string(), "hello".to_string()),
        ];
        let result = format_chatml(&msgs, false);
        assert!(result.contains("<|im_start|>system\nYou are helpful.<|im_end|>"));
        assert!(result.contains("<|im_start|>user\nhello<|im_end|>"));
        assert!(result.contains("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
    }

    #[test]
    fn chatml_with_thinking() {
        let msgs = vec![("user".to_string(), "hello".to_string())];
        let result = format_chatml(&msgs, true);
        assert!(result.contains("<|im_start|>assistant\n"));
        assert!(!result.contains("<think>"));
    }

    #[test]
    fn chatml_ends_with_assistant_prefix() {
        let msgs = vec![("user".to_string(), "test".to_string())];
        for thinking in [true, false] {
            let result = format_chatml(&msgs, thinking);
            assert!(
                result.contains("<|im_start|>assistant\n"),
                "should end with assistant prefix"
            );
        }
    }

    #[test]
    fn chatml_empty_messages() {
        let msgs: Vec<(String, String)> = vec![];
        let result = format_chatml(&msgs, false);
        // Should still have assistant prefix + thinking disable
        assert!(result.starts_with("<|im_start|>assistant\n"));
    }
}
