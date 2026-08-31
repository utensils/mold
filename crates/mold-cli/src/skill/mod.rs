//! Agent-specific skill bundle rendering and managed installation.

use anyhow::{Context, Result};
use clap::{ArgGroup, Parser, Subcommand, ValueEnum};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Component, Path, PathBuf};

const SKILL_DIR_NAME: &str = "mold";
const MANIFEST_FILE: &str = ".mold-skill.json";
const MANIFEST_SCHEMA: u32 = 1;
const DESCRIPTION: &str = "Generate and manage AI images and video with the mold CLI. Use when asked to create images or clips, transform source media, operate local or remote Mold servers, manage models and queues, inspect galleries, configure GPU inference, or automate a Mold CLI, REST, or MCP workflow.";
const TEMPLATE_MD: &str = include_str!("template.md");
const CLI_MD: &str = include_str!("references/cli.md");
const SAFETY_MD: &str = include_str!("references/safety.md");
const EXAMPLES_MD: &str = include_str!("examples/quickstart.md");

const SHARED_PROMPTING_PATH: &str = "references/prompting/shared.md";

#[derive(Clone, Copy)]
struct FamilyGuide {
    family: &'static str,
    path: &'static str,
    contents: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TaskGuide {
    H3BaseModes,
    H3Ref2va,
    WanTextToVideo,
    WanImageConditioned,
    Ltx2DubIt,
    Ltx2TextToAudio,
}

impl TaskGuide {
    fn family(self) -> &'static str {
        match self {
            Self::H3BaseModes | Self::H3Ref2va => "minimax-h3",
            Self::WanTextToVideo | Self::WanImageConditioned => "wan",
            Self::Ltx2DubIt | Self::Ltx2TextToAudio => "ltx2",
        }
    }

    fn path(self) -> &'static str {
        match self {
            Self::H3BaseModes => "references/prompting/minimax-h3/base-modes.md",
            Self::H3Ref2va => "references/prompting/minimax-h3/ref2va.md",
            Self::WanTextToVideo => "references/prompting/wan/text-to-video.md",
            Self::WanImageConditioned => "references/prompting/wan/image-conditioned.md",
            Self::Ltx2DubIt => "references/prompting/ltx2/dub-it.md",
            Self::Ltx2TextToAudio => "references/prompting/ltx2/text-to-audio.md",
        }
    }

    fn contents(self) -> &'static str {
        match self {
            Self::H3BaseModes => include_str!("references/prompting/minimax-h3/base-modes.md"),
            Self::H3Ref2va => include_str!("references/prompting/minimax-h3/ref2va.md"),
            Self::WanTextToVideo => include_str!("references/prompting/wan/text-to-video.md"),
            Self::WanImageConditioned => {
                include_str!("references/prompting/wan/image-conditioned.md")
            }
            Self::Ltx2DubIt => include_str!("references/prompting/ltx2/dub-it.md"),
            Self::Ltx2TextToAudio => include_str!("references/prompting/ltx2/text-to-audio.md"),
        }
    }
}

const TASK_GUIDES: &[TaskGuide] = &[
    TaskGuide::H3BaseModes,
    TaskGuide::H3Ref2va,
    TaskGuide::WanTextToVideo,
    TaskGuide::WanImageConditioned,
    TaskGuide::Ltx2DubIt,
    TaskGuide::Ltx2TextToAudio,
];

const FAMILY_GUIDES: &[FamilyGuide] = &[
    FamilyGuide {
        family: "flux",
        path: "references/prompting/families/flux.md",
        contents: include_str!("references/prompting/families/flux.md"),
    },
    FamilyGuide {
        family: "flux2",
        path: "references/prompting/families/flux2.md",
        contents: include_str!("references/prompting/families/flux2.md"),
    },
    FamilyGuide {
        family: "sd15",
        path: "references/prompting/families/sd15.md",
        contents: include_str!("references/prompting/families/sd15.md"),
    },
    FamilyGuide {
        family: "sdxl",
        path: "references/prompting/families/sdxl.md",
        contents: include_str!("references/prompting/families/sdxl.md"),
    },
    FamilyGuide {
        family: "sd3",
        path: "references/prompting/families/sd3.md",
        contents: include_str!("references/prompting/families/sd3.md"),
    },
    FamilyGuide {
        family: "z-image",
        path: "references/prompting/families/z-image.md",
        contents: include_str!("references/prompting/families/z-image.md"),
    },
    FamilyGuide {
        family: "hunyuan3d",
        path: "references/prompting/families/hunyuan3d.md",
        contents: include_str!("references/prompting/families/hunyuan3d.md"),
    },
    FamilyGuide {
        family: "wuerstchen",
        path: "references/prompting/families/wuerstchen.md",
        contents: include_str!("references/prompting/families/wuerstchen.md"),
    },
    FamilyGuide {
        family: "qwen-image",
        path: "references/prompting/families/qwen-image.md",
        contents: include_str!("references/prompting/families/qwen-image.md"),
    },
    FamilyGuide {
        family: "qwen-image-edit",
        path: "references/prompting/families/qwen-image-edit.md",
        contents: include_str!("references/prompting/families/qwen-image-edit.md"),
    },
    FamilyGuide {
        family: "ltx-video",
        path: "references/prompting/families/ltx-video.md",
        contents: include_str!("references/prompting/families/ltx-video.md"),
    },
    FamilyGuide {
        family: "ltx2",
        path: "references/prompting/families/ltx2.md",
        contents: include_str!("references/prompting/families/ltx2.md"),
    },
    FamilyGuide {
        family: "wan",
        path: "references/prompting/families/wan.md",
        contents: include_str!("references/prompting/families/wan.md"),
    },
    FamilyGuide {
        family: "minimax-h3",
        path: "references/prompting/families/minimax-h3.md",
        contents: include_str!("references/prompting/families/minimax-h3.md"),
    },
    FamilyGuide {
        family: "upscaler",
        path: "references/prompting/families/upscaler.md",
        contents: include_str!("references/prompting/families/upscaler.md"),
    },
];

#[derive(Parser, Debug)]
pub struct SkillArgs {
    #[command(subcommand)]
    pub command: SkillCommand,
}

#[derive(Subcommand, Debug)]
pub enum SkillCommand {
    /// List supported agents, their skill paths, and managed status
    List(SkillListArgs),
    /// Install a rendered Mold skill bundle for explicitly selected agents
    #[command(group(
        ArgGroup::new("targets")
            .required(true)
            .multiple(false)
            .args(["agents", "detected", "all"])
    ))]
    Install(SkillInstallArgs),
    /// Remove only files tracked by Mold's bundle manifest
    Uninstall(SkillUninstallArgs),
    /// Print a rendered bundle file (generic SKILL.md by default)
    Show(SkillShowArgs),
}

#[derive(Parser, Debug)]
pub struct SkillListArgs {
    /// Use ASCII table borders instead of Unicode
    #[arg(long)]
    ascii: bool,
}

#[derive(Parser, Debug)]
pub struct SkillInstallArgs {
    /// Agents to install for
    #[arg(value_enum)]
    agents: Vec<Agent>,
    /// Install under project-level skill directories in the current directory
    #[arg(long)]
    project: bool,
    /// Project directory to install into (implies --project)
    #[arg(long, value_name = "PATH")]
    dir: Option<PathBuf>,
    /// Install for agents detected from user configuration directories
    #[arg(long)]
    detected: bool,
    /// Install for every supported agent regardless of detection
    #[arg(long)]
    all: bool,
}

#[derive(Parser, Debug)]
pub struct SkillUninstallArgs {
    /// Agents to uninstall from (default: every agent path in scope)
    #[arg(value_enum)]
    agents: Vec<Agent>,
    /// Remove from project-level skill directories in the current directory
    #[arg(long)]
    project: bool,
    /// Project directory to uninstall from (implies --project)
    #[arg(long, value_name = "PATH")]
    dir: Option<PathBuf>,
}

#[derive(Parser, Debug)]
pub struct SkillShowArgs {
    /// Render for this agent (defaults to portable Agent Skills)
    #[arg(value_enum, default_value = "agents")]
    agent: Agent,
    /// Bundle-relative file to print
    #[arg(default_value = "SKILL.md")]
    file: PathBuf,
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Agent {
    Claude,
    Codex,
    Pi,
    Openclaw,
    Copilot,
    Cursor,
    Gemini,
    Amp,
    Goose,
    Agents,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
enum RenderProfile {
    Portable,
    Claude,
    Codex,
    Pi,
    Openclaw,
    Copilot,
    Cursor,
    Gemini,
    Goose,
}

impl RenderProfile {
    fn name(self) -> &'static str {
        match self {
            Self::Portable => "portable",
            Self::Claude => "claude",
            Self::Codex => "codex",
            Self::Pi => "pi",
            Self::Openclaw => "openclaw",
            Self::Copilot => "copilot",
            Self::Cursor => "cursor",
            Self::Gemini => "gemini",
            Self::Goose => "goose",
        }
    }
}

struct AgentSpec {
    agent: Agent,
    name: &'static str,
    label: &'static str,
    detect: &'static [&'static str],
    user_skills: &'static [&'static str],
    project_skills: &'static [&'static str],
    user_profile: RenderProfile,
    project_profile: RenderProfile,
}

const AGENTS: &[AgentSpec] = &[
    AgentSpec {
        agent: Agent::Claude,
        name: "claude",
        label: "Claude Code",
        detect: &[".claude"],
        user_skills: &[".claude", "skills"],
        project_skills: &[".claude", "skills"],
        user_profile: RenderProfile::Claude,
        project_profile: RenderProfile::Claude,
    },
    AgentSpec {
        agent: Agent::Codex,
        name: "codex",
        label: "OpenAI Codex CLI",
        detect: &[".codex"],
        user_skills: &[".codex", "skills"],
        project_skills: &[".agents", "skills"],
        user_profile: RenderProfile::Codex,
        project_profile: RenderProfile::Portable,
    },
    AgentSpec {
        agent: Agent::Pi,
        name: "pi",
        label: "Pi",
        detect: &[".pi"],
        user_skills: &[".pi", "agent", "skills"],
        project_skills: &[".pi", "skills"],
        user_profile: RenderProfile::Pi,
        project_profile: RenderProfile::Pi,
    },
    AgentSpec {
        agent: Agent::Openclaw,
        name: "openclaw",
        label: "OpenClaw",
        detect: &[".openclaw"],
        user_skills: &[".openclaw", "skills"],
        project_skills: &[".agents", "skills"],
        user_profile: RenderProfile::Openclaw,
        project_profile: RenderProfile::Portable,
    },
    AgentSpec {
        agent: Agent::Copilot,
        name: "copilot",
        label: "GitHub Copilot CLI",
        detect: &[".copilot"],
        user_skills: &[".copilot", "skills"],
        project_skills: &[".github", "skills"],
        user_profile: RenderProfile::Copilot,
        project_profile: RenderProfile::Copilot,
    },
    AgentSpec {
        agent: Agent::Cursor,
        name: "cursor",
        label: "Cursor",
        detect: &[".cursor"],
        user_skills: &[".cursor", "skills"],
        project_skills: &[".agents", "skills"],
        user_profile: RenderProfile::Cursor,
        project_profile: RenderProfile::Portable,
    },
    AgentSpec {
        agent: Agent::Gemini,
        name: "gemini",
        label: "Gemini CLI",
        detect: &[".gemini"],
        user_skills: &[".gemini", "skills"],
        project_skills: &[".agents", "skills"],
        user_profile: RenderProfile::Gemini,
        project_profile: RenderProfile::Portable,
    },
    // Amp's current personal Agent Skills root is shared with the portable spec.
    AgentSpec {
        agent: Agent::Amp,
        name: "amp",
        label: "Amp",
        detect: &[".config", "amp"],
        user_skills: &[".config", "agents", "skills"],
        project_skills: &[".agents", "skills"],
        user_profile: RenderProfile::Portable,
        project_profile: RenderProfile::Portable,
    },
    AgentSpec {
        agent: Agent::Goose,
        name: "goose",
        label: "Goose",
        detect: &[".config", "goose"],
        user_skills: &[".config", "goose", "skills"],
        project_skills: &[".agents", "skills"],
        user_profile: RenderProfile::Goose,
        project_profile: RenderProfile::Portable,
    },
    AgentSpec {
        agent: Agent::Agents,
        name: "agents",
        label: "Agent Skills (generic)",
        detect: &[".agents"],
        user_skills: &[".agents", "skills"],
        project_skills: &[".agents", "skills"],
        user_profile: RenderProfile::Portable,
        project_profile: RenderProfile::Portable,
    },
];

impl Agent {
    fn spec(self) -> &'static AgentSpec {
        AGENTS
            .iter()
            .find(|spec| spec.agent == self)
            .expect("every agent has a spec")
    }

    fn user_skill_dir(self, home: &Path) -> PathBuf {
        join_segments(home, self.spec().user_skills).join(SKILL_DIR_NAME)
    }

    fn project_skill_dir(self, root: &Path) -> PathBuf {
        join_segments(root, self.spec().project_skills).join(SKILL_DIR_NAME)
    }

    fn is_detected(self, home: &Path) -> bool {
        join_segments(home, self.spec().detect).is_dir()
    }
}

fn join_segments(base: &Path, segments: &[&str]) -> PathBuf {
    segments.iter().fold(base.to_path_buf(), |mut path, part| {
        path.push(part);
        path
    })
}

enum Scope {
    User(PathBuf),
    Project(PathBuf),
}

impl Scope {
    fn skill_dir(&self, agent: Agent) -> PathBuf {
        match self {
            Self::User(home) => agent.user_skill_dir(home),
            Self::Project(root) => agent.project_skill_dir(root),
        }
    }

    fn profile(&self, agent: Agent) -> RenderProfile {
        match self {
            Self::User(_) => agent.spec().user_profile,
            Self::Project(_) => agent.spec().project_profile,
        }
    }
}

fn resolve_scope(project: bool, dir: &Option<PathBuf>) -> Result<Scope> {
    if let Some(dir) = dir {
        Ok(Scope::Project(expand_tilde(dir)?))
    } else if project {
        Ok(Scope::Project(
            std::env::current_dir().context("could not determine current directory")?,
        ))
    } else {
        dirs::home_dir().map(Scope::User).ok_or_else(|| {
            anyhow::anyhow!("could not determine the home directory; use --project or --dir")
        })
    }
}

fn expand_tilde(path: &Path) -> Result<PathBuf> {
    let mut components = path.components();
    if !matches!(components.next(), Some(Component::Normal(part)) if part == "~") {
        return Ok(path.to_path_buf());
    }
    let home =
        dirs::home_dir().context("could not expand ~ because the home directory is unknown")?;
    Ok(home.join(components.as_path()))
}

#[derive(Clone)]
struct InstallTarget {
    names: Vec<&'static str>,
    profile: RenderProfile,
}

fn dedupe_targets(scope: &Scope, agents: &[Agent]) -> BTreeMap<PathBuf, InstallTarget> {
    let mut targets = BTreeMap::<PathBuf, InstallTarget>::new();
    for agent in agents {
        let profile = scope.profile(*agent);
        let target = targets
            .entry(scope.skill_dir(*agent))
            .or_insert_with(|| InstallTarget {
                names: Vec::new(),
                profile,
            });
        if target.profile != profile {
            target.profile = RenderProfile::Portable;
        }
        if !target.names.contains(&agent.spec().name) {
            target.names.push(agent.spec().name);
        }
    }
    targets
}

#[derive(Clone, Debug)]
struct Bundle {
    profile: RenderProfile,
    files: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ManagedFile {
    path: String,
    sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ManagedManifest {
    schema: u32,
    profile: RenderProfile,
    files: Vec<ManagedFile>,
}

impl Bundle {
    fn manifest(&self) -> ManagedManifest {
        ManagedManifest {
            schema: MANIFEST_SCHEMA,
            profile: self.profile,
            files: self
                .files
                .iter()
                .map(|(path, contents)| ManagedFile {
                    path: path.clone(),
                    sha256: sha256(contents.as_bytes()),
                })
                .collect(),
        }
    }

    fn files_with_manifest(&self) -> Result<BTreeMap<String, String>> {
        let mut files = self.files.clone();
        files.insert(
            MANIFEST_FILE.to_string(),
            format!("{}\n", serde_json::to_string_pretty(&self.manifest())?),
        );
        Ok(files)
    }
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn family_guide(family: &str) -> Option<&'static FamilyGuide> {
    FAMILY_GUIDES.iter().find(|guide| guide.family == family)
}

fn task_guide_for_identity(family: &str, model: &str) -> Option<TaskGuide> {
    match family {
        "minimax-h3" if model.starts_with("minimax-h3-ref2va:") => Some(TaskGuide::H3Ref2va),
        "minimax-h3" => Some(TaskGuide::H3BaseModes),
        "wan" if model.contains("-t2v-") => Some(TaskGuide::WanTextToVideo),
        "wan" if model.contains("-i2v-") || model.contains("-ti2v-") => {
            Some(TaskGuide::WanImageConditioned)
        }
        _ => None,
    }
}

fn prompting_route(
    family: &str,
    model: &str,
    explicit_task: Option<TaskGuide>,
) -> Result<Vec<&'static str>> {
    let base = family_guide(family)
        .with_context(|| format!("no canonical prompting guide for manifest family {family}"))?;
    let mut route = vec![SHARED_PROMPTING_PATH, base.path];
    let identity_task = task_guide_for_identity(family, model);
    if let Some(task) = explicit_task {
        if task.family() != family {
            anyhow::bail!("{} task guide cannot route family {family}", task.path());
        }
        if identity_task.is_some_and(|identity| identity != task) {
            anyhow::bail!(
                "{} task guide conflicts with model identity {model}",
                task.path()
            );
        }
    }
    if let Some(task) = explicit_task.or(identity_task) {
        route.push(task.path());
    }
    Ok(route)
}

fn prompting_routes_markdown(prefix: &str) -> String {
    let link = |path: &str| {
        let relative = path.strip_prefix("references/").unwrap_or(path);
        format!("[{path}]({prefix}/{relative})")
    };
    let mut output = format!(
        "- Always: {}.\n- Family bases (choose exactly one):\n",
        link(SHARED_PROMPTING_PATH)
    );
    for guide in FAMILY_GUIDES {
        output.push_str(&format!("  - `{}`: {}\n", guide.family, link(guide.path)));
    }
    output.push_str("- Task leaves (add exactly one only when applicable):\n");
    for (label, task) in [
        ("MiniMax H3 base modes", TaskGuide::H3BaseModes),
        ("MiniMax H3 Ref2VA", TaskGuide::H3Ref2va),
        ("Wan text-to-video", TaskGuide::WanTextToVideo),
        ("Wan image-conditioned", TaskGuide::WanImageConditioned),
        ("LTX-2 Dub-It", TaskGuide::Ltx2DubIt),
        ("LTX-2 text-to-audio", TaskGuide::Ltx2TextToAudio),
    ] {
        output.push_str(&format!("  - {label}: {}\n", link(task.path())));
    }
    output
}

fn render_bundle(profile: RenderProfile) -> Result<Bundle> {
    let mut files = BTreeMap::new();
    let (frontmatter, reference_prefix, agent_notes) = render_adapter(profile);
    let body = TEMPLATE_MD
        .replace("{{reference_prefix}}", reference_prefix)
        .replace("{{agent_notes}}", agent_notes)
        .replace(
            "{{prompt_routes}}",
            &prompting_routes_markdown(reference_prefix),
        );
    files.insert("SKILL.md".to_string(), format!("{frontmatter}\n\n{body}"));
    files.insert("references/cli.md".to_string(), CLI_MD.to_string());
    files.insert("references/safety.md".to_string(), SAFETY_MD.to_string());
    files.insert(
        "examples/quickstart.md".to_string(),
        EXAMPLES_MD.to_string(),
    );
    files.insert(
        SHARED_PROMPTING_PATH.to_string(),
        include_str!("references/prompting/shared.md").to_string(),
    );
    for guide in FAMILY_GUIDES {
        prompting_route(guide.family, "", None)?;
        files.insert(guide.path.to_string(), guide.contents.to_string());
    }
    for task in TASK_GUIDES {
        files.insert(task.path().to_string(), task.contents().to_string());
    }
    if profile == RenderProfile::Codex {
        files.insert("agents/openai.yaml".to_string(), "interface:\n  display_name: Mold\n  short_description: Generate and manage local AI media\n  default_prompt: Use $mold to handle this media-generation request safely.\n".to_string());
    }
    Ok(Bundle { profile, files })
}

fn render_adapter(profile: RenderProfile) -> (String, &'static str, &'static str) {
    let mut fields = vec![
        "name: mold".to_string(),
        format!("description: {DESCRIPTION}"),
    ];
    let (reference_prefix, note) = match profile {
        RenderProfile::Portable => ("references", "This portable bundle follows the Agent Skills directory and frontmatter standard."),
        RenderProfile::Claude => {
            fields.push("argument-hint: \"[prompt or mold subcommand]\"".to_string());
            ("references", "Claude Code: resolve all linked resources relative to this skill directory. No shell tools are pre-approved; retain normal permission checks.")
        }
        RenderProfile::Codex => ("references", "Codex: UI metadata lives in `agents/openai.yaml`; instructions and references remain portable and relative to this skill directory."),
        RenderProfile::Pi => {
            fields.push("compatibility: Requires the mold executable on PATH.".to_string());
            ("references", "Pi: load only the linked family or workflow reference needed for the current request.")
        }
        RenderProfile::Openclaw => {
            fields.push("metadata:\n  openclaw:\n    homepage: https://github.com/utensils/mold\n    requires:\n      bins:\n        - mold".to_string());
            ("{baseDir}/references", "OpenClaw: `{baseDir}` anchors installed resources even when this user-wide skill is invoked outside the Mold repository.")
        }
        RenderProfile::Copilot => {
            fields.push("argument-hint: \"[prompt or mold subcommand]\"".to_string());
            ("references", "GitHub Copilot: this skill intentionally omits `allowed-tools`, so commands retain the user's normal approval boundary.")
        }
        RenderProfile::Cursor => {
            fields.push("disable-model-invocation: false".to_string());
            ("references", "Cursor: resources use paths relative to the skill root and are loaded progressively.")
        }
        RenderProfile::Gemini => ("references", "Gemini CLI: activate only the relevant reference; skill activation and command execution retain Gemini's consent prompts."),
        RenderProfile::Goose => {
            fields.extend(["author: utensils".to_string(), "version: \"1\"".to_string(), "tags:\n  - image-generation\n  - video-generation\n  - local-ai".to_string()]);
            ("references", "Goose: use the Skills extension to load linked resources relative to this skill directory.")
        }
    };
    (
        format!("---\n{}\n---", fields.join("\n")),
        reference_prefix,
        note,
    )
}

fn safe_relative(path: &Path) -> bool {
    !path.as_os_str().is_empty()
        && path
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
}

fn reject_symlink_ancestors(root: &Path, relative: &Path, include_target: bool) -> Result<()> {
    let components = relative.components().collect::<Vec<_>>();
    let limit = if include_target {
        components.len()
    } else {
        components.len().saturating_sub(1)
    };
    let mut current = root.to_path_buf();
    for component in components.into_iter().take(limit) {
        let Component::Normal(part) = component else {
            anyhow::bail!("unsafe managed path {}", relative.display());
        };
        current.push(part);
        if current.is_symlink() {
            anyhow::bail!(
                "managed path {} crosses symlink {}",
                relative.display(),
                current.display()
            );
        }
    }
    Ok(())
}

fn read_manifest(dir: &Path) -> Result<Option<ManagedManifest>> {
    let path = dir.join(MANIFEST_FILE);
    if !path.exists() {
        return Ok(None);
    }
    let manifest: ManagedManifest = serde_json::from_slice(
        &std::fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?,
    )
    .with_context(|| format!("invalid Mold skill manifest at {}", path.display()))?;
    if manifest.schema != MANIFEST_SCHEMA {
        anyhow::bail!(
            "unsupported Mold skill manifest schema {} at {}",
            manifest.schema,
            path.display()
        );
    }
    let mut paths = BTreeSet::new();
    for file in &manifest.files {
        if !safe_relative(Path::new(&file.path)) || file.path == MANIFEST_FILE {
            anyhow::bail!("unsafe managed path {:?} in {}", file.path, path.display());
        }
        if !paths.insert(&file.path) {
            anyhow::bail!(
                "duplicate managed path {:?} in {}",
                file.path,
                path.display()
            );
        }
        if file.sha256.len() != 64 || !file.sha256.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            anyhow::bail!(
                "invalid sha256 for managed path {:?} in {}",
                file.path,
                path.display()
            );
        }
    }
    Ok(Some(manifest))
}

fn validate_managed_files(dir: &Path, manifest: &ManagedManifest) -> Result<()> {
    for file in &manifest.files {
        let relative = Path::new(&file.path);
        reject_symlink_ancestors(dir, relative, true)?;
        let target = dir.join(relative);
        let metadata = match std::fs::symlink_metadata(&target) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("failed to inspect {}", target.display()));
            }
        };
        if !metadata.is_file() {
            anyhow::bail!("managed path {} is not a regular file", target.display());
        }
        let actual = sha256(
            &std::fs::read(&target)
                .with_context(|| format!("failed to verify {}", target.display()))?,
        );
        if actual != file.sha256 {
            anyhow::bail!(
                "managed file {} was modified; refusing to overwrite or delete it",
                target.display()
            );
        }
    }
    Ok(())
}

fn copy_tree(source: &Path, destination: &Path) -> Result<()> {
    for entry in walkdir::WalkDir::new(source)
        .min_depth(1)
        .follow_links(false)
    {
        let entry = entry.with_context(|| format!("failed to inspect {}", source.display()))?;
        let relative = entry
            .path()
            .strip_prefix(source)
            .expect("walk entry below source");
        let target = destination.join(relative);
        if entry.file_type().is_dir() {
            std::fs::create_dir_all(&target)?;
        } else if entry.file_type().is_file() {
            if let Some(parent) = target.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::copy(entry.path(), &target)?;
        } else if entry.file_type().is_symlink() {
            copy_symlink(entry.path(), &target)?;
        }
    }
    Ok(())
}

#[cfg(unix)]
fn copy_symlink(source: &Path, destination: &Path) -> Result<()> {
    std::os::unix::fs::symlink(std::fs::read_link(source)?, destination)?;
    Ok(())
}

#[cfg(windows)]
fn copy_symlink(source: &Path, destination: &Path) -> Result<()> {
    let link = std::fs::read_link(source)?;
    if std::fs::metadata(source)?.is_dir() {
        std::os::windows::fs::symlink_dir(link, destination)?;
    } else {
        std::os::windows::fs::symlink_file(link, destination)?;
    }
    Ok(())
}

fn remove_stage_file(stage: &Path, relative: &str) -> Result<()> {
    if !safe_relative(Path::new(relative)) {
        anyhow::bail!("unsafe managed path {relative:?}");
    }
    reject_symlink_ancestors(stage, Path::new(relative), false)?;
    let target = stage.join(relative);
    if target.is_file() || target.is_symlink() {
        std::fs::remove_file(&target)?;
    }
    Ok(())
}

fn is_legacy_mold_skill(contents: &str) -> bool {
    contents.contains("name: mold") && contents.contains("# mold — Local AI Media Generation CLI")
}

fn replace_bundle(dir: &Path, bundle: &Bundle) -> Result<PathBuf> {
    let parent = dir.parent().context("skill directory has no parent")?;
    std::fs::create_dir_all(parent)
        .with_context(|| format!("failed to create {}", parent.display()))?;
    if dir.is_symlink() {
        anyhow::bail!(
            "refusing to replace symlinked skill directory {}",
            dir.display()
        );
    }
    let installed_manifest = read_manifest(dir)?;
    if let Some(manifest) = &installed_manifest {
        validate_managed_files(dir, manifest)?;
    }
    let staging = tempfile::tempdir_in(parent)
        .with_context(|| format!("failed to stage skill beside {}", dir.display()))?;
    let staged_dir = staging.path().join(SKILL_DIR_NAME);
    std::fs::create_dir(&staged_dir)?;
    if dir.exists() {
        copy_tree(dir, &staged_dir)?;
    }
    if let Some(manifest) = installed_manifest {
        for file in manifest.files {
            remove_stage_file(&staged_dir, &file.path)?;
        }
        remove_stage_file(&staged_dir, MANIFEST_FILE)?;
    } else if staged_dir.join("SKILL.md").exists() {
        let skill = std::fs::read_to_string(staged_dir.join("SKILL.md"))?;
        if !is_legacy_mold_skill(&skill) {
            anyhow::bail!(
                "refusing to overwrite unmanaged skill at {}; remove or rename it first",
                dir.display()
            );
        }
        for legacy in ["SKILL.md", "references/model-prompting.md"] {
            remove_stage_file(&staged_dir, legacy)?;
        }
    }
    for (relative, contents) in bundle.files_with_manifest()? {
        let relative_path = Path::new(&relative);
        if !safe_relative(relative_path) {
            anyhow::bail!("renderer produced unsafe path {relative:?}");
        }
        reject_symlink_ancestors(&staged_dir, relative_path, true)?;
        let target = staged_dir.join(relative_path);
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&target, contents)
            .with_context(|| format!("failed to stage {}", target.display()))?;
    }
    let backup = parent.join(format!(".{SKILL_DIR_NAME}.backup-{}", uuid::Uuid::new_v4()));
    let had_existing = dir.exists();
    if had_existing {
        std::fs::rename(dir, &backup)
            .with_context(|| format!("failed to stage existing {}", dir.display()))?;
    }
    if let Err(error) = std::fs::rename(&staged_dir, dir) {
        if had_existing {
            let _ = std::fs::rename(&backup, dir);
        }
        return Err(error)
            .with_context(|| format!("failed to atomically install {}", dir.display()));
    }
    if had_existing {
        let _ = std::fs::remove_dir_all(&backup);
    }
    Ok(dir.join("SKILL.md"))
}

fn remove_empty_parents(mut path: PathBuf, stop: &Path) {
    while path.starts_with(stop) && path != stop {
        if std::fs::remove_dir(&path).is_err() {
            break;
        }
        let Some(parent) = path.parent() else { break };
        path = parent.to_path_buf();
    }
}

fn uninstall_bundle(dir: &Path) -> Result<usize> {
    if dir.is_symlink() {
        anyhow::bail!(
            "refusing to uninstall through symlinked skill directory {}",
            dir.display()
        );
    }
    let installed_manifest = read_manifest(dir)?;
    if let Some(manifest) = &installed_manifest {
        validate_managed_files(dir, manifest)?;
    }
    let managed = if let Some(manifest) = installed_manifest {
        manifest
            .files
            .into_iter()
            .map(|file| file.path)
            .collect::<Vec<_>>()
    } else {
        let legacy = std::fs::read_to_string(dir.join("SKILL.md")).unwrap_or_default();
        if is_legacy_mold_skill(&legacy) {
            vec![
                "SKILL.md".to_string(),
                "references/model-prompting.md".to_string(),
            ]
        } else {
            Vec::new()
        }
    };
    let mut removed = 0;
    for relative in managed {
        reject_symlink_ancestors(dir, Path::new(&relative), false)?;
        let target = dir.join(&relative);
        if target.is_file() || target.is_symlink() {
            std::fs::remove_file(&target)
                .with_context(|| format!("failed to remove {}", target.display()))?;
            removed += 1;
            if let Some(parent) = target.parent() {
                remove_empty_parents(parent.to_path_buf(), dir);
            }
        }
    }
    let manifest = dir.join(MANIFEST_FILE);
    if manifest.is_file() {
        std::fs::remove_file(&manifest)?;
        removed += 1;
    }
    let _ = std::fs::remove_dir(dir);
    Ok(removed)
}

fn bundle_status(dir: &Path, profile: RenderProfile) -> String {
    let path = display_path(dir);
    if !dir.join("SKILL.md").is_file() {
        return path;
    }
    let Ok(Some(installed)) = read_manifest(dir) else {
        return format!("{path} [legacy]");
    };
    let Ok(expected) = render_bundle(profile) else {
        return format!("{path} [unreadable]");
    };
    let current = expected.manifest();
    if installed.profile == current.profile
        && installed.files.len() == current.files.len()
        && installed.files.iter().all(|file| {
            current
                .files
                .iter()
                .any(|expected| expected.path == file.path && expected.sha256 == file.sha256)
                && std::fs::read(dir.join(&file.path))
                    .map(|bytes| sha256(&bytes) == file.sha256)
                    .unwrap_or(false)
        })
    {
        format!("{path} [current]")
    } else {
        format!("{path} [outdated]")
    }
}

fn display_path(path: &Path) -> String {
    if let Some(home) = dirs::home_dir() {
        if let Ok(rest) = path.strip_prefix(home) {
            return format!("~/{}", rest.display());
        }
    }
    path.display().to_string()
}

fn selected_agents(args: &SkillInstallArgs) -> Result<Vec<Agent>> {
    if !args.agents.is_empty() {
        return Ok(args.agents.clone());
    }
    if args.detected {
        let home = dirs::home_dir().context("could not determine home directory for --detected")?;
        let detected = AGENTS
            .iter()
            .map(|spec| spec.agent)
            .filter(|agent| *agent != Agent::Agents && agent.is_detected(&home))
            .collect::<Vec<_>>();
        if detected.is_empty() {
            anyhow::bail!(
                "no supported agents detected; name agents explicitly, use `agents`, or use --all"
            );
        }
        return Ok(detected);
    }
    if args.all {
        return Ok(AGENTS.iter().map(|spec| spec.agent).collect());
    }
    anyhow::bail!("name at least one agent, or pass --detected or --all")
}

pub fn run(args: &SkillArgs) -> Result<()> {
    match &args.command {
        SkillCommand::List(args) => list(args),
        SkillCommand::Install(args) => install(args),
        SkillCommand::Uninstall(args) => uninstall(args),
        SkillCommand::Show(args) => show(args),
    }
}

fn install(args: &SkillInstallArgs) -> Result<()> {
    let scope = resolve_scope(args.project, &args.dir)?;
    for (dir, target) in dedupe_targets(&scope, &selected_agents(args)?) {
        let installed = replace_bundle(&dir, &render_bundle(target.profile)?)?;
        eprintln!(
            "Installed mold skill: {} ({}, {} profile)",
            display_path(&installed),
            target.names.join(", "),
            target.profile.name()
        );
    }
    Ok(())
}

fn uninstall(args: &SkillUninstallArgs) -> Result<()> {
    let scope = resolve_scope(args.project, &args.dir)?;
    let agents = if args.agents.is_empty() {
        AGENTS.iter().map(|spec| spec.agent).collect::<Vec<_>>()
    } else {
        args.agents.clone()
    };
    let mut removed = 0;
    for (dir, _) in dedupe_targets(&scope, &agents) {
        let count = uninstall_bundle(&dir)?;
        if count > 0 {
            eprintln!(
                "Removed {count} managed Mold skill files from {}",
                display_path(&dir)
            );
        }
        removed += count;
        if dir.exists() {
            eprintln!("Left non-empty directory in place: {}", display_path(&dir));
        }
    }
    if removed == 0 {
        eprintln!("No installed Mold skill found in the selected scope.");
    }
    Ok(())
}

fn show(args: &SkillShowArgs) -> Result<()> {
    if !safe_relative(&args.file) {
        anyhow::bail!("bundle file must be a safe relative path");
    }
    let bundle = render_bundle(args.agent.spec().user_profile)?;
    let files = bundle.files_with_manifest()?;
    let key = args.file.to_string_lossy();
    let contents = files.get(key.as_ref()).with_context(|| {
        format!(
            "no rendered {} bundle file {}; available: {}",
            args.agent.spec().name,
            args.file.display(),
            files.keys().cloned().collect::<Vec<_>>().join(", ")
        )
    })?;
    print!("{contents}");
    Ok(())
}

fn list(args: &SkillListArgs) -> Result<()> {
    use comfy_table::{presets::ASCII_FULL, presets::UTF8_FULL, ContentArrangement, Table};
    let home = dirs::home_dir();
    let project = std::env::current_dir().ok();
    let mut table = Table::new();
    table
        .load_preset(if args.ascii { ASCII_FULL } else { UTF8_FULL })
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(["Agent", "Name", "Detected", "User skill", "Project skill"]);
    for spec in AGENTS {
        let detected = home
            .as_ref()
            .map(|home| {
                if spec.agent.is_detected(home) {
                    "yes"
                } else {
                    "no"
                }
            })
            .unwrap_or("-");
        let user = home
            .as_ref()
            .map(|home| bundle_status(&spec.agent.user_skill_dir(home), spec.user_profile))
            .unwrap_or_else(|| "-".to_string());
        let project = project
            .as_ref()
            .map(|root| bundle_status(&spec.agent.project_skill_dir(root), spec.project_profile))
            .unwrap_or_else(|| "-".to_string());
        table.add_row([spec.label, spec.name, detected, &user, &project]);
    }
    println!("{table}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    use std::collections::BTreeSet;

    #[derive(Deserialize)]
    struct AgentFixture {
        agent: String,
        profile: String,
        user: String,
        project: String,
    }

    #[test]
    fn target_matrix_and_render_profiles_match_fixture() {
        let fixtures: Vec<AgentFixture> =
            serde_json::from_str(include_str!("fixtures/agent-bundles.json")).unwrap();
        assert_eq!(fixtures.len(), AGENTS.len());
        let home = Path::new("/home/u");
        let project = Path::new("/project");
        for fixture in fixtures {
            let spec = AGENTS
                .iter()
                .find(|spec| spec.name == fixture.agent)
                .unwrap();
            assert_eq!(spec.user_profile.name(), fixture.profile);
            assert_eq!(
                spec.agent.user_skill_dir(home),
                home.join(fixture.user).join("mold")
            );
            assert_eq!(
                spec.agent.project_skill_dir(project),
                project.join(fixture.project).join("mold")
            );
        }
    }

    #[test]
    fn every_agent_profile_renders_valid_discoverable_bundle() {
        let profiles = AGENTS
            .iter()
            .flat_map(|spec| [spec.user_profile, spec.project_profile])
            .collect::<BTreeSet<_>>();
        for profile in profiles {
            let bundle = render_bundle(profile).unwrap();
            let skill = &bundle.files["SKILL.md"];
            assert!(skill.starts_with("---\nname: mold\ndescription: "));
            assert!(skill.contains("\n---\n\n# Mold media generation"));
            assert!(
                !skill.contains("{{"),
                "unrendered placeholder for {}",
                profile.name()
            );
            assert!(bundle.files.contains_key("examples/quickstart.md"));
            assert!(bundle.files.contains_key("references/safety.md"));
            assert_eq!(
                bundle.files.contains_key("agents/openai.yaml"),
                profile == RenderProfile::Codex
            );
        }
        let openclaw = render_bundle(RenderProfile::Openclaw).unwrap();
        assert!(openclaw.files["SKILL.md"].contains("{baseDir}/references"));
        assert!(openclaw.files["SKILL.md"]
            .contains("metadata:\n  openclaw:\n    homepage: https://github.com/utensils/mold\n"));
        assert!(!openclaw.files["SKILL.md"].contains("\nhomepage:"));
        assert!(openclaw.files["SKILL.md"].contains("requires:\n      bins:\n        - mold"));
        let copilot = render_bundle(RenderProfile::Copilot).unwrap();
        let copilot_frontmatter = copilot.files["SKILL.md"].split("\n---\n").next().unwrap();
        assert!(!copilot_frontmatter.contains("allowed-tools:"));
    }

    #[test]
    fn shared_paths_use_portable_intersection_profile() {
        let scope = Scope::Project(PathBuf::from("/project"));
        let targets = dedupe_targets(&scope, &[Agent::Codex, Agent::Cursor, Agent::Gemini]);
        let target = targets
            .get(Path::new("/project/.agents/skills/mold"))
            .unwrap();
        assert_eq!(target.profile, RenderProfile::Portable);
        assert_eq!(target.names, vec!["codex", "cursor", "gemini"]);
    }

    #[test]
    fn prompting_routes_cover_manifest_families_tasks_and_identities() {
        let manifest_families = mold_core::manifest::known_manifests()
            .iter()
            .filter(|manifest| manifest.is_generation_model() || manifest.is_upscaler())
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

        for manifest in mold_core::manifest::known_manifests()
            .iter()
            .filter(|manifest| manifest.is_generation_model() || manifest.is_upscaler())
        {
            let route = prompting_route(&manifest.family, &manifest.name, None).unwrap();
            assert_eq!(route[0], SHARED_PROMPTING_PATH, "{}", manifest.name);
            assert_eq!(
                route
                    .iter()
                    .filter(|path| path.contains("/families/"))
                    .count(),
                1,
                "{}",
                manifest.name
            );
            let leaves = route
                .iter()
                .filter(|path| !path.contains("/families/") && **path != SHARED_PROMPTING_PATH)
                .count();
            match manifest.family.as_str() {
                "minimax-h3" | "wan" => assert_eq!(leaves, 1, "{}", manifest.name),
                _ => assert_eq!(leaves, 0, "{}", manifest.name),
            }
        }

        assert_eq!(
            prompting_route("minimax-h3", "minimax-h3-fl2va:q8", None)
                .unwrap()
                .last(),
            Some(&TaskGuide::H3BaseModes.path())
        );
        assert_eq!(
            prompting_route("minimax-h3", "minimax-h3-ref2va:q8", None)
                .unwrap()
                .last(),
            Some(&TaskGuide::H3Ref2va.path())
        );
        assert_eq!(
            prompting_route("wan", "wan22-t2v-a14b:q4", None)
                .unwrap()
                .last(),
            Some(&TaskGuide::WanTextToVideo.path())
        );
        assert_eq!(
            prompting_route("wan", "wan22-ti2v-5b:q8", None)
                .unwrap()
                .last(),
            Some(&TaskGuide::WanImageConditioned.path())
        );
        assert_eq!(
            prompting_route("ltx2", "ltx-2.5-22b:q6", Some(TaskGuide::Ltx2DubIt))
                .unwrap()
                .last(),
            Some(&TaskGuide::Ltx2DubIt.path())
        );
        assert_eq!(
            prompting_route("ltx2", "ltx-2.5-22b:q6", Some(TaskGuide::Ltx2TextToAudio))
                .unwrap()
                .last(),
            Some(&TaskGuide::Ltx2TextToAudio.path())
        );
        assert!(prompting_route("flux", "flux-dev:q8", Some(TaskGuide::Ltx2DubIt)).is_err());
        assert!(prompting_route(
            "minimax-h3",
            "minimax-h3-ref2va:q8",
            Some(TaskGuide::H3BaseModes),
        )
        .is_err());
    }

    #[test]
    fn prompting_tree_is_identity_agnostic_and_byte_identical_across_renderers() {
        let portable = render_bundle(RenderProfile::Portable).unwrap();
        let canonical = portable
            .files
            .iter()
            .filter(|(path, _)| path.starts_with("references/prompting/"))
            .map(|(path, contents)| (path.clone(), sha256(contents.as_bytes())))
            .collect::<BTreeMap<_, _>>();
        for profile in [
            RenderProfile::Claude,
            RenderProfile::Codex,
            RenderProfile::Pi,
            RenderProfile::Openclaw,
            RenderProfile::Copilot,
            RenderProfile::Cursor,
            RenderProfile::Gemini,
            RenderProfile::Goose,
        ] {
            let bundle = render_bundle(profile).unwrap();
            let hashes = bundle
                .files
                .iter()
                .filter(|(path, _)| path.starts_with("references/prompting/"))
                .map(|(path, contents)| (path.clone(), sha256(contents.as_bytes())))
                .collect::<BTreeMap<_, _>>();
            assert_eq!(hashes, canonical, "{} forked prompting", profile.name());
        }

        for path in canonical.keys() {
            assert!(!path.contains(':'), "identity tag leaked into {path}");
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
    fn every_direct_relative_skill_link_resolves_in_each_bundle() {
        for profile in [
            RenderProfile::Portable,
            RenderProfile::Claude,
            RenderProfile::Codex,
            RenderProfile::Pi,
            RenderProfile::Openclaw,
            RenderProfile::Copilot,
            RenderProfile::Cursor,
            RenderProfile::Gemini,
            RenderProfile::Goose,
        ] {
            let bundle = render_bundle(profile).unwrap();
            let skill = &bundle.files["SKILL.md"];
            for tail in skill.split("](").skip(1) {
                let target = tail.split(')').next().unwrap();
                if target.starts_with("http") {
                    continue;
                }
                let relative = target.strip_prefix("{baseDir}/").unwrap_or(target);
                assert!(
                    bundle.files.contains_key(relative),
                    "{} has broken link {target}",
                    profile.name()
                );
            }
        }
    }

    #[test]
    fn install_upgrade_and_uninstall_follow_managed_inventory() {
        let root = tempfile::tempdir().unwrap();
        let dir = root.path().join("mold");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("notes.md"), "keep").unwrap();
        std::fs::write(
            dir.join("SKILL.md"),
            "---\nname: mold\n---\n# mold — Local AI Media Generation CLI\n",
        )
        .unwrap();
        std::fs::create_dir_all(dir.join("references")).unwrap();
        std::fs::write(dir.join("references/model-prompting.md"), "legacy").unwrap();
        replace_bundle(&dir, &render_bundle(RenderProfile::Codex).unwrap()).unwrap();
        assert!(dir.join("agents/openai.yaml").is_file());
        assert!(!dir.join("references/model-prompting.md").exists());
        assert_eq!(
            std::fs::read_to_string(dir.join("notes.md")).unwrap(),
            "keep"
        );
        assert!(bundle_status(&dir, RenderProfile::Codex).ends_with("[current]"));
        replace_bundle(&dir, &render_bundle(RenderProfile::Portable).unwrap()).unwrap();
        assert!(
            !dir.join("agents/openai.yaml").exists(),
            "stale managed file removed on profile change"
        );
        assert_eq!(
            std::fs::read_to_string(dir.join("notes.md")).unwrap(),
            "keep"
        );
        let removed = uninstall_bundle(&dir).unwrap();
        assert!(removed > 10);
        assert_eq!(
            std::fs::read_to_string(dir.join("notes.md")).unwrap(),
            "keep"
        );
        assert!(!dir.join("SKILL.md").exists());
        assert!(!dir.join(MANIFEST_FILE).exists());
    }

    #[test]
    fn modified_managed_file_blocks_upgrade_and_uninstall() {
        let root = tempfile::tempdir().unwrap();
        let dir = root.path().join("mold");
        replace_bundle(&dir, &render_bundle(RenderProfile::Portable).unwrap()).unwrap();
        std::fs::write(dir.join("SKILL.md"), "user customization").unwrap();

        let upgrade =
            replace_bundle(&dir, &render_bundle(RenderProfile::Codex).unwrap()).unwrap_err();
        assert!(upgrade.to_string().contains("was modified"));
        let uninstall = uninstall_bundle(&dir).unwrap_err();
        assert!(uninstall.to_string().contains("was modified"));
        assert_eq!(
            std::fs::read_to_string(dir.join("SKILL.md")).unwrap(),
            "user customization"
        );
        assert!(dir.join(MANIFEST_FILE).is_file());
    }

    #[test]
    fn tampered_manifest_cannot_claim_a_user_file() {
        let root = tempfile::tempdir().unwrap();
        let dir = root.path().join("mold");
        replace_bundle(&dir, &render_bundle(RenderProfile::Portable).unwrap()).unwrap();
        std::fs::write(dir.join("notes.md"), "keep this").unwrap();
        let mut manifest = read_manifest(&dir).unwrap().unwrap();
        manifest.files[0].path = "notes.md".to_string();
        std::fs::write(
            dir.join(MANIFEST_FILE),
            format!("{}\n", serde_json::to_string_pretty(&manifest).unwrap()),
        )
        .unwrap();

        let error = uninstall_bundle(&dir).unwrap_err();
        assert!(error.to_string().contains("was modified"));
        assert_eq!(
            std::fs::read_to_string(dir.join("notes.md")).unwrap(),
            "keep this"
        );
        assert!(dir.join("SKILL.md").is_file());
    }

    #[test]
    fn duplicate_manifest_paths_are_rejected_before_deletion() {
        let root = tempfile::tempdir().unwrap();
        let dir = root.path().join("mold");
        replace_bundle(&dir, &render_bundle(RenderProfile::Portable).unwrap()).unwrap();
        let mut manifest = read_manifest(&dir).unwrap().unwrap();
        manifest.files.push(manifest.files[0].clone());
        std::fs::write(
            dir.join(MANIFEST_FILE),
            format!("{}\n", serde_json::to_string_pretty(&manifest).unwrap()),
        )
        .unwrap();

        let error = uninstall_bundle(&dir).unwrap_err();
        assert!(error.to_string().contains("duplicate managed path"));
        assert!(dir.join("SKILL.md").is_file());
    }

    #[test]
    fn managed_paths_are_rejected_before_filesystem_use() {
        for path in ["", "/tmp/x", "../x", "references/../x", "./x"] {
            assert!(!safe_relative(Path::new(path)), "accepted unsafe {path:?}");
        }
        assert!(safe_relative(Path::new(
            "references/prompting/families/flux2.md"
        )));
    }

    #[cfg(unix)]
    #[test]
    fn managed_paths_refuse_symlinked_ancestors() {
        let root = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        std::os::unix::fs::symlink(outside.path(), root.path().join("references")).unwrap();
        let error = reject_symlink_ancestors(
            root.path(),
            Path::new("references/prompting/families/flux2.md"),
            true,
        )
        .unwrap_err();
        assert!(error.to_string().contains("crosses symlink"));
    }

    #[test]
    fn uninstall_does_not_claim_an_unmanaged_skill_without_a_manifest() {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join("SKILL.md"), "user-owned").unwrap();
        assert_eq!(uninstall_bundle(root.path()).unwrap(), 0);
        assert_eq!(
            std::fs::read_to_string(root.path().join("SKILL.md")).unwrap(),
            "user-owned"
        );
    }

    #[test]
    fn install_refuses_to_overwrite_an_unmanaged_skill() {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join("SKILL.md"), "user-owned").unwrap();
        let error = replace_bundle(
            root.path(),
            &render_bundle(RenderProfile::Portable).unwrap(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("unmanaged skill"));
        assert_eq!(
            std::fs::read_to_string(root.path().join("SKILL.md")).unwrap(),
            "user-owned"
        );
    }

    #[cfg(unix)]
    #[test]
    fn uninstall_refuses_to_follow_a_symlinked_skill_root() {
        let root = tempfile::tempdir().unwrap();
        let external = tempfile::tempdir().unwrap();
        replace_bundle(
            external.path(),
            &render_bundle(RenderProfile::Portable).unwrap(),
        )
        .unwrap();
        let link = root.path().join("mold");
        std::os::unix::fs::symlink(external.path(), &link).unwrap();

        let error = uninstall_bundle(&link).unwrap_err();
        assert!(error.to_string().contains("symlinked skill directory"));
        assert!(link.is_symlink());
        assert!(external.path().join("SKILL.md").is_file());
        assert!(external.path().join(MANIFEST_FILE).is_file());
    }

    #[test]
    fn documented_quickstart_commands_parse_with_the_cli() {
        let examples: &[&[&str]] = &[
            &["list"],
            &["info", "flux2-klein:q8"],
            &[
                "run",
                "flux2-klein:q8",
                "A red fox in falling snow",
                "--seed",
                "42",
                "--output",
                "fox.png",
            ],
            &[
                "run",
                "qwen-image-edit-2511:q8",
                "Change the chair to red leather",
                "--image",
                "chair.png",
                "--output",
                "edited.png",
            ],
            &[
                "upscale",
                "input.png",
                "--model",
                "real-esrgan-x4plus:fp16",
                "--output",
                "output-4x.png",
            ],
            &["queue", "list"],
            &["queue", "cancel", "job-abc123"],
            &["server", "status"],
            &["skill", "show", "codex"],
            &[
                "skill",
                "show",
                "codex",
                "references/prompting/families/flux2.md",
            ],
        ];
        for argv in examples {
            let rendered = format!(
                "mold {}",
                argv.iter()
                    .map(|arg| if arg.contains(' ') {
                        format!("\"{arg}\"")
                    } else {
                        (*arg).to_string()
                    })
                    .collect::<Vec<_>>()
                    .join(" ")
            );
            assert!(
                EXAMPLES_MD.contains(&rendered),
                "example fixture missing {rendered}"
            );
            crate::Cli::try_parse_from(std::iter::once("mold").chain(argv.iter().copied()))
                .unwrap_or_else(|error| panic!("invalid example {rendered}: {error}"));
        }
    }
}
