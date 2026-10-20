//! Agent-specific skill bundle rendering and managed installation.

use anyhow::{Context, Result};
use clap::{ArgGroup, Parser, Subcommand, ValueEnum};
use mold_core::prompting;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Component, Path, PathBuf};

mod claude_desktop;

const SKILL_DIR_NAME: &str = "mold";
const MANIFEST_FILE: &str = ".mold-skill.json";
const MANIFEST_SCHEMA: u32 = 1;
const DESCRIPTION: &str = "Generate and manage AI images and video with the mold CLI. Use when asked to create images or clips, transform source media, operate local or remote Mold servers, manage models and queues, inspect galleries, configure GPU inference, or automate a Mold CLI, REST, or MCP workflow.";
const TEMPLATE_MD: &str = include_str!("template.md");
const CLI_MD: &str = include_str!("references/cli.md");
const SAFETY_MD: &str = include_str!("references/safety.md");
const EXAMPLES_MD: &str = include_str!("examples/quickstart.md");

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
    /// claude-desktop: where to write the skill ZIP (default:
    /// ~/Downloads/mold-skill.zip)
    #[arg(long, short = 'o', value_name = "FILE")]
    output: Option<PathBuf>,
    /// claude-desktop: only write the ZIP; leave claude_desktop_config.json alone
    #[arg(long)]
    no_connector: bool,
    /// claude-desktop: do not open the Skills page or reveal the ZIP
    #[arg(long)]
    no_open: bool,
    /// claude-desktop: replace an --output file or an mcpServers.mold entry
    /// that mold did not write
    #[arg(long)]
    force: bool,
}

impl SkillArgs {
    /// True for `mold skill install claude-desktop`, the one skill command
    /// the global `--host` applies to: it pins the Desktop connector.
    pub fn takes_host(&self) -> bool {
        matches!(&self.command, SkillCommand::Install(args) if args.agents.contains(&Agent::ClaudeDesktop))
    }
}

impl SkillInstallArgs {
    /// The Claude Desktop flags someone passed, to refuse them when
    /// `claude-desktop` is not being installed.
    fn desktop_flags(&self) -> Vec<&'static str> {
        [
            (self.output.is_some(), "--output"),
            (self.no_connector, "--no-connector"),
            (self.no_open, "--no-open"),
            (self.force, "--force"),
        ]
        .into_iter()
        .filter_map(|(set, flag)| set.then_some(flag))
        .collect()
    }
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
    #[arg(default_value = "SKILL.md", add = clap_complete::engine::ArgValueCandidates::new(complete_bundle_file))]
    file: PathBuf,
}

/// Completion candidates for `mold skill show <FILE>`: the bundle's own file
/// list, which is compiled in, so this answers without touching the disk or
/// the network. The portable profile's tree is the right one to offer —
/// `references/prompting/**` is byte-identical across every profile, and the
/// only per-profile difference is the SKILL.md body, whose path is the same.
fn complete_bundle_file() -> Vec<clap_complete::engine::CompletionCandidate> {
    render_bundle(RenderProfile::Portable)
        .map(|bundle| {
            bundle
                .files
                .keys()
                .map(clap_complete::engine::CompletionCandidate::new)
                .collect()
        })
        .unwrap_or_default()
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
    /// Claude Desktop and claude.ai: a skill ZIP to upload plus the local
    /// `mold mcp` connector. Not a skill directory, so it has no `AgentSpec`.
    ClaudeDesktop,
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
    #[serde(rename = "claude-desktop")]
    ClaudeDesktop,
}

impl RenderProfile {
    #[cfg(test)]
    const ALL: [Self; 10] = [
        Self::Portable,
        Self::Claude,
        Self::Codex,
        Self::Pi,
        Self::Openclaw,
        Self::Copilot,
        Self::Cursor,
        Self::Gemini,
        Self::Goose,
        Self::ClaudeDesktop,
    ];

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
            Self::ClaudeDesktop => "claude-desktop",
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
    /// The profile `mold skill show` renders for this agent.
    fn show_profile(self) -> RenderProfile {
        match self {
            Self::ClaudeDesktop => RenderProfile::ClaudeDesktop,
            agent => agent.spec().user_profile,
        }
    }

    fn cli_name(self) -> &'static str {
        match self {
            Self::ClaudeDesktop => "claude-desktop",
            agent => agent.spec().name,
        }
    }

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

const PROMPTING_PREFIX: &str = "references/prompting/";

fn prompting_routes_markdown(prefix: &str) -> String {
    let link = |path: &str| format!("[{PROMPTING_PREFIX}{path}]({prefix}/prompting/{path})");
    let mut output = format!(
        "- Always: {}.\n- Family bases (choose exactly one):\n",
        link(prompting::SHARED_PATH)
    );
    for guide in prompting::FAMILY_GUIDES {
        output.push_str(&format!("  - `{}`: {}\n", guide.family, link(guide.path)));
    }
    output.push_str("- Task leaves (add exactly one only when applicable):\n");
    for leaf in prompting::TASK_LEAVES {
        output.push_str(&format!("  - {}: {}\n", leaf.label, link(leaf.path)));
    }
    if !prompting::MODEL_LEAVES.is_empty() {
        output.push_str("- Model leaves (add only when the exact checkpoint is listed):\n");
        for leaf in prompting::MODEL_LEAVES {
            output.push_str(&format!(
                "  - {} (`{}`): {}\n",
                leaf.label,
                leaf.models.join("`, `"),
                link(leaf.path)
            ));
        }
    }
    output
}

fn render_bundle(profile: RenderProfile) -> Result<Bundle> {
    let mut files = BTreeMap::new();
    let (frontmatter, reference_prefix, agent_notes) = render_adapter(profile);
    let body = TEMPLATE_MD
        .replace("{{surface_preamble}}", surface_preamble(profile))
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
    for guide in prompting::FAMILY_GUIDES {
        prompting::route(guide.family, None, None).map_err(|error| anyhow::anyhow!("{error}"))?;
    }
    for (path, contents) in prompting::rendered_files() {
        files.insert(format!("{PROMPTING_PREFIX}{path}"), contents);
    }
    if profile == RenderProfile::Codex {
        files.insert("agents/openai.yaml".to_string(), "interface:\n  display_name: Mold\n  short_description: Generate and manage local AI media\n  default_prompt: Use $mold to handle this media-generation request safely.\n".to_string());
    }
    Ok(Bundle { profile, files })
}

/// Text that must come before everything else in `SKILL.md`. Only Claude
/// Desktop has any: its chat runs in a cloud sandbox with no `mold` binary,
/// so every CLI example below it describes what the connector does instead.
fn surface_preamble(profile: RenderProfile) -> &'static str {
    match profile {
        RenderProfile::ClaudeDesktop => CLAUDE_DESKTOP_PREAMBLE,
        _ => "",
    }
}

/// Tools this preamble names; a test pins each to `mold mcp`'s tool list.
#[cfg(test)]
const CLAUDE_DESKTOP_TOOLS: &[&str] = &[
    "server_status",
    "list_models",
    "generate_image",
    "generate_image_async",
    "generation_status",
    "list_gallery",
    "get_gallery_image",
    "expand_prompt",
];

const CLAUDE_DESKTOP_PREAMBLE: &str = "\
## In Claude Desktop and claude.ai

You cannot run the `mold` command here: this conversation's code sandbox has
no `mold` binary and cannot reach the user's machines. Work through the
**mold connector** instead, which runs `mold mcp` on the user's computer and
talks to their default mold host:

- `server_status` and `list_models` first, to see the host and what it can run.
- `generate_image` for a quick still; `generate_image_async` then
  `generation_status` for video, large batches or anything slow.
- `list_gallery` and `get_gallery_image` to find and show existing prints.
- `expand_prompt` to apply the prompting guides below to a short idea.

If no mold tools are available, ask the user to run
`mold skill install claude-desktop` in a terminal on their computer and restart
Claude Desktop. Pairing a host (`mold host add`) and anything else marked
`mold …` below also happens in their terminal, never in this sandbox. Never
ask the user to paste an API key or pairing link into the chat.

The command examples in this skill describe the CLI the connector wraps; use
them to choose models, sizes and prompts, then call the matching tool.

";

fn render_adapter(profile: RenderProfile) -> (String, &'static str, &'static str) {
    let mut fields = vec![
        "name: mold".to_string(),
        format!("description: {DESCRIPTION}"),
    ];
    let (reference_prefix, note) = match profile {
        RenderProfile::Portable => ("references", "This portable bundle follows the Agent Skills directory and frontmatter standard."),
        RenderProfile::Claude => {
            fields.push("argument-hint: \"[prompt or mold subcommand]\"".to_string());
            ("references", "Claude Code: resolve all linked resources relative to this skill directory. No shell tools are pre-approved; retain normal permission checks. When the user must run a command in their own terminal (pairing with `mold host add`), suggest typing `! mold host add` at the Claude Code prompt so it runs in this session and the link never enters chat.")
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
        RenderProfile::ClaudeDesktop => ("references", "Claude Desktop: this skill was uploaded under Customize > Skills; generation goes through the mold connector tools described at the top, never a sandbox shell."),
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
        let mut detected = AGENTS
            .iter()
            .map(|spec| spec.agent)
            .filter(|agent| *agent != Agent::Agents && agent.is_detected(&home))
            .collect::<Vec<_>>();
        if auto_includes_desktop(args, &home) {
            detected.push(Agent::ClaudeDesktop);
        }
        if detected.is_empty() {
            anyhow::bail!(
                "no supported agents detected; name agents explicitly, use `agents`, or use --all"
            );
        }
        return Ok(detected);
    }
    if args.all {
        let mut all = AGENTS.iter().map(|spec| spec.agent).collect::<Vec<_>>();
        if dirs::home_dir().is_some_and(|home| auto_includes_desktop(args, &home)) {
            all.push(Agent::ClaudeDesktop);
        }
        return Ok(all);
    }
    anyhow::bail!("name at least one agent, or pass --detected or --all")
}

/// `host` is a typed global `--host`, which only `install claude-desktop`
/// accepts (see [`SkillArgs::takes_host`]).
pub fn run(args: &SkillArgs, host: Option<&str>) -> Result<()> {
    match &args.command {
        SkillCommand::List(args) => list(args),
        SkillCommand::Install(args) => install(args, host),
        SkillCommand::Uninstall(args) => uninstall(args),
        SkillCommand::Show(args) => show(args),
    }
}

fn install(args: &SkillInstallArgs, host: Option<&str>) -> Result<()> {
    let (desktop, agents) = split_desktop(selected_agents(args)?);
    if desktop && (args.project || args.dir.is_some()) {
        anyhow::bail!(
            "claude-desktop installs to your Claude account, not a project; drop --project/--dir"
        );
    }
    let stray = args.desktop_flags();
    if !desktop && !stray.is_empty() {
        anyhow::bail!("{} only apply to claude-desktop", stray.join(", "));
    }
    if !agents.is_empty() {
        install_directories(args, &agents)?;
    }
    if desktop {
        // Only a named `claude-desktop` opens the browser; `--detected` and
        // `--all` print the same steps without taking over the screen.
        let named = args.agents.contains(&Agent::ClaudeDesktop);
        install_claude_desktop(args, host, named && !args.no_open)?;
    }
    Ok(())
}

/// Pull `claude-desktop` out of an agent list: it is not a skill directory.
fn split_desktop(agents: Vec<Agent>) -> (bool, Vec<Agent>) {
    let desktop = agents.contains(&Agent::ClaudeDesktop);
    let rest = agents
        .into_iter()
        .filter(|agent| *agent != Agent::ClaudeDesktop)
        .collect();
    (desktop, rest)
}

fn desktop_config_dir(home: &Path) -> Option<PathBuf> {
    claude_desktop::config_dir(
        home,
        std::env::var_os("APPDATA").map(PathBuf::from).as_deref(),
    )
}

/// `--detected` and `--all` add Claude Desktop only in user scope: it lives
/// on the Claude account, so a project install never includes it.
fn auto_includes_desktop(args: &SkillInstallArgs, home: &Path) -> bool {
    !args.project && args.dir.is_none() && desktop_detected(home)
}

fn desktop_detected(home: &Path) -> bool {
    desktop_config_dir(home).is_some_and(|dir| dir.is_dir())
}

fn install_claude_desktop(args: &SkillInstallArgs, host: Option<&str>, open: bool) -> Result<()> {
    let home = dirs::home_dir().context("could not determine the home directory")?;
    let zip = match &args.output {
        Some(path) => expand_tilde(path)?,
        None => claude_desktop::default_zip_path(
            dirs::download_dir(),
            &std::env::current_dir().context("could not determine current directory")?,
        ),
    };
    let bundle = render_bundle(RenderProfile::ClaudeDesktop)?;
    claude_desktop::write_zip(&bundle.files_with_manifest()?, &zip, args.force)?;
    eprintln!("✓ Wrote the Claude Desktop skill: {}", display_path(&zip));

    if args.no_connector {
        eprintln!("  Skipped the connector (--no-connector).");
    } else {
        match desktop_config_dir(&home) {
            None => eprintln!(
                "  Claude Desktop has no build for this platform, so no connector was added. The ZIP still uploads at {}.",
                claude_desktop::UPLOAD_URL
            ),
            Some(dir) if !dir.is_dir() => eprintln!(
                "  Claude Desktop is not installed here ({} is missing), so no connector was added. Install it and run this again.",
                display_path(&dir)
            ),
            Some(dir) => add_desktop_connector(args, host, &dir)?,
        }
    }

    eprintln!();
    eprintln!("Next, in Claude Desktop:");
    if !args.no_connector {
        eprintln!("  1. Quit and reopen Claude Desktop so it starts the mold connector.");
    } else {
        eprintln!("  1. (No connector changes; skip the restart.)");
    }
    eprintln!(
        "  2. Customize > Skills > Add > Upload a skill, choose {}, and turn it on.",
        display_path(&zip)
    );
    eprintln!("     Remove or replace any older \"mold\" skill there first.");
    eprintln!(
        "  3. Settings > Capabilities: \"Code execution and file creation\" must be on for skills."
    );
    eprintln!("  Skills page: {}", claude_desktop::UPLOAD_URL);
    if open && std::io::IsTerminal::is_terminal(&std::io::stderr()) {
        claude_desktop::open_upload_page(&zip);
    }
    Ok(())
}

fn add_desktop_connector(args: &SkillInstallArgs, host: Option<&str>, dir: &Path) -> Result<()> {
    let host_label = match host {
        Some(host) => connector_host(host)?,
        None => match mold_secure_client::hosts::default() {
            Ok(Some(saved)) => format!("your default host, {}", saved.name),
            _ => {
                let saved = mold_secure_client::hosts::list()
                    .unwrap_or_default()
                    .into_iter()
                    .map(|host| host.name)
                    .collect::<Vec<_>>();
                if saved.is_empty() {
                    "http://localhost:7680, since no default host is set (`mold host use NAME`)"
                        .to_string()
                } else {
                    format!(
                        "http://localhost:7680, since no default host is set; pin a saved host with `mold skill install claude-desktop --host NAME` ({})",
                        saved.join(", ")
                    )
                }
            }
        },
    };
    let exe = std::env::current_exe().context("could not locate the mold binary")?;
    let exe = exe.canonicalize().unwrap_or(exe);
    let cwd = std::env::current_dir().context("could not determine current directory")?;
    let mold_home = std::env::var("MOLD_HOME")
        .ok()
        .filter(|home| !home.trim().is_empty())
        .map(|home| absolute_mold_home(&home, &cwd))
        .transpose()?;
    let connector = claude_desktop::Connector::new(exe.display().to_string(), host, mold_home);
    let path = claude_desktop::config_path(dir);
    let mut config = claude_desktop::read_config(&path)?;
    let owned = claude_desktop::read_owned(&path);
    let merge =
        claude_desktop::merge_connector(&mut config, &connector, owned.as_ref(), args.force)?;
    match merge {
        claude_desktop::Merge::Unchanged => eprintln!(
            "✓ The mold connector in {} is already current.",
            display_path(&path)
        ),
        claude_desktop::Merge::Added | claude_desktop::Merge::Updated => {
            let backup = claude_desktop::write_config(&path, &config)?;
            eprintln!(
                "✓ {} the mold connector in {}.",
                if merge == claude_desktop::Merge::Added {
                    "Added"
                } else {
                    "Updated"
                },
                display_path(&path)
            );
            if let Some(backup) = backup {
                eprintln!("  Previous config saved as {}", display_path(&backup));
            }
        }
    }
    // Record what is now under mcpServers.mold, so later runs (and
    // uninstall) can tell mold's entry from one a person wrote or edited.
    claude_desktop::write_owned(&path, claude_desktop::connector_entry(&config))?;
    eprintln!("  It talks to {host_label}.");
    if exe.components().any(|part| part.as_os_str() == "target") {
        eprintln!(
            "  Note: the connector runs {}, a build output; install mold and run this again for a stable path.",
            display_path(&exe)
        );
    }
    if std::env::var_os("MOLD_HOST").is_some() && host.is_none() {
        eprintln!(
            "  Note: Claude Desktop does not see MOLD_HOST; pass --host to pin the connector."
        );
    }
    Ok(())
}

/// MOLD_HOME as Desktop must see it: Desktop starts the connector from its
/// own working directory, so `~` and relative paths are resolved here.
fn absolute_mold_home(raw: &str, cwd: &Path) -> Result<String> {
    let path = expand_tilde(Path::new(raw))?;
    let path = if path.is_absolute() {
        path
    } else {
        cwd.join(path)
    };
    Ok(path.display().to_string())
}

/// Check a `--host` value before it is written into Desktop's config, where a
/// typo would only show up as a broken connector.
fn connector_host(host: &str) -> Result<String> {
    if let Some(saved) = mold_secure_client::hosts::resolve(host)? {
        return Ok(format!("host {}", saved.name));
    }
    let looks_like_address = host.contains("://") || host.contains(':') || host.contains('.');
    // A bare name that is not saved can still be a DNS or MagicDNS name, as
    // `mold mcp --host` accepts; only one that resolves nowhere is a typo.
    let resolves = || {
        use std::net::ToSocketAddrs;
        (host, 7680)
            .to_socket_addrs()
            .is_ok_and(|mut addrs| addrs.next().is_some())
    };
    if !looks_like_address && !resolves() {
        let saved = mold_secure_client::hosts::list()
            .unwrap_or_default()
            .into_iter()
            .map(|host| host.name)
            .collect::<Vec<_>>();
        anyhow::bail!(
            "'{host}' is not a saved host and does not resolve{}; pair it with `mold host add` or pass a URL",
            if saved.is_empty() {
                String::new()
            } else {
                format!(" (saved: {})", saved.join(", "))
            }
        );
    }
    Ok(mold_secure_client::hosts::normalize_address(host))
}

fn uninstall_claude_desktop(named: bool) -> Result<usize> {
    let mut removed = 0;
    let home = dirs::home_dir().context("could not determine the home directory")?;
    if let Some(dir) = desktop_config_dir(&home).filter(|dir| dir.is_dir()) {
        let path = claude_desktop::config_path(&dir);
        let mut config = claude_desktop::read_config(&path)?;
        let owned = claude_desktop::read_owned(&path);
        match claude_desktop::remove_connector(&mut config, owned.as_ref())? {
            claude_desktop::Removal::Removed => {
                let backup = claude_desktop::write_config(&path, &config)?;
                claude_desktop::write_owned(&path, None)?;
                eprintln!("Removed the mold connector from {}", display_path(&path));
                if let Some(backup) = backup {
                    eprintln!("  Previous config saved as {}", display_path(&backup));
                }
                removed += 1;
            }
            claude_desktop::Removal::NotOurs => eprintln!(
                "Left mcpServers.mold in {} alone: mold did not write it.",
                display_path(&path)
            ),
            claude_desktop::Removal::Absent => {}
        }
    }
    if let Ok(cwd) = std::env::current_dir() {
        let zip = claude_desktop::default_zip_path(dirs::download_dir(), &cwd);
        if claude_desktop::is_mold_skill_zip(&zip) {
            std::fs::remove_file(&zip)
                .with_context(|| format!("failed to remove {}", zip.display()))?;
            eprintln!("Removed {}", display_path(&zip));
            removed += 1;
        }
    }
    if named || removed > 0 {
        eprintln!(
            "The uploaded skill stays on your Claude account; remove it at {}.",
            claude_desktop::UPLOAD_URL
        );
    }
    Ok(removed)
}

/// The `mold skill list` cell for Claude Desktop.
fn desktop_status(home: &Path) -> String {
    let connector = match desktop_config_dir(home).filter(|dir| dir.is_dir()) {
        None => "-".to_string(),
        Some(dir) => match claude_desktop::read_config(&claude_desktop::config_path(&dir)) {
            Ok(config) => claude_desktop::connector_state(
                &config,
                claude_desktop::read_owned(&claude_desktop::config_path(&dir)).as_ref(),
            )
            .to_string(),
            Err(_) => "config unreadable".to_string(),
        },
    };
    let zip = std::env::current_dir()
        .map(|cwd| claude_desktop::default_zip_path(dirs::download_dir(), &cwd))
        .ok()
        .filter(|zip| zip.is_file());
    let zip = match zip {
        None => "no ZIP".to_string(),
        Some(zip) => {
            let current = render_bundle(RenderProfile::ClaudeDesktop)
                .ok()
                .and_then(|bundle| bundle.files_with_manifest().ok())
                .and_then(|files| files.get(MANIFEST_FILE).cloned());
            let state = if !claude_desktop::is_mold_skill_zip(&zip) {
                "not a mold skill"
            } else if claude_desktop::zipped_manifest(&zip) == current {
                "current"
            } else {
                "outdated"
            };
            format!("{} [{state}]", display_path(&zip))
        }
    };
    format!("{connector}; {zip}")
}

fn install_directories(args: &SkillInstallArgs, agents: &[Agent]) -> Result<()> {
    let scope = resolve_scope(args.project, &args.dir)?;
    for (dir, target) in dedupe_targets(&scope, agents) {
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
    let project = args.project || args.dir.is_some();
    let (named_desktop, named) = split_desktop(args.agents.clone());
    if named_desktop && project {
        anyhow::bail!(
            "claude-desktop installs to your Claude account, not a project; drop --project/--dir"
        );
    }
    let mut removed = 0;
    // With no agents named, user scope covers Claude Desktop too.
    if named_desktop || (args.agents.is_empty() && !project) {
        removed += uninstall_claude_desktop(named_desktop)?;
        if named.is_empty() && !args.agents.is_empty() {
            return Ok(());
        }
    }
    let scope = resolve_scope(args.project, &args.dir)?;
    let agents = if args.agents.is_empty() {
        AGENTS.iter().map(|spec| spec.agent).collect::<Vec<_>>()
    } else {
        named
    };
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
    let bundle = render_bundle(args.agent.show_profile())?;
    let files = bundle.files_with_manifest()?;
    let key = args.file.to_string_lossy();
    let contents = files.get(key.as_ref()).with_context(|| {
        format!(
            "no rendered {} bundle file {}; available: {}",
            args.agent.cli_name(),
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
    if let Some(home) = &home {
        let detected = if desktop_detected(home) { "yes" } else { "no" };
        table.add_row([
            "Claude Desktop",
            "claude-desktop",
            detected,
            &desktop_status(home),
            "-",
        ]);
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
            .chain([RenderProfile::ClaudeDesktop])
            .collect::<BTreeSet<_>>();
        assert_eq!(profiles.len(), RenderProfile::ALL.len());
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
    fn every_profile_routes_to_executable_video_sequence_instructions() {
        for profile in RenderProfile::ALL {
            let bundle = render_bundle(profile).unwrap();
            assert!(bundle.files["SKILL.md"].contains("For multi-scene video"));
            let reference = &bundle.files["references/cli.md"];
            let section = reference
                .split("## Scripted sequences")
                .nth(1)
                .unwrap()
                .split("\n## ")
                .next()
                .unwrap();
            let toml = section
                .split("```toml\n")
                .nth(1)
                .expect("sequence instructions need a complete script")
                .split("```")
                .next()
                .unwrap();
            let script = mold_core::chain_toml::read_script(toml).unwrap();
            let request = crate::commands::chain::build_request_from_script(&script)
                .unwrap()
                .normalise_with_family(Some("ltx2"))
                .unwrap();
            assert_eq!(request.stages.len(), 2);
            assert_eq!(
                request.stages[1].transition,
                mold_core::chain::TransitionMode::Cut
            );
            assert!(section.contains("mold jobs show job-abc123 --json --host workstation"));
            assert!(section.contains("mold library download PRINT.mp4 --host workstation"));
            assert!(section.contains("MiniMax H3"));
            assert!(section.contains("CLI machine"));
            assert!(section.contains("finalizes[-1].gallery_filename"));
            assert!(section.contains("--stage 1 --mode splice"));
            assert!(
                !section.contains("--fps 30 --no-audio"),
                "generic amend must preserve H3 timing and audio"
            );
        }
    }

    #[test]
    fn documented_chain_api_request_deserializes_and_normalizes() {
        let guide = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../website/guide/video.md"
        ));
        let json = guide
            .split("### HTTP API")
            .nth(1)
            .unwrap()
            .split("```json\n")
            .nth(1)
            .unwrap()
            .split("```")
            .next()
            .unwrap();
        let request: mold_core::chain::ChainRequest = serde_json::from_str(json).unwrap();
        let request = request.normalise_with_family(Some("ltx2")).unwrap();
        assert_eq!(request.stages.len(), 2);
        assert_eq!(
            request.stages[1].transition,
            mold_core::chain::TransitionMode::Cut
        );
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
    fn every_skill_profile_explains_h3_reference_transport_and_license_separately() {
        for profile in RenderProfile::ALL {
            let bundle = render_bundle(profile).unwrap();
            let reference = &bundle
                .files
                .iter()
                .find(|(path, _)| path.ends_with("prompting/minimax-h3/ref2va.md"))
                .unwrap()
                .1;
            assert!(reference.contains("authentication disabled"));
            assert!(reference.contains("32 MiB"));
            assert!(reference.contains("combined across all batch siblings"));
            assert!(reference.contains("License acceptance"));
            assert!(!reference.contains("MOLD_API_KEY` must be"));
            assert!(reference.contains("<Subject n>"));
        }
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
        for profile in RenderProfile::ALL {
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
            if path.starts_with("references/prompting/models/") {
                // Model leaves are named after checkpoint base names.
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
    fn every_direct_relative_skill_link_resolves_in_each_bundle() {
        for profile in RenderProfile::ALL {
            let bundle = render_bundle(profile).unwrap();
            let skill = &bundle.files["SKILL.md"];
            for tail in skill.split("](").skip(1) {
                let target = tail.split(')').next().unwrap();
                if target.starts_with("http") {
                    continue;
                }
                let relative = target.strip_prefix("{baseDir}/").unwrap_or(target);
                // Markdown fragments name sections inside the bundled file.
                // The sequence-guide contract separately checks its heading.
                let relative = relative.split('#').next().unwrap();
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
            parse_on_large_stack(argv.iter().map(|arg| (*arg).to_string()).collect())
                .unwrap_or_else(|error| panic!("invalid example {rendered}: {error}"));
        }
    }

    /// The full clap tree no longer fits the default test-thread stack, so
    /// parse on a thread sized like the real `main`.
    fn parse_on_large_stack(argv: Vec<String>) -> std::result::Result<(), String> {
        std::thread::Builder::new()
            .stack_size(64 << 20)
            .spawn(move || {
                match crate::Cli::try_parse_from(std::iter::once("mold".to_string()).chain(argv)) {
                    Ok(_) => Ok(()),
                    // `mold --help` and `mold <cmd> --help` are documented on
                    // purpose; clap reports them as errors that print help.
                    Err(error)
                        if matches!(
                            error.kind(),
                            clap::error::ErrorKind::DisplayHelp
                                | clap::error::ErrorKind::DisplayVersion
                        ) =>
                    {
                        Ok(())
                    }
                    Err(error) => Err(error.to_string()),
                }
            })
            .expect("spawn parse thread")
            .join()
            .expect("parse thread panicked")
    }

    /// Split one documented shell command into argv the way a POSIX shell
    /// would for the subset the corpus uses: single and double quotes,
    /// backslash escapes, `$(...)` kept as one word inside quotes, an
    /// optional `KEY=VALUE` environment prefix, and a trailing `| pager`.
    fn shell_words(command: &str) -> Vec<String> {
        let mut words = Vec::new();
        let mut current = String::new();
        let mut in_word = false;
        let mut quote: Option<char> = None;
        let mut chars = command.chars().peekable();
        while let Some(c) = chars.next() {
            match quote {
                Some('\'') => {
                    if c == '\'' {
                        quote = None;
                    } else {
                        current.push(c);
                    }
                }
                Some('"') => match c {
                    '"' => quote = None,
                    '\\' => {
                        if let Some(next) = chars.next() {
                            current.push(next);
                        }
                    }
                    _ => current.push(c),
                },
                _ => match c {
                    '\'' | '"' => {
                        quote = Some(c);
                        in_word = true;
                    }
                    '\\' => {
                        if let Some(next) = chars.next() {
                            if next != '\n' {
                                current.push(next);
                                in_word = true;
                            }
                        }
                    }
                    '|' | '#' => break,
                    // A redirection ends the command the CLI is asked to
                    // parse: `mold completions fish > ~/.config/…` documents
                    // a shell redirection, not an argument. The space is
                    // what distinguishes it from a `<PLACEHOLDER>` in a
                    // documented synopsis, which stays a word.
                    '>' | '<' if chars.peek().is_none_or(|next| next.is_whitespace()) => break,
                    c if c.is_whitespace() => {
                        if in_word {
                            words.push(std::mem::take(&mut current));
                            in_word = false;
                        }
                    }
                    _ => {
                        current.push(c);
                        in_word = true;
                    }
                },
            }
        }
        if in_word {
            words.push(current);
        }
        words
    }

    /// Every `mold ...` line in every fenced bash block of a markdown file.
    fn documented_commands(markdown: &str) -> Vec<String> {
        let mut commands = Vec::new();
        let mut in_bash = false;
        let mut pending = String::new();
        for line in markdown.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("```") {
                if in_bash {
                    in_bash = false;
                } else {
                    let info = trimmed.trim_start_matches('`').trim().to_ascii_lowercase();
                    in_bash = matches!(info.as_str(), "bash" | "sh" | "shell" | "console");
                }
                continue;
            }
            if !in_bash || trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            if let Some(head) = trimmed.strip_suffix('\\') {
                pending.push_str(head);
                pending.push(' ');
                continue;
            }
            pending.push_str(trimmed);
            let command = std::mem::take(&mut pending);
            // `cat x | mold ...` and `KEY=VALUE mold ...` prefixes.
            let start = command.find("mold ").filter(|index| {
                *index == 0 || command[..*index].ends_with(' ') || command[..*index].ends_with('|')
            });
            if let Some(start) = start {
                commands.push(command[start..].to_string());
            }
        }
        commands
    }

    #[test]
    fn every_documented_bash_command_in_the_corpus_parses_with_the_cli() {
        let bundle = render_bundle(RenderProfile::Portable).unwrap();
        let mut seen = 0usize;
        for (path, contents) in &bundle.files {
            if !path.ends_with(".md") {
                continue;
            }
            for command in documented_commands(contents) {
                let argv = shell_words(&command);
                assert_eq!(
                    argv.first().map(String::as_str),
                    Some("mold"),
                    "{path}: {command}"
                );
                seen += 1;
                parse_on_large_stack(argv[1..].to_vec())
                    .unwrap_or_else(|error| panic!("{path}: invalid example `{command}`: {error}"));
            }
        }
        assert!(
            seen >= 20,
            "expected the corpus to document CLI examples, found {seen}"
        );
    }

    /// Commands the skill and corpus deliberately do not carry an example
    /// for, each with the reason it is not agent-facing work.
    ///
    /// This list is the whole escape hatch: anything not named here must
    /// appear in at least one `bash` fence, so a NEW command is a failing
    /// test rather than a silent omission — which is exactly how
    /// `mold run --script`, `mold chain validate` and eight of ten
    /// `library` verbs went untaught while every fence still parsed.
    const UNDOCUMENTED_BY_DESIGN: &[(&str, &str)] = &[
        // Interactive or long-running surfaces an agent cannot drive from a
        // shell. `discord` is feature-gated, so it enters the walk only in a
        // build that carries it — CI's feature-union build does, a default
        // `cargo test` does not.
        ("discord", "runs the Discord bot (feature `discord`)"),
        (
            "server discover",
            "LAN discovery (feature `mdns`); the machines guide documents it",
        ),
        // Machine setup a person does once, not work an agent performs.
        ("completions", "shell setup performed by the user"),
        ("update", "self-update of the installed binary"),
        (
            "skill install",
            "installs THIS skill; a person's setup step",
        ),
        (
            "skill uninstall",
            "removes THIS skill; a person's setup step",
        ),
        ("config edit", "opens $EDITOR"),
        // Destructive or lifecycle actions the safety reference governs and
        // that deliberately carry no copy-paste example, exactly as
        // `mold queue cancel --all --yes` does.
        ("server start", "starts a daemon; safety.md governs it"),
        ("server stop", "stops a daemon; safety.md governs it"),
        ("trash empty", "purges every trashed print"),
        ("config reset", "discards user settings"),
        ("gpu disable", "changes host device lifecycle"),
        ("gpu enable", "changes host device lifecycle"),
        ("system metal-memory set", "root-only kernel mutation"),
        ("system metal-memory reset", "root-only kernel mutation"),
        ("rm", "permanently deletes installed weights"),
        ("clean", "deletes cached files once forced"),
        // Paid cloud provisioning, deliberately outside the corpus.
        ("runpod", "paid cloud provisioning"),
        ("lambda", "paid cloud provisioning"),
    ];

    /// Every command path in the clap tree that a user can invoke.
    ///
    /// A command whose subcommand is optional (`mold licenses`) is a
    /// destination in its own right as well as a parent.
    fn invocable_command_paths() -> Vec<String> {
        fn walk(command: &clap::Command, prefix: Vec<String>, out: &mut Vec<String>) {
            let subcommands: Vec<&clap::Command> = command
                .get_subcommands()
                .filter(|sub| !sub.is_hide_set() && sub.get_name() != "help")
                .collect();
            if !prefix.is_empty()
                && (subcommands.is_empty() || !command.is_subcommand_required_set())
            {
                out.push(prefix.join(" "));
            }
            for sub in subcommands {
                let mut next = prefix.clone();
                next.push(sub.get_name().to_string());
                walk(sub, next, out);
            }
        }
        let command = clap_tree();
        let mut out = Vec::new();
        walk(&command, Vec::new(), &mut out);
        out
    }

    /// The command path one documented invocation names, by walking the clap
    /// tree with its argv: `mold library export chair.glb --format stl`
    /// answers `library export`.
    fn documented_command_path(command: &clap::Command, argv: &[String]) -> Option<String> {
        let mut current = command;
        let mut path = Vec::new();
        for token in argv.iter().skip(1) {
            let Some(next) = current
                .get_subcommands()
                .find(|sub| sub.get_name() == token || sub.get_all_aliases().any(|a| a == token))
            else {
                break;
            };
            path.push(next.get_name().to_string());
            current = next;
        }
        (!path.is_empty()).then(|| path.join(" "))
    }

    /// Every command a user can run appears in at least one `bash` fence of
    /// the rendered skill or the prompting corpus.
    ///
    /// The existing fence test can only check the examples that EXIST, so an
    /// untaught command is invisible to it. This is the other half: an
    /// omission fails the build.
    #[test]
    fn every_user_facing_command_appears_in_a_documented_example() {
        let command = clap_tree();
        let bundle = render_bundle(RenderProfile::Portable).unwrap();
        let mut documented = std::collections::BTreeSet::new();
        for (path, contents) in &bundle.files {
            if !path.ends_with(".md") {
                continue;
            }
            for line in documented_commands(contents) {
                if let Some(found) = documented_command_path(&command, &shell_words(&line)) {
                    documented.insert(found);
                }
            }
        }
        // `mold run` and the other flat commands are named by their own line.
        let exempt: std::collections::BTreeSet<&str> = UNDOCUMENTED_BY_DESIGN
            .iter()
            .map(|(name, _)| *name)
            .collect();
        let missing: Vec<String> = invocable_command_paths()
            .into_iter()
            .filter(|path| {
                !documented.contains(path)
                    && !exempt
                        .iter()
                        .any(|name| path == name || path.starts_with(&format!("{name} ")))
            })
            .collect();
        assert!(
            missing.is_empty(),
            "these commands appear in no skill or corpus example, and are not in \
             UNDOCUMENTED_BY_DESIGN:\n  {}",
            missing.join("\n  ")
        );
    }

    /// Exemptions for commands that exist only behind a cargo feature. They
    /// are allowed to be absent from a default-feature walk; a build that
    /// carries the feature still checks them like any other.
    const FEATURE_GATED_EXEMPTIONS: &[&str] = &["discord", "server discover"];

    /// The interactive terminal app is retired, so no command path may name
    /// it or the Library grid that was its one other door. A clap variant
    /// left behind would still be reachable from the shell and would still
    /// demand a skill example.
    #[test]
    fn no_command_path_names_the_retired_terminal_app() {
        let paths = invocable_command_paths();
        for retired in ["tui", "library grid"] {
            assert!(
                !paths
                    .iter()
                    .any(|path| path == retired || path.starts_with(&format!("{retired} "))),
                "`mold {retired}` is retired but is still an invocable command"
            );
        }
    }

    /// Every exemption names a command that exists, so a renamed or retired
    /// command cannot leave a stale excuse behind.
    #[test]
    fn every_coverage_exemption_names_a_real_command() {
        let paths = invocable_command_paths();
        for (name, reason) in UNDOCUMENTED_BY_DESIGN {
            if FEATURE_GATED_EXEMPTIONS.contains(name) && !paths.iter().any(|path| path == name) {
                continue;
            }
            assert!(
                paths
                    .iter()
                    .any(|path| path == name || path.starts_with(&format!("{name} "))),
                "`{name}` is exempt ({reason}) but is not a command"
            );
        }
    }

    /// Build the clap tree on a thread sized like the real `main`.
    fn clap_tree() -> clap::Command {
        std::thread::Builder::new()
            .stack_size(64 << 20)
            .spawn(<crate::Cli as clap::CommandFactory>::command)
            .expect("spawn clap thread")
            .join()
            .expect("clap thread panicked")
    }

    /// Examples in the user docs that only parse in a build carrying the
    /// named optional feature. Each is matched as a prefix.
    ///
    /// The test binary is built with default features, so these would read
    /// as broken documentation when they are simply documentation for a
    /// different build. Every entry is checked to be USED, so an example
    /// that stops appearing cannot leave a stale excuse behind.
    const FEATURE_GATED_DOC_EXAMPLES: &[(&str, &str)] = &[
        ("mold discord", "discord"),
        ("mold serve --discord", "discord"),
        ("mold server discover", "mdns"),
    ];

    /// A documented line that shows a command's SHAPE rather than a command
    /// to run: `mold library show <FILENAME> [--json | --preview]`.
    ///
    /// Those cannot be parsed — the placeholders are the point — so the
    /// literal examples beside them are what this checks. A synopsis is
    /// recognised by its placeholder brackets, never by which file it is in.
    fn is_synopsis(command: &str) -> bool {
        command.contains('<') || command.contains('[') || command.contains("OPTIONS")
    }

    /// Every runnable `mold ...` line in a `bash` fence of the user-facing
    /// docs parses with the real CLI.
    ///
    /// The corpus has had this check since the skill existed; README and the
    /// VitePress site had NOTHING, so a flag renamed in clap stayed right in
    /// the agent's copy and wrong in the human's — which is how
    /// `--negative` (the flag is `--negative-prompt`) survived in a model
    /// page. `website/guide/prompting.md` is excluded because it is
    /// GENERATED from the corpus, which this module already checks.
    #[test]
    fn every_documented_bash_command_in_the_user_docs_parses_with_the_cli() {
        fn markdown_files(root: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
            let Ok(entries) = std::fs::read_dir(root) else {
                return;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    if path.file_name().is_some_and(|name| {
                        name == "node_modules" || name == ".vitepress" || name == "dist"
                    }) {
                        continue;
                    }
                    markdown_files(&path, out);
                } else if path.extension().is_some_and(|ext| ext == "md") {
                    out.push(path);
                }
            }
        }

        let repo = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|path| path.parent())
            .expect("repo root")
            .to_path_buf();
        let generated = repo.join("website/guide/prompting.md");
        let mut files = vec![repo.join("README.md")];
        markdown_files(&repo.join("website"), &mut files);
        files.retain(|path| path != &generated);
        files.sort();

        let mut seen = 0usize;
        let mut used_gates = std::collections::BTreeSet::new();
        for file in &files {
            let contents = std::fs::read_to_string(file)
                .unwrap_or_else(|error| panic!("{}: {error}", file.display()));
            for command in documented_commands(&contents) {
                if is_synopsis(&command) {
                    continue;
                }
                if let Some((prefix, feature)) = FEATURE_GATED_DOC_EXAMPLES
                    .iter()
                    .find(|(prefix, _)| command.starts_with(prefix))
                {
                    used_gates.insert(*prefix);
                    let _ = feature;
                    continue;
                }
                let argv = shell_words(&command);
                assert_eq!(
                    argv.first().map(String::as_str),
                    Some("mold"),
                    "{}: {command}",
                    file.display()
                );
                seen += 1;
                parse_on_large_stack(argv[1..].to_vec()).unwrap_or_else(|error| {
                    panic!("{}: invalid example `{command}`: {error}", file.display())
                });
            }
        }
        assert!(
            seen >= 100,
            "expected the user docs to carry CLI examples, found {seen}"
        );
        for (prefix, feature) in FEATURE_GATED_DOC_EXAMPLES {
            assert!(
                used_gates.contains(prefix),
                "`{prefix}` is excused as needing the `{feature}` feature but appears in no example"
            );
        }
    }

    #[test]
    fn shell_words_handles_the_corpus_quoting_forms() {
        assert_eq!(
            shell_words(r#"mold run flux-dev:q4 "a cat, \"quoted\"" --seed 1 | viu -"#),
            vec![
                "mold",
                "run",
                "flux-dev:q4",
                "a cat, \"quoted\"",
                "--seed",
                "1"
            ]
        );
        assert_eq!(
            shell_words("mold run h3 \"$(cat h3-prompt.txt)\" --duration 5"),
            vec![
                "mold",
                "run",
                "h3",
                "$(cat h3-prompt.txt)",
                "--duration",
                "5"
            ]
        );
        assert_eq!(
            shell_words("mold run 'it''s' -o out.png # comment"),
            vec!["mold", "run", "its", "-o", "out.png"]
        );
        assert_eq!(
            shell_words("mold completions fish > ~/.config/fish/completions/mold.fish"),
            vec!["mold", "completions", "fish"]
        );
        assert_eq!(
            shell_words("mold upscale - < input.png > output.png"),
            vec!["mold", "upscale", "-"]
        );
        assert_eq!(
            documented_commands(
                "```bash\nMOLD_HOST=http://h mold server status\ncat p.png | mold run \"x\" \\\n  --image -\n```\n```text\nmold not-a-command\n```"
            ),
            vec![
                "mold server status".to_string(),
                "mold run \"x\"  --image -".to_string()
            ]
        );
    }
}

#[cfg(test)]
mod claude_desktop_tests {
    use super::*;

    fn parse(args: &[&str]) -> SkillArgs {
        SkillArgs::try_parse_from(std::iter::once("skill").chain(args.iter().copied())).unwrap()
    }

    fn install_error(args: &[&str]) -> String {
        let SkillCommand::Install(install_args) = parse(args).command else {
            panic!("not an install");
        };
        install(&install_args, None).unwrap_err().to_string()
    }

    #[test]
    fn only_the_desktop_skill_opens_with_the_connector_preamble() {
        let desktop = render_bundle(RenderProfile::ClaudeDesktop).unwrap();
        let skill = &desktop.files["SKILL.md"];
        let body = skill.split("\n---\n\n").nth(1).unwrap();
        assert!(
            body.starts_with("# Mold media generation\n\n## In Claude Desktop and claude.ai\n"),
            "{}",
            &body[..200]
        );
        for profile in RenderProfile::ALL {
            if profile != RenderProfile::ClaudeDesktop {
                let other = render_bundle(profile).unwrap();
                assert!(!other.files["SKILL.md"].contains("## In Claude Desktop"));
                assert!(other.files["SKILL.md"]
                    .contains("# Mold media generation\n\nUse `mold` to generate"));
            }
        }
    }

    #[test]
    fn the_desktop_skill_frontmatter_is_name_and_description_only() {
        let desktop = render_bundle(RenderProfile::ClaudeDesktop).unwrap();
        let frontmatter = desktop.files["SKILL.md"].split("\n---\n").next().unwrap();
        assert_eq!(
            frontmatter,
            format!("---\nname: mold\ndescription: {DESCRIPTION}")
        );
        // claude.ai's upload limit (and the Agent Skills specification's).
        assert!(DESCRIPTION.chars().count() <= 1024);
    }

    #[test]
    fn every_tool_the_desktop_preamble_names_is_a_mold_mcp_tool() {
        let tools = crate::commands::mcp::tool_definitions();
        let names = tools
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|tool| tool["name"].as_str())
            .collect::<BTreeSet<_>>();
        for tool in CLAUDE_DESKTOP_TOOLS {
            assert!(names.contains(tool), "mold mcp has no {tool} tool");
            assert!(
                CLAUDE_DESKTOP_PREAMBLE.contains(&format!("`{tool}`")),
                "{tool} is listed but not named in the preamble"
            );
        }
        for named in CLAUDE_DESKTOP_PREAMBLE
            .split('`')
            .skip(1)
            .step_by(2)
            .filter(|word| word.chars().all(|c| c.is_ascii_lowercase() || c == '_'))
            .filter(|word| word.contains('_'))
        {
            assert!(
                CLAUDE_DESKTOP_TOOLS.contains(&named),
                "the preamble names `{named}`; add it to CLAUDE_DESKTOP_TOOLS"
            );
        }
    }

    #[test]
    fn the_desktop_zip_is_the_rendered_bundle_under_mold() {
        let bundle = render_bundle(RenderProfile::ClaudeDesktop).unwrap();
        let files = bundle.files_with_manifest().unwrap();
        let bytes = claude_desktop::zip_bytes(&files).unwrap();
        let archive = zip::ZipArchive::new(std::io::Cursor::new(bytes)).unwrap();
        let names = archive.file_names().collect::<BTreeSet<_>>();
        assert!(names.contains("mold/SKILL.md"));
        assert!(names.contains("mold/.mold-skill.json"));
        assert!(names.contains("mold/references/cli.md"));
        assert!(names.iter().all(|name| name.starts_with("mold/")));
        assert_eq!(names.len(), files.len() + 1, "files plus the mold/ folder");
    }

    #[test]
    fn claude_desktop_parses_with_its_flags() {
        let SkillCommand::Install(args) = parse(&[
            "install",
            "claude-desktop",
            "-o",
            "out.zip",
            "--no-connector",
            "--no-open",
            "--force",
        ])
        .command
        else {
            panic!("not an install");
        };
        assert_eq!(args.agents, [Agent::ClaudeDesktop]);
        assert_eq!(args.output.as_deref(), Some(Path::new("out.zip")));
        assert!(args.no_connector && args.no_open && args.force);
        assert!(matches!(
            parse(&["show", "claude-desktop"]).command,
            SkillCommand::Show(SkillShowArgs {
                agent: Agent::ClaudeDesktop,
                ..
            })
        ));
        assert!(matches!(
            parse(&["uninstall", "claude-desktop"]).command,
            SkillCommand::Uninstall(_)
        ));
    }

    #[test]
    fn claude_desktop_refuses_project_scope_and_desktop_flags_need_it() {
        let error = install_error(&["install", "claude-desktop", "--project"]);
        assert!(error.contains("not a project"), "{error}");
        let error = install_error(&["install", "claude", "--force", "--no-open"]);
        assert!(
            error.contains("--no-open, --force only apply to claude-desktop"),
            "{error}"
        );
        assert!(parse(&["install", "claude-desktop"]).takes_host());
        assert!(!parse(&["install", "claude"]).takes_host());
        assert!(!parse(&["show", "claude-desktop"]).takes_host());
    }

    #[test]
    fn bulk_project_installs_never_pick_up_claude_desktop() {
        let home = tempfile::tempdir().unwrap();
        if let Some(dir) = desktop_config_dir(home.path()) {
            std::fs::create_dir_all(dir).unwrap();
        }
        for args in [
            &["install", "--all", "--project"][..],
            &["install", "--detected", "--dir", "/project"],
        ] {
            let SkillCommand::Install(install_args) = parse(args).command else {
                panic!("not an install");
            };
            assert!(
                !auto_includes_desktop(&install_args, home.path()),
                "{args:?}"
            );
        }
        let SkillCommand::Install(user) = parse(&["install", "--all"]).command else {
            panic!("not an install");
        };
        assert_eq!(
            auto_includes_desktop(&user, home.path()),
            desktop_config_dir(home.path()).is_some()
        );
    }

    #[test]
    #[cfg(unix)]
    fn the_connector_gets_an_absolute_mold_home() {
        let cwd = Path::new("/work");
        assert_eq!(
            absolute_mold_home("./mold-data", cwd).unwrap(),
            "/work/./mold-data"
        );
        assert_eq!(absolute_mold_home("/srv/mold", cwd).unwrap(), "/srv/mold");
        let home = dirs::home_dir().unwrap();
        assert_eq!(
            absolute_mold_home("~/mold", cwd).unwrap(),
            home.join("mold").display().to_string()
        );
    }

    #[test]
    fn claude_desktop_has_its_own_show_profile() {
        assert_eq!(
            Agent::ClaudeDesktop.show_profile(),
            RenderProfile::ClaudeDesktop
        );
        assert_eq!(Agent::ClaudeDesktop.cli_name(), "claude-desktop");
        assert_eq!(Agent::Codex.show_profile(), RenderProfile::Codex);
    }
}
