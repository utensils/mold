//! Agent Skill generation and installation.
//!
//! The canonical Mold skill bundle is embedded in the CLI and installed into
//! supported agents' user-wide or project-local skill paths.

use anyhow::{Context, Result};
use clap::{ArgGroup, Parser, Subcommand, ValueEnum};
use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Component, Path, PathBuf};

/// The canonical skill content installed by `mold skill install`.
pub const SKILL_MD: &str = include_str!("SKILL.md");
/// Model-family prompting recipes installed with the canonical skill.
pub const MODEL_PROMPTING_MD: &str = include_str!("references/model-prompting.md");
const SKILL_DIR_NAME: &str = "mold";

#[derive(Parser, Debug)]
pub struct SkillArgs {
    #[command(subcommand)]
    pub command: SkillCommand,
}

#[derive(Subcommand, Debug)]
pub enum SkillCommand {
    /// List supported agents, their skill paths, and install status
    List(SkillListArgs),

    /// Install the mold skill for explicitly selected agents
    ///
    /// Name one or more agents, or explicitly pass `--detected` or `--all`.
    /// User-wide is the default scope; use `--project` or `--dir` for a
    /// project-level install.
    Install(SkillInstallArgs),

    /// Remove installed mold skills
    ///
    /// With no agent arguments, removes the skill from every known agent path
    /// in the selected scope. Only Mold-managed skill files are deleted;
    /// non-empty directories and unrelated files are preserved.
    Uninstall(SkillUninstallArgs),

    /// Print the generated SKILL.md to stdout
    Show,
}

#[derive(Parser, Debug)]
pub struct SkillListArgs {
    /// Use ASCII table borders instead of Unicode
    #[arg(long)]
    ascii: bool,
}

#[derive(Parser, Debug)]
#[command(group(
    ArgGroup::new("targets")
        .required(true)
        .multiple(false)
        .args(["agents", "detected", "all"])
))]
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

/// AI coding agents supported by Mold. This intentionally matches nxv's
/// installer matrix so users get the same targets from both CLIs.
#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Agent {
    /// Claude Code (~/.claude/skills, .claude/skills)
    Claude,
    /// OpenAI Codex CLI (~/.codex/skills, .agents/skills)
    Codex,
    /// Pi (~/.pi/agent/skills, .pi/skills)
    Pi,
    /// OpenClaw (~/.openclaw/skills, .agents/skills)
    Openclaw,
    /// GitHub Copilot CLI (~/.copilot/skills, .github/skills)
    Copilot,
    /// Cursor (~/.cursor/skills, .agents/skills)
    Cursor,
    /// Gemini CLI (~/.gemini/skills, .agents/skills)
    Gemini,
    /// Amp (~/.config/amp/skills, .agents/skills)
    Amp,
    /// Goose (~/.config/goose/skills, .agents/skills)
    Goose,
    /// Generic Agent Skills directory (~/.agents/skills, .agents/skills)
    Agents,
}

struct AgentSpec {
    agent: Agent,
    name: &'static str,
    label: &'static str,
    detect: &'static [&'static str],
    user_skills: &'static [&'static str],
    project_skills: &'static [&'static str],
}

const AGENTS: &[AgentSpec] = &[
    AgentSpec {
        agent: Agent::Claude,
        name: "claude",
        label: "Claude Code",
        detect: &[".claude"],
        user_skills: &[".claude", "skills"],
        project_skills: &[".claude", "skills"],
    },
    AgentSpec {
        agent: Agent::Codex,
        name: "codex",
        label: "OpenAI Codex CLI",
        detect: &[".codex"],
        user_skills: &[".codex", "skills"],
        project_skills: &[".agents", "skills"],
    },
    AgentSpec {
        agent: Agent::Pi,
        name: "pi",
        label: "Pi",
        detect: &[".pi"],
        user_skills: &[".pi", "agent", "skills"],
        project_skills: &[".pi", "skills"],
    },
    AgentSpec {
        agent: Agent::Openclaw,
        name: "openclaw",
        label: "OpenClaw",
        detect: &[".openclaw"],
        user_skills: &[".openclaw", "skills"],
        project_skills: &[".agents", "skills"],
    },
    AgentSpec {
        agent: Agent::Copilot,
        name: "copilot",
        label: "GitHub Copilot CLI",
        detect: &[".copilot"],
        user_skills: &[".copilot", "skills"],
        project_skills: &[".github", "skills"],
    },
    AgentSpec {
        agent: Agent::Cursor,
        name: "cursor",
        label: "Cursor",
        detect: &[".cursor"],
        user_skills: &[".cursor", "skills"],
        project_skills: &[".agents", "skills"],
    },
    AgentSpec {
        agent: Agent::Gemini,
        name: "gemini",
        label: "Gemini CLI",
        detect: &[".gemini"],
        user_skills: &[".gemini", "skills"],
        project_skills: &[".agents", "skills"],
    },
    AgentSpec {
        agent: Agent::Amp,
        name: "amp",
        label: "Amp",
        detect: &[".config", "amp"],
        user_skills: &[".config", "amp", "skills"],
        project_skills: &[".agents", "skills"],
    },
    AgentSpec {
        agent: Agent::Goose,
        name: "goose",
        label: "Goose",
        detect: &[".config", "goose"],
        user_skills: &[".config", "goose", "skills"],
        project_skills: &[".agents", "skills"],
    },
    AgentSpec {
        agent: Agent::Agents,
        name: "agents",
        label: "Agent Skills (generic)",
        detect: &[".agents"],
        user_skills: &[".agents", "skills"],
        project_skills: &[".agents", "skills"],
    },
];

impl Agent {
    fn spec(self) -> &'static AgentSpec {
        AGENTS
            .iter()
            .find(|spec| spec.agent == self)
            .expect("every Agent variant has a specification")
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
}

fn resolve_scope(project: bool, dir: &Option<PathBuf>) -> Result<Scope> {
    if let Some(dir) = dir {
        Ok(Scope::Project(expand_tilde(dir)?))
    } else if project {
        Ok(Scope::Project(
            std::env::current_dir().context("could not determine the current directory")?,
        ))
    } else {
        dirs::home_dir().map(Scope::User).ok_or_else(|| {
            anyhow::anyhow!(
                "could not determine the home directory (is HOME set?); use --project or --dir for a project-level install"
            )
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

fn dedupe_targets(scope: &Scope, agents: &[Agent]) -> BTreeMap<PathBuf, Vec<&'static str>> {
    let mut targets: BTreeMap<PathBuf, Vec<&'static str>> = BTreeMap::new();
    for agent in agents {
        let names = targets.entry(scope.skill_dir(*agent)).or_default();
        let name = agent.spec().name;
        if !names.contains(&name) {
            names.push(name);
        }
    }
    targets
}

fn persist_embedded_file(dir: &Path, relative: &Path, contents: &str) -> Result<PathBuf> {
    let target = dir.join(relative);
    let parent = target.parent().expect("embedded skill file has a parent");
    std::fs::create_dir_all(parent)
        .with_context(|| format!("failed to create skill directory {}", parent.display()))?;
    let temp = tempfile::NamedTempFile::new_in(parent)
        .with_context(|| format!("failed to create temporary file in {}", parent.display()))?;
    temp.as_file()
        .write_all(contents.as_bytes())
        .with_context(|| format!("failed to write skill content in {}", parent.display()))?;
    temp.persist(&target)
        .with_context(|| format!("failed to install skill at {}", target.display()))?;
    Ok(target)
}

fn write_skill_bundle(dir: &Path) -> Result<PathBuf> {
    let target = persist_embedded_file(dir, Path::new("SKILL.md"), SKILL_MD)?;
    persist_embedded_file(
        dir,
        Path::new("references/model-prompting.md"),
        MODEL_PROMPTING_MD,
    )?;
    Ok(target)
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
        let home = dirs::home_dir().context(
            "could not determine the home directory needed for --detected; name agents explicitly instead",
        )?;
        let detected: Vec<_> = AGENTS
            .iter()
            .map(|spec| spec.agent)
            .filter(|agent| *agent != Agent::Agents && agent.is_detected(&home))
            .collect();
        if detected.is_empty() {
            anyhow::bail!(
                "no supported agents were detected; name agents explicitly, use `agents` for the generic Agent Skills directory, or use --all"
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
        SkillCommand::Show => {
            print!("{SKILL_MD}");
            Ok(())
        }
    }
}

fn install(args: &SkillInstallArgs) -> Result<()> {
    let scope = resolve_scope(args.project, &args.dir)?;
    for (dir, names) in dedupe_targets(&scope, &selected_agents(args)?) {
        let target = write_skill_bundle(&dir)?;
        eprintln!(
            "Installed mold skill: {} ({})",
            display_path(&target),
            names.join(", ")
        );
    }
    Ok(())
}

fn uninstall(args: &SkillUninstallArgs) -> Result<()> {
    let scope = resolve_scope(args.project, &args.dir)?;
    let agents: Vec<_> = if args.agents.is_empty() {
        AGENTS.iter().map(|spec| spec.agent).collect()
    } else {
        args.agents.clone()
    };

    let mut removed = 0;
    for (dir, _) in dedupe_targets(&scope, &agents) {
        for relative in ["SKILL.md", "references/model-prompting.md"] {
            let target = dir.join(relative);
            if !target.exists() {
                continue;
            }
            std::fs::remove_file(&target)
                .with_context(|| format!("failed to remove {}", target.display()))?;
            removed += 1;
            eprintln!("Removed mold skill: {}", display_path(&target));
        }
        let references = dir.join("references");
        let _ = std::fs::remove_dir(&references);
        if std::fs::remove_dir(&dir).is_err() && dir.exists() {
            eprintln!("Left non-empty directory in place: {}", display_path(&dir));
        }
    }
    if removed == 0 {
        eprintln!("No installed mold skill found in the selected scope.");
    }
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
            .map(|home| status_cell(&spec.agent.user_skill_dir(home)))
            .unwrap_or_else(|| "-".to_string());
        let project = project
            .as_ref()
            .map(|root| status_cell(&spec.agent.project_skill_dir(root)))
            .unwrap_or_else(|| "-".to_string());
        table.add_row([spec.label, spec.name, detected, &user, &project]);
    }
    println!("{table}");
    Ok(())
}

fn status_cell(dir: &Path) -> String {
    let path = display_path(dir);
    if dir.join("SKILL.md").is_file() {
        format!("{path} [installed]")
    } else {
        path
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn agent_path_mapping_matches_nxv() {
        let home = Path::new("/home/u");
        let project = Path::new("/project");
        let expected = [
            (Agent::Claude, ".claude/skills", ".claude/skills"),
            (Agent::Codex, ".codex/skills", ".agents/skills"),
            (Agent::Pi, ".pi/agent/skills", ".pi/skills"),
            (Agent::Openclaw, ".openclaw/skills", ".agents/skills"),
            (Agent::Copilot, ".copilot/skills", ".github/skills"),
            (Agent::Cursor, ".cursor/skills", ".agents/skills"),
            (Agent::Gemini, ".gemini/skills", ".agents/skills"),
            (Agent::Amp, ".config/amp/skills", ".agents/skills"),
            (Agent::Goose, ".config/goose/skills", ".agents/skills"),
            (Agent::Agents, ".agents/skills", ".agents/skills"),
        ];
        assert_eq!(expected.len(), AGENTS.len());
        for (agent, user_path, project_path) in expected {
            assert_eq!(
                agent.user_skill_dir(home),
                home.join(user_path).join("mold")
            );
            assert_eq!(
                agent.project_skill_dir(project),
                project.join(project_path).join("mold")
            );
        }
    }

    #[test]
    fn shared_project_paths_are_written_once() {
        let scope = Scope::Project(PathBuf::from("/project"));
        let targets = dedupe_targets(&scope, &[Agent::Codex, Agent::Cursor, Agent::Gemini]);
        assert_eq!(targets.len(), 1);
        assert_eq!(
            targets.get(Path::new("/project/.agents/skills/mold")),
            Some(&vec!["codex", "cursor", "gemini"])
        );
    }

    #[test]
    fn write_installs_embedded_skill_bundle_without_touching_user_files() {
        let root = tempfile::tempdir().unwrap();
        let skill_dir = root.path().join("mold");
        std::fs::create_dir_all(&skill_dir).unwrap();
        std::fs::write(skill_dir.join("notes.md"), "keep").unwrap();
        std::fs::write(skill_dir.join("SKILL.md"), "stale").unwrap();

        let target = write_skill_bundle(&skill_dir).unwrap();
        assert_eq!(std::fs::read_to_string(target).unwrap(), SKILL_MD);
        assert_eq!(
            std::fs::read_to_string(skill_dir.join("references/model-prompting.md")).unwrap(),
            MODEL_PROMPTING_MD
        );
        assert_eq!(
            std::fs::read_to_string(skill_dir.join("notes.md")).unwrap(),
            "keep"
        );
    }

    #[test]
    fn embedded_skill_uses_standard_frontmatter_and_documents_installer() {
        let rest = SKILL_MD.strip_prefix("---\n").expect("frontmatter opening");
        let (frontmatter, body) = rest.split_once("\n---\n").expect("frontmatter closing");
        let mut keys = Vec::new();
        for line in frontmatter.lines() {
            let (key, value) = line
                .split_once(": ")
                .unwrap_or_else(|| panic!("invalid frontmatter line: {line:?}"));
            assert!(!value.trim().is_empty());
            keys.push(key);
            if key == "name" {
                assert_eq!(value, "mold");
            }
            if key == "description" {
                assert!((100..=1024).contains(&value.chars().count()));
            }
        }
        assert!(keys.contains(&"name"));
        assert!(keys.contains(&"description"));
        assert!(!keys.contains(&"argument-hint"));
        assert!(body.contains("mold skill install"));
    }

    #[test]
    fn prompting_reference_covers_every_generation_family() {
        let family_headings = [
            ("flux", "## FLUX.1"),
            ("flux2", "## FLUX.2"),
            ("sd15", "## SD 1.5"),
            ("sdxl", "## SDXL"),
            ("sd3", "## SD 3.5"),
            ("z-image", "## Z-Image"),
            ("wuerstchen", "## Wuerstchen v2"),
            ("qwen-image", "## Qwen-Image"),
            ("qwen-image-edit", "### Qwen-Image Edit"),
            ("ltx-video", "## LTX-Video 0.9.x"),
            ("ltx2", "## LTX-2, LTX-2.3, and LTX-2.5"),
            ("wan", "## Wan 2.1 and 2.2"),
            ("minimax-h3", "## MiniMax H3"),
            ("upscaler", "## Upscalers"),
        ];

        let manifest_families = mold_core::manifest::known_manifests()
            .iter()
            .filter(|manifest| manifest.is_generation_model() || manifest.is_upscaler())
            .map(|manifest| manifest.family.as_str())
            .collect::<std::collections::BTreeSet<_>>();
        let documented_families = family_headings
            .iter()
            .map(|(family, _)| *family)
            .collect::<std::collections::BTreeSet<_>>();

        assert_eq!(manifest_families, documented_families);
        for (family, heading) in family_headings {
            assert!(
                MODEL_PROMPTING_MD.contains(heading),
                "missing prompting section for {family}: {heading}"
            );
        }
    }
}
