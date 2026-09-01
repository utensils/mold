//! Render the prompting corpus (`mold_core::prompting`) into the user-facing
//! website page and a machine-readable export, and verify them in CI with
//! `--check` so the docs can never drift from the guides the skill bundle and
//! the expander read.

use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use mold_core::prompting::{self, FAMILY_GUIDES, MODEL_LEAVES, SHARED_PATH, TASK_LEAVES};
use mold_core::ExpandTask;
use serde::Serialize;

const MARKDOWN_PATH: &str = "website/guide/prompting.md";
const JSON_PATH: &str = "docs/generated/prompting-guides-v1.json";

#[derive(Serialize)]
struct GuideDocument<'a> {
    path: &'a str,
    contents: String,
    /// The text the expander receives for this file alone.
    expansion_excerpt: String,
}

#[derive(Serialize)]
struct FamilyDocument<'a> {
    family: &'a str,
    aliases: &'a [&'a str],
    word_limit: u32,
    guide: GuideDocument<'a>,
}

#[derive(Serialize)]
struct TaskLeafDocument<'a> {
    family: &'a str,
    label: &'a str,
    tasks: Vec<String>,
    word_limit: Option<u32>,
    standalone: bool,
    guide: GuideDocument<'a>,
}

#[derive(Serialize)]
struct ModelLeafDocument<'a> {
    family: &'a str,
    label: &'a str,
    models: &'a [&'a str],
    guide: GuideDocument<'a>,
}

#[derive(Serialize)]
struct RouteDocument {
    model: String,
    family: String,
    paths: Vec<&'static str>,
    word_limit: u32,
    excerpt_words: usize,
}

#[derive(Serialize)]
struct CorpusDocument<'a> {
    schema_version: u32,
    excerpt_word_budget: usize,
    agent_only_sections: &'a [&'a str],
    shared: GuideDocument<'a>,
    families: Vec<FamilyDocument<'a>>,
    task_leaves: Vec<TaskLeafDocument<'a>>,
    model_leaves: Vec<ModelLeafDocument<'a>>,
    routes: Vec<RouteDocument>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let check = match std::env::args().nth(1).as_deref() {
        None => false,
        Some("--check") => true,
        Some(argument) => return Err(format!("unknown argument: {argument}").into()),
    };
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("mold-core must live under crates/")
        .to_path_buf();
    let corpus = corpus_document();
    let json = format!("{}\n", serde_json::to_string_pretty(&corpus)?);
    let markdown = render_markdown(&corpus);
    update(&root.join(JSON_PATH), &json, check)?;
    update(&root.join(MARKDOWN_PATH), &markdown, check)?;
    Ok(())
}

fn task_name(task: ExpandTask) -> String {
    serde_json::to_value(task)
        .ok()
        .and_then(|value| value.as_str().map(str::to_string))
        .unwrap_or_else(|| format!("{task:?}"))
}

fn guide_document<'a>(path: &'a str, contents: &str, word_limit: u32) -> GuideDocument<'a> {
    GuideDocument {
        path,
        contents: prompting::substitute(contents, word_limit),
        expansion_excerpt: prompting::excerpt(contents, word_limit),
    }
}

fn corpus_document() -> CorpusDocument<'static> {
    let rendered = prompting::rendered_files();
    let shared_contents = rendered
        .iter()
        .find(|(path, _)| *path == SHARED_PATH)
        .map(|(_, contents)| contents.clone())
        .unwrap_or_default();
    let shared = GuideDocument {
        path: SHARED_PATH,
        expansion_excerpt: prompting::excerpt(&shared_contents, 150),
        contents: shared_contents,
    };
    let families = FAMILY_GUIDES
        .iter()
        .map(|guide| FamilyDocument {
            family: guide.family,
            aliases: guide.aliases,
            word_limit: guide.word_limit,
            guide: guide_document(guide.path, guide.contents, guide.word_limit),
        })
        .collect();
    let task_leaves = TASK_LEAVES
        .iter()
        .map(|leaf| TaskLeafDocument {
            family: leaf.family,
            label: leaf.label,
            tasks: leaf.tasks.iter().copied().map(task_name).collect(),
            word_limit: leaf.word_limit,
            standalone: leaf.standalone,
            guide: guide_document(
                leaf.path,
                leaf.contents,
                prompting::file_word_limit(leaf.path),
            ),
        })
        .collect();
    let model_leaves = MODEL_LEAVES
        .iter()
        .map(|leaf| ModelLeafDocument {
            family: leaf.family,
            label: leaf.label,
            models: leaf.models,
            guide: guide_document(
                leaf.path,
                leaf.contents,
                prompting::file_word_limit(leaf.path),
            ),
        })
        .collect();
    let mut routes = Vec::new();
    for manifest in mold_core::manifest::visible_manifests()
        .filter(|manifest| manifest.is_generation_model() || manifest.is_upscaler())
    {
        let base = mold_core::manifest::model_base_name(&manifest.name);
        if routes
            .iter()
            .any(|route: &RouteDocument| route.model == base)
        {
            continue;
        }
        let route = prompting::route(&manifest.family, Some(&manifest.name), None)
            .expect("every manifest family has a prompting guide");
        routes.push(RouteDocument {
            model: base.to_string(),
            family: manifest.family.clone(),
            paths: route.paths(),
            word_limit: route.word_limit(),
            excerpt_words: prompting::word_count(&route.expansion_excerpt()),
        });
    }
    CorpusDocument {
        schema_version: 1,
        excerpt_word_budget: prompting::EXCERPT_WORD_BUDGET,
        agent_only_sections: prompting::AGENT_ONLY_SECTIONS,
        shared,
        families,
        task_leaves,
        model_leaves,
        routes,
    }
}

fn demote_headings(markdown: &str, by: usize) -> String {
    let mut out = String::new();
    let mut in_fence = false;
    for line in markdown.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
            in_fence = !in_fence;
        }
        if !in_fence && line.starts_with('#') {
            let hashes = line.chars().take_while(|c| *c == '#').count();
            let rest = &line[hashes..];
            let depth = (hashes + by).min(6);
            out.push_str(&"#".repeat(depth));
            out.push_str(rest);
        } else {
            out.push_str(line);
        }
        out.push('\n');
    }
    out
}

fn render_markdown(corpus: &CorpusDocument<'_>) -> String {
    let mut out = String::new();
    out.push_str("---\nlayout: doc\n---\n\n");
    out.push_str("<!-- GENERATED FILE. Do not edit by hand.\n");
    out.push_str("     Source: crates/mold-core/src/prompting/ (the prompting corpus).\n");
    out.push_str("     Regenerate: cargo run -p mold-ai-core --bin generate_prompting_guides\n");
    out.push_str("     Verified in CI with --check. -->\n\n");
    out.push_str("# Prompting Guides\n\n");
    out.push_str(
        "Every guide on this page comes from one corpus in the Mold source tree. The same \
         files are installed with the agent skill (`mold skill install`), published to MCP \
         hosts as `mold://prompting/` resources, and injected into the prompt expander that \
         powers `mold expand`, `mold remix`, `--expand`, and the Expand and Remix actions \
         in the desktop, web, and iPhone apps. The expander receives every section except \
         `CLI` and `Sources`, plus a generation-context block naming the exact model, canvas, \
         frame count, fps, duration, and ordered references.\n\n",
    );
    writeln!(
        out,
        "The expander budget is {} words per route. Word limits below are the corpus defaults; \
         `[expand.families.<family>] word_limit` in `config.toml` overrides them, and \
         `style_notes` replaces the guide text entirely.\n",
        corpus.excerpt_word_budget
    )
    .unwrap();

    out.push_str("## Routes by model\n\n");
    out.push_str("| Model | Family | Guides read in order | Word limit | Excerpt words |\n");
    out.push_str("| --- | --- | --- | --- | --- |\n");
    for route in &corpus.routes {
        writeln!(
            out,
            "| `{}` | `{}` | {} | {} | {} |",
            route.model,
            route.family,
            route
                .paths
                .iter()
                .map(|path| format!("`{path}`"))
                .collect::<Vec<_>>()
                .join(", "),
            route.word_limit,
            route.excerpt_words
        )
        .unwrap();
    }
    out.push('\n');

    out.push_str("## Shared practice\n\n");
    out.push_str(&demote_headings(&corpus.shared.contents, 2));
    out.push('\n');

    out.push_str("## Families\n\n");
    for family in &corpus.families {
        let aliases = if family.aliases.is_empty() {
            String::new()
        } else {
            format!(
                " Also accepted on the wire as {}.",
                family
                    .aliases
                    .iter()
                    .map(|alias| format!("`{alias}`"))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        };
        writeln!(
            out,
            "<!-- {} -->\n\nDefault word limit: {}.{aliases}\n",
            family.guide.path, family.word_limit
        )
        .unwrap();
        out.push_str(&demote_headings(&family.guide.contents, 2));
        out.push('\n');
    }

    if !corpus.task_leaves.is_empty() {
        out.push_str("## Task leaves\n\n");
        out.push_str(
            "A task leaf is added below its family base when the model identity or the \
             expansion task selects it.\n\n",
        );
        for leaf in &corpus.task_leaves {
            writeln!(
                out,
                "<!-- {} -->\n\nFamily `{}`; tasks: {}; word limit: {}.\n",
                leaf.guide.path,
                leaf.family,
                if leaf.tasks.is_empty() {
                    "explicit selection only".to_string()
                } else {
                    leaf.tasks
                        .iter()
                        .map(|task| format!("`{task}`"))
                        .collect::<Vec<_>>()
                        .join(", ")
                },
                leaf.word_limit
                    .map_or_else(|| "family default".to_string(), |limit| limit.to_string())
            )
            .unwrap();
            out.push_str(&demote_headings(&leaf.guide.contents, 2));
            out.push('\n');
        }
    }

    if !corpus.model_leaves.is_empty() {
        out.push_str("## Model leaves\n\n");
        out.push_str(
            "A model leaf carries quirks of one checkpoint and is added after the task leaf.\n\n",
        );
        for leaf in &corpus.model_leaves {
            writeln!(
                out,
                "<!-- {} -->\n\nModels: {}.\n",
                leaf.guide.path,
                leaf.models
                    .iter()
                    .map(|model| format!("`{model}`"))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
            .unwrap();
            out.push_str(&demote_headings(&leaf.guide.contents, 2));
            out.push('\n');
        }
    }
    out
}

fn update(path: &Path, contents: &str, check: bool) -> Result<(), Box<dyn std::error::Error>> {
    if check {
        let existing = fs::read_to_string(path)
            .map_err(|error| format!("{} is missing or unreadable: {error}", path.display()))?;
        if existing != contents {
            return Err(format!(
                "{} is stale; run the generation command without --check",
                path.display()
            )
            .into());
        }
        return Ok(());
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, contents)?;
    println!("generated {}", path.display());
    Ok(())
}
