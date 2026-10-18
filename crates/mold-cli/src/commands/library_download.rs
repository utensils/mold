//! `mold library download` — fetch prints' original files from the host.
//!
//! Names resolve against the host's listing (a unique prefix is enough, an
//! exact name always wins), every destination is checked before anything is
//! fetched, and each file streams to `<dest>.part`, is synced, then renamed,
//! so an interrupted download never leaves a file that looks complete.

use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{bail, Context, Result};
use mold_core::{GalleryImage, MoldClient};

/// How many matches an ambiguous prefix lists.
const SHOWN_MATCHES: usize = 5;

pub(crate) struct Request {
    pub filenames: Vec<String>,
    pub output: Option<String>,
    pub latest: Option<usize>,
    pub trash: bool,
    pub force: bool,
}

pub(crate) async fn run(client: &MoldClient, request: Request) -> Result<()> {
    let view = if request.trash { "trash" } else { "library" };
    let rows = if request.trash {
        client.list_gallery_view("trash").await
    } else {
        client.list_gallery().await
    }
    .with_context(|| format!("could not list the {view} on {}", client.host()))?;
    let names = match request.latest {
        Some(count) => latest(&rows, count)?,
        None => resolve_names(&rows, &request.filenames)?,
    };
    let plan = plan_destinations(&names, request.output.as_deref())?;
    match plan {
        Plan::Stdout(name) => download_to_stdout(client, &name, request.trash).await,
        Plan::Files(files) => download_files(client, files, request.trash, request.force).await,
    }
}

/// The listing's filename for each input: an exact name, else the one name
/// it is a prefix of. An ambiguous prefix lists a few of its matches.
fn resolve_names(rows: &[GalleryImage], inputs: &[String]) -> Result<Vec<String>> {
    let mut names = Vec::with_capacity(inputs.len());
    for input in inputs {
        if rows.iter().any(|row| &row.filename == input) {
            names.push(input.clone());
            continue;
        }
        let matches: Vec<&str> = rows
            .iter()
            .map(|row| row.filename.as_str())
            .filter(|name| name.starts_with(input.as_str()))
            .collect();
        match matches.as_slice() {
            [one] => names.push((*one).to_string()),
            [] => bail!("no print named or starting with '{input}' (see `mold library list`)"),
            many => bail!(
                "'{input}' matches {} prints: {}{}; give more of the name",
                many.len(),
                many.iter()
                    .take(SHOWN_MATCHES)
                    .copied()
                    .collect::<Vec<_>>()
                    .join(", "),
                if many.len() > SHOWN_MATCHES {
                    ", …"
                } else {
                    ""
                }
            ),
        }
    }
    // Asking twice for one print would write it twice to the same place.
    let mut seen = std::collections::HashSet::new();
    names.retain(|name| seen.insert(name.clone()));
    Ok(names)
}

/// The newest `count` prints, newest first (the `mold library list` order).
fn latest(rows: &[GalleryImage], count: usize) -> Result<Vec<String>> {
    anyhow::ensure!(count > 0, "--latest needs at least 1");
    let mut rows: Vec<&GalleryImage> = rows.iter().collect();
    anyhow::ensure!(!rows.is_empty(), "no prints to download");
    rows.sort_by(|a, b| {
        b.timestamp
            .cmp(&a.timestamp)
            .then_with(|| a.filename.cmp(&b.filename))
    });
    Ok(rows
        .into_iter()
        .take(count)
        .map(|row| row.filename.clone())
        .collect())
}

#[derive(Debug, PartialEq, Eq)]
enum Plan {
    Stdout(String),
    Files(Vec<(String, PathBuf)>),
}

/// Where each print goes: the current directory under its own name by
/// default; `-o -` is stdout (one print); `-o dir/` or an existing directory
/// holds them all; any other `-o` is the one print's file.
fn plan_destinations(names: &[String], output: Option<&str>) -> Result<Plan> {
    for name in names {
        anyhow::ensure!(
            safe_name(name),
            "the host named a print '{name}', which is not a plain file name"
        );
    }
    let into = |dir: &Path| {
        Plan::Files(
            names
                .iter()
                .map(|name| (name.clone(), dir.join(name)))
                .collect(),
        )
    };
    let Some(output) = output else {
        return Ok(into(Path::new(".")));
    };
    if output == "-" {
        return match names {
            [one] => Ok(Plan::Stdout(one.clone())),
            _ => bail!("-o - (stdout) takes one print; give -o a directory for several"),
        };
    }
    let path = Path::new(output);
    if output.ends_with('/') || output.ends_with(std::path::MAIN_SEPARATOR) || path.is_dir() {
        return Ok(into(path));
    }
    match names {
        [one] => Ok(Plan::Files(vec![(one.clone(), path.to_path_buf())])),
        _ => bail!(
            "-o {output} is not a directory; several prints need one (end it with / to create it)"
        ),
    }
}

/// A print name the host returned is used as a file name only when it is
/// one: no directories, nothing hidden or relative.
fn safe_name(name: &str) -> bool {
    !name.is_empty()
        && name != "."
        && name != ".."
        && !name.contains(['/', '\\', '\0'])
        && Path::new(name).file_name().is_some_and(|f| f == name)
}

async fn download_to_stdout(client: &MoldClient, name: &str, trash: bool) -> Result<()> {
    let mut stdout = tokio::io::stdout();
    let bar = Progress::new(name, false);
    client
        .download_gallery_image(name, trash, &mut stdout, |done, total| {
            bar.update(done, total)
        })
        .await
        .with_context(|| format!("could not download {name}"))?;
    bar.finish();
    Ok(())
}

async fn download_files(
    client: &MoldClient,
    files: Vec<(String, PathBuf)>,
    trash: bool,
    force: bool,
) -> Result<()> {
    // Refuse every existing destination before fetching anything.
    let existing: Vec<String> = files
        .iter()
        .filter(|(_, dest)| dest.exists())
        .map(|(_, dest)| dest.display().to_string())
        .collect();
    if !existing.is_empty() && !force {
        bail!(
            "{} already exist{}: {}; pass --force to overwrite",
            if existing.len() == 1 {
                "a file"
            } else {
                "files"
            },
            if existing.len() == 1 { "s" } else { "" },
            existing.join(", ")
        );
    }
    let total = files.len();
    let mut failed = Vec::new();
    for (name, dest) in files {
        match download_one(client, &name, &dest, trash).await {
            Ok(size) => eprintln!(
                "Saved {} ({})",
                dest.display(),
                crate::ui::format_bytes(size)
            ),
            Err(error) => {
                eprintln!("{} {name}: {error:#}", crate::theme::prefix_error());
                failed.push(name);
            }
        }
    }
    if !failed.is_empty() {
        bail!(
            "{} of {total} downloads failed: {}",
            failed.len(),
            failed.join(", ")
        );
    }
    Ok(())
}

/// Stream one print to `<dest>.part`, sync it, and rename it into place.
/// The partial file is removed on any failure, and by the Ctrl-C handler
/// while one is in flight.
async fn download_one(client: &MoldClient, name: &str, dest: &Path, trash: bool) -> Result<u64> {
    if let Some(parent) = dest.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("could not create {}", parent.display()))?;
    }
    let part = partial_path(dest);
    let result = async {
        let mut file = tokio::fs::File::create(&part)
            .await
            .with_context(|| format!("could not create {}", part.display()))?;
        set_in_flight(Some(part.clone()));
        let bar = Progress::new(name, true);
        let size = client
            .download_gallery_image(name, trash, &mut file, |done, total| {
                bar.update(done, total)
            })
            .await?;
        bar.finish();
        file.sync_all().await?;
        drop(file);
        tokio::fs::rename(&part, dest)
            .await
            .with_context(|| format!("could not move the download into {}", dest.display()))?;
        Ok(size)
    }
    .await;
    set_in_flight(None);
    if result.is_err() {
        let _ = std::fs::remove_file(&part);
    }
    result
}

fn partial_path(dest: &Path) -> PathBuf {
    let mut name = dest.file_name().unwrap_or_default().to_os_string();
    name.push(".part");
    dest.with_file_name(name)
}

/// The `.part` file being written right now, for the Ctrl-C handler.
static IN_FLIGHT: Mutex<Option<PathBuf>> = Mutex::new(None);

fn set_in_flight(path: Option<PathBuf>) {
    *IN_FLIGHT.lock().unwrap_or_else(|e| e.into_inner()) = path;
}

/// Delete a partial download interrupted by Ctrl-C (called from the
/// process's signal handler; never blocks on the lock).
pub(crate) fn remove_partial_download() {
    if let Ok(mut guard) = IN_FLIGHT.try_lock() {
        if let Some(path) = guard.take() {
            let _ = std::fs::remove_file(path);
        }
    }
}

/// A byte progress bar on a terminal stderr; nothing otherwise.
struct Progress(Option<indicatif::ProgressBar>);

impl Progress {
    fn new(name: &str, to_file: bool) -> Self {
        // Writing to stdout means stdout is the data; only stderr shows a bar.
        if !std::io::stderr().is_terminal() || (to_file && !std::io::stdout().is_terminal()) {
            return Self(None);
        }
        let bar =
            indicatif::ProgressBar::with_draw_target(None, indicatif::ProgressDrawTarget::stderr());
        bar.set_message(name.to_string());
        Self(Some(bar))
    }

    fn update(&self, done: u64, total: Option<u64>) {
        let Some(bar) = &self.0 else { return };
        if bar.length().is_none() {
            if let Some(total) = total {
                bar.set_length(total);
                bar.set_style(
                    indicatif::ProgressStyle::with_template(
                        "{msg} [{bar:30}] {bytes}/{total_bytes} {bytes_per_sec}",
                    )
                    .unwrap_or_else(|_| indicatif::ProgressStyle::default_bar())
                    .progress_chars("=> "),
                );
            } else {
                bar.set_style(
                    indicatif::ProgressStyle::with_template("{msg} {bytes} {bytes_per_sec}")
                        .unwrap_or_else(|_| indicatif::ProgressStyle::default_spinner()),
                );
            }
        }
        bar.set_position(done);
    }

    fn finish(&self) {
        if let Some(bar) = &self.0 {
            bar.finish_and_clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(name: &str, timestamp: u64) -> GalleryImage {
        serde_json::from_value(serde_json::json!({
            "filename": name,
            "metadata": {
                "prompt": "p", "model": "m", "seed": 1, "steps": 1, "guidance": 1.0,
                "width": 8, "height": 8, "version": "t"
            },
            "timestamp": timestamp
        }))
        .unwrap()
    }

    fn rows() -> Vec<GalleryImage> {
        vec![
            row("mold-flux-owl-1.png", 1),
            row("mold-flux-owl-1.png.webp", 5),
            row("mold-flux-owl-2.png", 2),
            row("mold-wan-clip-3.mp4", 3),
        ]
    }

    fn names(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn names_resolve_exactly_then_by_a_unique_prefix() {
        let rows = rows();
        // An exact name wins even when it is also a prefix of another.
        assert_eq!(
            resolve_names(&rows, &names(&["mold-flux-owl-1.png"])).unwrap(),
            ["mold-flux-owl-1.png"]
        );
        assert_eq!(
            resolve_names(&rows, &names(&["mold-wan"])).unwrap(),
            ["mold-wan-clip-3.mp4"]
        );
        let ambiguous = resolve_names(&rows, &names(&["mold-flux"]))
            .unwrap_err()
            .to_string();
        assert!(ambiguous.contains("matches 3 prints"), "{ambiguous}");
        assert!(ambiguous.contains("mold-flux-owl-2.png"), "{ambiguous}");
        let none = resolve_names(&rows, &names(&["zzz"]))
            .unwrap_err()
            .to_string();
        assert!(none.contains("'zzz'"), "{none}");
        assert_eq!(
            resolve_names(&rows, &names(&["mold-wan", "mold-wan-clip-3.mp4"])).unwrap(),
            ["mold-wan-clip-3.mp4"],
            "the same print twice downloads once"
        );
    }

    #[test]
    fn an_ambiguous_prefix_lists_at_most_five() {
        let many: Vec<GalleryImage> = (0..8).map(|i| row(&format!("a-{i}.png"), i)).collect();
        let error = resolve_names(&many, &names(&["a-"]))
            .unwrap_err()
            .to_string();
        assert!(error.contains("matches 8 prints") && error.ends_with("give more of the name"));
        assert_eq!(error.matches(".png").count(), SHOWN_MATCHES);
    }

    #[test]
    fn latest_is_newest_first() {
        assert_eq!(
            latest(&rows(), 2).unwrap(),
            ["mold-flux-owl-1.png.webp", "mold-wan-clip-3.mp4"]
        );
        assert!(latest(&rows(), 0).is_err());
        assert!(latest(&[], 1).is_err());
    }

    #[test]
    fn destinations_follow_dash_o() {
        let one = names(&["a.png"]);
        let two = names(&["a.png", "b.mp4"]);
        assert_eq!(
            plan_destinations(&one, None).unwrap(),
            Plan::Files(vec![("a.png".into(), PathBuf::from("./a.png"))])
        );
        assert_eq!(
            plan_destinations(&one, Some("out.png")).unwrap(),
            Plan::Files(vec![("a.png".into(), PathBuf::from("out.png"))])
        );
        assert_eq!(
            plan_destinations(&two, Some("renders/")).unwrap(),
            Plan::Files(vec![
                ("a.png".into(), PathBuf::from("renders/a.png")),
                ("b.mp4".into(), PathBuf::from("renders/b.mp4")),
            ])
        );
        let dir = tempfile::tempdir().unwrap();
        let existing = dir.path().to_str().unwrap();
        assert_eq!(
            plan_destinations(&one, Some(existing)).unwrap(),
            Plan::Files(vec![("a.png".into(), dir.path().join("a.png"))])
        );
        assert_eq!(
            plan_destinations(&one, Some("-")).unwrap(),
            Plan::Stdout("a.png".into())
        );
        assert!(plan_destinations(&two, Some("-")).is_err());
        assert!(plan_destinations(&two, Some("one.png")).is_err());
        for bad in ["../x.png", "dir/x.png", "..", ""] {
            assert!(plan_destinations(&names(&[bad]), None).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn the_partial_file_sits_beside_its_destination() {
        assert_eq!(
            partial_path(Path::new("renders/a.png")),
            PathBuf::from("renders/a.png.part")
        );
    }
}
