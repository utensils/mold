//! Claude Desktop: a skill ZIP for **Customize > Skills** plus the local
//! `mold mcp` connector in `claude_desktop_config.json`.
//!
//! Desktop keeps skills on the claude.ai account, so the upload is the one
//! step a person has to do; chat skills run in Anthropic's cloud sandbox,
//! which has no `mold` binary, so the connector is how Desktop reaches mold.

use anyhow::{bail, Context, Result};
use serde_json::{json, Map, Value};
use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use super::SKILL_DIR_NAME;

/// Where Desktop and claude.ai list, upload and turn on skills.
pub(super) const UPLOAD_URL: &str = "https://claude.ai/customize/skills";
/// The default ZIP file name.
pub(super) const ZIP_NAME: &str = "mold-skill.zip";
/// The `mcpServers` key mold owns.
const CONNECTOR: &str = "mold";
const CONFIG_FILE: &str = "claude_desktop_config.json";
const BACKUP_SUFFIX: &str = ".mold-backup";

/// Claude Desktop's config directory: `~/Library/Application Support/Claude`
/// on macOS, `%APPDATA%\Claude` on Windows. Desktop has no Linux build.
pub(super) fn config_dir(home: &Path, appdata: Option<&Path>) -> Option<PathBuf> {
    if cfg!(target_os = "macos") {
        Some(home.join("Library/Application Support/Claude"))
    } else if cfg!(windows) {
        appdata.map(|appdata| appdata.join("Claude"))
    } else {
        None
    }
}

pub(super) fn config_path(dir: &Path) -> PathBuf {
    dir.join(CONFIG_FILE)
}

/// Where the ZIP goes without `--output`: Downloads, else the current directory.
pub(super) fn default_zip_path(downloads: Option<PathBuf>, cwd: &Path) -> PathBuf {
    downloads
        .filter(|dir| dir.is_dir())
        .unwrap_or_else(|| cwd.to_path_buf())
        .join(ZIP_NAME)
}

// ── ZIP ────────────────────────────────────────────────────────────────────

/// Write `files` as `mold/<path>` entries. Desktop looks for
/// `<skill-name>/SKILL.md`, so the skill folder must be the archive's top
/// level. Entries are stored (no compression), sorted and stamped
/// 1980-01-01, so the same bundle always produces the same bytes.
pub(super) fn zip_bytes(files: &BTreeMap<String, String>) -> Result<Vec<u8>> {
    use zip::{write::SimpleFileOptions, CompressionMethod, DateTime, ZipWriter};
    let options = SimpleFileOptions::default()
        .compression_method(CompressionMethod::Stored)
        .last_modified_time(DateTime::DEFAULT)
        .unix_permissions(0o644);
    let mut writer = ZipWriter::new(std::io::Cursor::new(Vec::new()));
    writer.add_directory(
        format!("{SKILL_DIR_NAME}/"),
        options.unix_permissions(0o755),
    )?;
    for (path, contents) in files {
        writer.start_file(format!("{SKILL_DIR_NAME}/{path}"), options)?;
        writer.write_all(contents.as_bytes())?;
    }
    Ok(writer.finish()?.into_inner())
}

/// True for a mold skill ZIP: every entry under `mold/`, with `mold/SKILL.md`.
/// Judged from the entry names alone, so a ZIP someone made by hand with
/// `zip -r` (deflated, which this store-only build cannot decompress) counts.
pub(super) fn is_mold_skill_zip(path: &Path) -> bool {
    let Some(archive) = std::fs::File::open(path)
        .ok()
        .and_then(|file| zip::ZipArchive::new(file).ok())
    else {
        return false;
    };
    let prefix = format!("{SKILL_DIR_NAME}/");
    let skill = format!("{prefix}SKILL.md");
    let mut names = archive.file_names().peekable();
    names.peek().is_some()
        && archive.file_names().any(|name| name == skill)
        && names.all(|name| name.starts_with(&prefix) || name.starts_with("__MACOSX/"))
}

/// The manifest inside a ZIP mold wrote, to tell a current ZIP from an old
/// one; `None` for any other ZIP.
pub(super) fn zipped_manifest(path: &Path) -> Option<String> {
    let mut archive = zip::ZipArchive::new(std::fs::File::open(path).ok()?).ok()?;
    let mut entry = archive
        .by_name(&format!("{SKILL_DIR_NAME}/{}", super::MANIFEST_FILE))
        .ok()?;
    let mut manifest = String::new();
    entry.read_to_string(&mut manifest).ok()?;
    Some(manifest)
}

/// Write the ZIP atomically. An existing file is replaced only when it is a
/// mold skill ZIP, or with `force`.
pub(super) fn write_zip(files: &BTreeMap<String, String>, path: &Path, force: bool) -> Result<()> {
    if path.is_dir() {
        bail!(
            "{} is a directory; pass a file path such as {}",
            path.display(),
            path.join(ZIP_NAME).display()
        );
    }
    if path.exists() && !force && !is_mold_skill_zip(path) {
        bail!(
            "refusing to overwrite {}, which is not a mold skill ZIP; pass --force or choose another --output",
            path.display()
        );
    }
    let parent = match path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent,
        _ => Path::new("."),
    };
    let mut temp = tempfile::NamedTempFile::new_in(parent)
        .with_context(|| format!("failed to create a file in {}", parent.display()))?;
    temp.write_all(&zip_bytes(files)?)?;
    temp.as_file().sync_all()?;
    temp.persist(path)
        .with_context(|| format!("failed to write {}", path.display()))?;
    Ok(())
}

// ── Connector ──────────────────────────────────────────────────────────────

/// The `mcpServers.mold` entry: the mold binary started as `mold mcp`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Connector {
    pub command: String,
    pub args: Vec<String>,
    pub env: BTreeMap<String, String>,
}

impl Connector {
    pub fn new(command: String, host: Option<&str>, mold_home: Option<String>) -> Self {
        let mut args = vec!["mcp".to_string()];
        if let Some(host) = host {
            args.extend(["--host".to_string(), host.to_string()]);
        }
        // Desktop starts servers without the login shell's environment, so a
        // non-default MOLD_HOME (where the saved hosts live) is passed through.
        let env = mold_home
            .map(|home| BTreeMap::from([("MOLD_HOME".to_string(), home)]))
            .unwrap_or_default();
        Self { command, args, env }
    }

    fn entry(&self) -> Value {
        let mut entry = json!({ "command": self.command, "args": self.args });
        if !self.env.is_empty() {
            entry["env"] = json!(self.env);
        }
        entry
    }
}

/// True when an `mcpServers` entry is one mold wrote: the mold binary run as
/// `mold mcp`. Anything else under the `mold` key belongs to the user.
fn is_ours(entry: &Value) -> bool {
    let binary = entry
        .get("command")
        .and_then(Value::as_str)
        .map(|command| {
            // Split by hand: a config copied from Windows keeps `\` separators,
            // which `Path` treats as ordinary characters elsewhere.
            let file = command.rsplit(['/', '\\']).next().unwrap_or(command);
            file.eq_ignore_ascii_case("mold") || file.eq_ignore_ascii_case("mold.exe")
        })
        .unwrap_or(false);
    let runs_mcp = entry
        .get("args")
        .and_then(Value::as_array)
        .and_then(|args| args.first())
        .and_then(Value::as_str)
        == Some("mcp");
    binary && runs_mcp
}

#[derive(Debug, PartialEq, Eq)]
pub(super) enum Merge {
    Added,
    Updated,
    Unchanged,
}

fn servers(config: &mut Value) -> Result<&mut Map<String, Value>> {
    let Some(object) = config.as_object_mut() else {
        bail!("{CONFIG_FILE} is not a JSON object");
    };
    let servers = object
        .entry("mcpServers")
        .or_insert_with(|| Value::Object(Map::new()));
    servers
        .as_object_mut()
        .with_context(|| format!("mcpServers in {CONFIG_FILE} is not a JSON object"))
}

/// Add or update `mcpServers.mold`, leaving every other key alone.
pub(super) fn merge_connector(
    config: &mut Value,
    connector: &Connector,
    force: bool,
) -> Result<Merge> {
    let servers = servers(config)?;
    let entry = connector.entry();
    let merge = match servers.get(CONNECTOR) {
        None => Merge::Added,
        Some(existing) if *existing == entry => return Ok(Merge::Unchanged),
        Some(existing) if is_ours(existing) || force => Merge::Updated,
        Some(_) => bail!(
            "{CONFIG_FILE} already has an mcpServers.{CONNECTOR} entry that mold did not write; pass --force to replace it, or --no-connector to leave it"
        ),
    };
    servers.insert(CONNECTOR.to_string(), entry);
    Ok(merge)
}

#[derive(Debug, PartialEq, Eq)]
pub(super) enum Removal {
    Removed,
    Absent,
    /// The `mold` key holds something mold did not write; it is left alone.
    NotOurs,
}

/// Remove `mcpServers.mold` when mold wrote it.
pub(super) fn remove_connector(config: &mut Value) -> Result<Removal> {
    let Some(servers) = config
        .as_object_mut()
        .and_then(|object| object.get_mut("mcpServers"))
        .and_then(Value::as_object_mut)
    else {
        return Ok(Removal::Absent);
    };
    match servers.get(CONNECTOR) {
        None => Ok(Removal::Absent),
        Some(entry) if !is_ours(entry) => Ok(Removal::NotOurs),
        Some(_) => {
            servers.remove(CONNECTOR);
            Ok(Removal::Removed)
        }
    }
}

/// What `mcpServers.mold` holds right now, for `mold skill list`.
pub(super) fn connector_state(config: &Value) -> &'static str {
    match config
        .get("mcpServers")
        .and_then(|servers| servers.get(CONNECTOR))
    {
        None => "no connector",
        Some(entry) if is_ours(entry) => "connector",
        Some(_) => "connector (not mold's)",
    }
}

/// Read the config; a missing or blank file is an empty object. A file that
/// does not parse is refused rather than replaced.
pub(super) fn read_config(path: &Path) -> Result<Value> {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(json!({})),
        Err(error) => {
            return Err(error).with_context(|| format!("failed to read {}", path.display()))
        }
    };
    if text.trim().is_empty() {
        return Ok(json!({}));
    }
    serde_json::from_str(&text).with_context(|| {
        format!(
            "{} is not valid JSON; fix it (or move it aside) and run this again",
            path.display()
        )
    })
}

/// Write the config atomically, keeping the previous file as
/// `claude_desktop_config.json.mold-backup`.
pub(super) fn write_config(path: &Path, config: &Value) -> Result<Option<PathBuf>> {
    let dir = path
        .parent()
        .with_context(|| format!("{} has no parent directory", path.display()))?;
    let backup = if path.exists() {
        let mut name = path.as_os_str().to_os_string();
        name.push(BACKUP_SUFFIX);
        let backup = PathBuf::from(name);
        std::fs::copy(path, &backup)
            .with_context(|| format!("failed to back up {}", path.display()))?;
        Some(backup)
    } else {
        None
    };
    let mut temp = tempfile::NamedTempFile::new_in(dir)
        .with_context(|| format!("failed to create a file in {}", dir.display()))?;
    temp.write_all(format!("{}\n", serde_json::to_string_pretty(config)?).as_bytes())?;
    temp.as_file().sync_all()?;
    temp.persist(path)
        .with_context(|| format!("failed to write {}", path.display()))?;
    Ok(backup)
}

// ── Opening the upload page ────────────────────────────────────────────────

/// Open the Skills page and show the ZIP in the file manager. Best effort:
/// the printed steps say the same thing.
pub(super) fn open_upload_page(zip: &Path) {
    use std::process::{Command, Stdio};
    let run = |program: &str, args: &[&std::ffi::OsStr]| {
        let _ = Command::new(program)
            .args(args)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    };
    let url = std::ffi::OsStr::new(UPLOAD_URL);
    if cfg!(target_os = "macos") {
        run("open", &[url]);
        run("open", &["-R".as_ref(), zip.as_os_str()]);
    } else if cfg!(windows) {
        run("cmd", &["/C".as_ref(), "start".as_ref(), "".as_ref(), url]);
        let mut select = std::ffi::OsString::from("/select,");
        select.push(zip.as_os_str());
        run("explorer", &[select.as_os_str()]);
    } else {
        run("xdg-open", &[url]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn files() -> BTreeMap<String, String> {
        BTreeMap::from([
            (
                "SKILL.md".to_string(),
                "---\nname: mold\ndescription: d\n---\n".to_string(),
            ),
            ("references/cli.md".to_string(), "# CLI\n".to_string()),
        ])
    }

    fn entries(bytes: &[u8]) -> Vec<String> {
        let archive = zip::ZipArchive::new(std::io::Cursor::new(bytes)).unwrap();
        archive.file_names().map(str::to_string).collect()
    }

    #[test]
    fn the_zip_has_the_skill_folder_at_its_top_level() {
        let bytes = zip_bytes(&files()).unwrap();
        let mut names = entries(&bytes);
        names.sort();
        assert_eq!(names, ["mold/", "mold/SKILL.md", "mold/references/cli.md"]);
    }

    #[test]
    fn the_same_bundle_zips_to_the_same_bytes() {
        assert_eq!(zip_bytes(&files()).unwrap(), zip_bytes(&files()).unwrap());
    }

    #[test]
    fn an_old_mold_zip_is_replaced_but_another_file_is_not() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(ZIP_NAME);
        write_zip(&files(), &path, false).unwrap();
        assert!(is_mold_skill_zip(&path));
        let mut newer = files();
        newer.insert("examples/quickstart.md".to_string(), "new\n".to_string());
        write_zip(&newer, &path, false).unwrap();
        assert!(entries(&std::fs::read(&path).unwrap())
            .contains(&"mold/examples/quickstart.md".to_string()));

        let other = dir.path().join("other.zip");
        std::fs::write(&other, b"not mine").unwrap();
        let error = write_zip(&files(), &other, false).unwrap_err().to_string();
        assert!(error.contains("not a mold skill ZIP"), "{error}");
        assert_eq!(std::fs::read(&other).unwrap(), b"not mine");
        write_zip(&files(), &other, true).unwrap();
        assert!(is_mold_skill_zip(&other));
    }

    #[test]
    fn a_zip_made_by_hand_counts_by_its_layout() {
        use zip::{write::SimpleFileOptions, ZipWriter};
        let dir = tempfile::tempdir().unwrap();
        let write = |name: &str, entries: &[&str]| {
            let path = dir.path().join(name);
            let mut writer = ZipWriter::new(std::fs::File::create(&path).unwrap());
            for entry in entries {
                writer
                    .start_file(*entry, SimpleFileOptions::default())
                    .unwrap();
                writer.write_all(b"x").unwrap();
            }
            writer.finish().unwrap();
            path
        };
        let by_hand = write(
            "a.zip",
            &[
                "mold/SKILL.md",
                "mold/references/cli.md",
                "__MACOSX/mold/._SKILL.md",
            ],
        );
        assert!(is_mold_skill_zip(&by_hand));
        assert_eq!(zipped_manifest(&by_hand), None, "not one mold wrote");
        assert!(!is_mold_skill_zip(&write("b.zip", &["SKILL.md"])));
        assert!(!is_mold_skill_zip(&write(
            "c.zip",
            &["mold/SKILL.md", "notes.txt"]
        )));
        assert!(!is_mold_skill_zip(&write("d.zip", &[])));
    }

    #[test]
    fn a_directory_output_is_refused_with_a_file_suggestion() {
        let dir = tempfile::tempdir().unwrap();
        let error = write_zip(&files(), dir.path(), false)
            .unwrap_err()
            .to_string();
        assert!(error.contains("is a directory"), "{error}");
    }

    #[test]
    fn the_default_zip_goes_to_downloads_when_it_exists() {
        let dir = tempfile::tempdir().unwrap();
        let cwd = dir.path().join("cwd");
        let downloads = dir.path().join("Downloads");
        assert_eq!(
            default_zip_path(Some(downloads.clone()), &cwd),
            cwd.join(ZIP_NAME)
        );
        std::fs::create_dir(&downloads).unwrap();
        assert_eq!(
            default_zip_path(Some(downloads.clone()), &cwd),
            downloads.join(ZIP_NAME)
        );
        assert_eq!(default_zip_path(None, &cwd), cwd.join(ZIP_NAME));
    }

    fn connector() -> Connector {
        Connector::new("/opt/bin/mold".to_string(), None, None)
    }

    #[test]
    fn the_connector_is_added_beside_other_servers_and_settings() {
        let mut config = json!({
            "mcpServers": { "time": { "command": "uvx", "args": ["mcp-server-time"] } },
            "preferences": { "sidebarMode": "chat" }
        });
        assert_eq!(
            merge_connector(&mut config, &connector(), false).unwrap(),
            Merge::Added
        );
        assert_eq!(config["mcpServers"]["time"]["command"], "uvx");
        assert_eq!(config["preferences"]["sidebarMode"], "chat");
        assert_eq!(
            config["mcpServers"]["mold"],
            json!({ "command": "/opt/bin/mold", "args": ["mcp"] })
        );
        assert_eq!(
            merge_connector(&mut config, &connector(), false).unwrap(),
            Merge::Unchanged
        );
    }

    #[test]
    fn an_empty_config_gets_an_mcp_servers_object() {
        let mut config = json!({});
        merge_connector(&mut config, &connector(), false).unwrap();
        assert!(config["mcpServers"]["mold"].is_object());
    }

    #[test]
    fn a_pinned_host_and_mold_home_are_passed_to_mold_mcp() {
        let pinned = Connector::new(
            "/opt/bin/mold".to_string(),
            Some("workstation"),
            Some("/Volumes/Data/mold".to_string()),
        );
        let mut config = json!({});
        merge_connector(&mut config, &connector(), false).unwrap();
        assert_eq!(
            merge_connector(&mut config, &pinned, false).unwrap(),
            Merge::Updated
        );
        assert_eq!(
            config["mcpServers"]["mold"],
            json!({
                "command": "/opt/bin/mold",
                "args": ["mcp", "--host", "workstation"],
                "env": { "MOLD_HOME": "/Volumes/Data/mold" }
            })
        );
    }

    #[test]
    fn a_mold_entry_mold_did_not_write_needs_force() {
        let mut config = json!({
            "mcpServers": { "mold": { "command": "npx", "args": ["some-other-mold"] } }
        });
        let error = merge_connector(&mut config, &connector(), false)
            .unwrap_err()
            .to_string();
        assert!(error.contains("--force"), "{error}");
        assert_eq!(config["mcpServers"]["mold"]["command"], "npx");
        assert_eq!(remove_connector(&mut config).unwrap(), Removal::NotOurs);
        assert_eq!(connector_state(&config), "connector (not mold's)");
        assert_eq!(
            merge_connector(&mut config, &connector(), true).unwrap(),
            Merge::Updated
        );
        assert_eq!(connector_state(&config), "connector");
    }

    #[test]
    fn a_windows_mold_exe_counts_as_ours() {
        let mut config = json!({
            "mcpServers": { "mold": { "command": "C:\\Tools\\mold.exe", "args": ["mcp"] } }
        });
        assert_eq!(
            merge_connector(&mut config, &connector(), false).unwrap(),
            Merge::Updated
        );
    }

    #[test]
    fn remove_deletes_only_the_mold_connector() {
        let mut config = json!({ "mcpServers": { "time": {} } });
        assert_eq!(remove_connector(&mut config).unwrap(), Removal::Absent);
        merge_connector(&mut config, &connector(), false).unwrap();
        assert_eq!(remove_connector(&mut config).unwrap(), Removal::Removed);
        assert_eq!(config, json!({ "mcpServers": { "time": {} } }));
        assert_eq!(remove_connector(&mut json!({})).unwrap(), Removal::Absent);
    }

    #[test]
    fn a_config_that_is_not_an_object_is_refused() {
        assert!(merge_connector(&mut json!([]), &connector(), false).is_err());
        assert!(merge_connector(&mut json!({ "mcpServers": [] }), &connector(), false).is_err());
    }

    #[test]
    fn config_files_round_trip_with_a_backup_and_refuse_bad_json() {
        let dir = tempfile::tempdir().unwrap();
        let path = config_path(dir.path());
        assert_eq!(read_config(&path).unwrap(), json!({}));
        std::fs::write(&path, "  \n").unwrap();
        assert_eq!(read_config(&path).unwrap(), json!({}));

        std::fs::write(&path, r#"{"preferences":{"a":1}}"#).unwrap();
        let mut config = read_config(&path).unwrap();
        merge_connector(&mut config, &connector(), false).unwrap();
        let backup = write_config(&path, &config).unwrap().unwrap();
        assert_eq!(
            std::fs::read_to_string(backup).unwrap(),
            r#"{"preferences":{"a":1}}"#
        );
        assert_eq!(read_config(&path).unwrap(), config);

        std::fs::write(&path, "{ not json").unwrap();
        let error = read_config(&path).unwrap_err().to_string();
        assert!(error.contains("not valid JSON"), "{error}");
    }

    #[test]
    fn desktop_config_lives_where_each_platform_puts_it() {
        let home = Path::new("/Users/u");
        let appdata = Path::new("C:\\Users\\u\\AppData\\Roaming");
        let dir = config_dir(home, Some(appdata));
        if cfg!(target_os = "macos") {
            assert_eq!(dir, Some(home.join("Library/Application Support/Claude")));
        } else if cfg!(windows) {
            assert_eq!(dir, Some(appdata.join("Claude")));
        } else {
            assert_eq!(dir, None);
        }
    }
}
