//! `mold host` — the saved hosts `--host` and `MOLD_HOST` accept by name.
//!
//! The registry is `mold_secure_client::hosts`, which reads the credential
//! profiles pairing already writes; nothing here probes the network.
use anyhow::Result;
use clap_complete::engine::CompletionCandidate;
use mold_secure_client::hosts::{self, SavedHost};

use crate::theme;
use crate::HostAction;

pub fn run(action: HostAction) -> Result<()> {
    // Running a host command is the one time derived names are written
    // down, so they stop depending on what else is saved.
    hosts::persist_derived_names()?;
    match action {
        HostAction::List { json } => list(json),
        HostAction::Use { name, clear } => {
            if clear {
                let cleared = hosts::clear_default()?;
                println!(
                    "{} {}; commands target {} unless --host or MOLD_HOST names a host.",
                    theme::icon_ok(),
                    if cleared {
                        "Default host cleared"
                    } else {
                        "No default host was set"
                    },
                    hosts::LOCAL_HOST
                );
                return Ok(());
            }
            let name = name.expect("clap requires NAME unless --clear");
            let host = hosts::set_default(&name)?;
            println!(
                "{} Default host is now '{}' ({}).",
                theme::icon_ok(),
                host.name,
                host.origin
            );
            if std::env::var("MOLD_HOST").is_ok_and(|v| !v.trim().is_empty()) {
                eprintln!(
                    "{} MOLD_HOST is set in this shell and still takes precedence over the default.",
                    theme::icon_warn()
                );
            }
            Ok(())
        }
        HostAction::Rm { name } => {
            let host = hosts::remove(&name)?;
            println!(
                "{} Removed '{}' ({}) and its saved credentials. Revoke this device on the host to end its grant there too.",
                theme::icon_ok(),
                host.name,
                host.origin
            );
            Ok(())
        }
    }
}

fn list(json: bool) -> Result<()> {
    let saved = hosts::list()?;
    let default = hosts::default()?.map(|h| h.origin);
    if json {
        let rows: Vec<serde_json::Value> = saved
            .iter()
            .map(|h| row_json(h, default.as_deref() == Some(h.origin.as_str())))
            .collect();
        println!("{}", serde_json::to_string_pretty(&rows)?);
        return Ok(());
    }
    if saved.is_empty() {
        println!(
            "No saved hosts. Pair one with `mold pair --invite-file <file>` (an invitation from `mold pairing create` on the host)."
        );
        return Ok(());
    }
    print!("{}", render_table(&saved, default.as_deref()));
    Ok(())
}

fn row_json(host: &SavedHost, is_default: bool) -> serde_json::Value {
    serde_json::json!({
        "name": host.name,
        "mode": host.mode.as_str(),
        "origin": host.origin,
        "default": is_default,
        "named": host.named,
        "ambiguous": host.ambiguous,
        "instance_id": host.instance_id,
        "hostname": host.hostname,
        "endpoints": host.endpoints,
    })
}

fn render_table(saved: &[SavedHost], default: Option<&str>) -> String {
    let label = |h: &SavedHost| {
        if h.ambiguous {
            format!("{} (ambiguous)", h.name)
        } else {
            h.name.clone()
        }
    };
    let name_width = saved
        .iter()
        .map(|h| label(h).len())
        .max()
        .unwrap_or(0)
        .max(4);
    let origin_width = saved
        .iter()
        .map(|h| h.origin.len())
        .max()
        .unwrap_or(0)
        .max(6);
    let mut out = format!(
        "{:<name_width$}  {:<7}  {:<origin_width$}  DEFAULT\n",
        "NAME", "MODE", "ORIGIN"
    );
    for host in saved {
        let marker = if default == Some(host.origin.as_str()) {
            "*"
        } else {
            ""
        };
        out.push_str(
            format!(
                "{:<name_width$}  {:<7}  {:<origin_width$}  {marker}",
                label(host),
                host.mode.as_str(),
                host.origin
            )
            .trim_end(),
        );
        out.push('\n');
    }
    if saved.iter().any(|h| h.ambiguous) {
        out.push_str(
            "\nAn ambiguous name belongs to more than one host and resolves to an error; re-pair one of them so it gets its own name.\n",
        );
    }
    out
}

/// Saved host names, for `mold host use|rm <TAB>`. Never fails.
pub fn complete_saved_name() -> Vec<CompletionCandidate> {
    hosts::list()
        .unwrap_or_default()
        .into_iter()
        .map(|h| CompletionCandidate::new(h.name))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use mold_secure_client::ConnectionMode;

    fn host(name: &str, origin: &str, mode: ConnectionMode) -> SavedHost {
        SavedHost {
            name: name.into(),
            origin: origin.into(),
            mode,
            instance_id: None,
            hostname: None,
            endpoints: vec![],
            named: true,
            ambiguous: false,
        }
    }

    #[test]
    fn the_table_marks_the_default_and_aligns_columns() {
        let saved = [
            host(
                "workstation",
                "https://abc.mold-link.urandom.io",
                ConnectionMode::Secure,
            ),
            host("lab", "http://10.0.0.8:7680", ConnectionMode::Regular),
        ];
        let table = render_table(&saved, Some("http://10.0.0.8:7680"));
        let lines: Vec<&str> = table.lines().collect();
        assert_eq!(
            lines,
            [
                "NAME         MODE     ORIGIN                            DEFAULT",
                "workstation  secure   https://abc.mold-link.urandom.io",
                "lab          regular  http://10.0.0.8:7680              *",
            ]
        );
    }

    #[test]
    fn ambiguous_rows_are_marked_and_explained() {
        let mut first = host("lab", "http://10.0.0.8:7680", ConnectionMode::Regular);
        first.ambiguous = true;
        first.named = false;
        let mut second = first.clone();
        second.origin = "http://10.0.0.9:7680".into();
        let table = render_table(&[first, second], None);
        assert_eq!(table.matches("lab (ambiguous)").count(), 2, "{table}");
        assert!(table.contains("resolves to an error"), "{table}");
    }

    #[test]
    fn json_rows_carry_the_default_flag() {
        let row = row_json(&host("lab", "http://x:7680", ConnectionMode::Regular), true);
        assert_eq!(row["name"], "lab");
        assert_eq!(row["mode"], "regular");
        assert_eq!(row["default"], true);
    }
}
