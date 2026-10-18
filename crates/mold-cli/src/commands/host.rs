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
        "instance_id": host.instance_id,
        "hostname": host.hostname,
        "endpoints": host.endpoints,
    })
}

fn render_table(saved: &[SavedHost], default: Option<&str>) -> String {
    let name_width = saved.iter().map(|h| h.name.len()).max().unwrap_or(0).max(4);
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
                host.name,
                host.mode.as_str(),
                host.origin
            )
            .trim_end(),
        );
        out.push('\n');
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
    fn json_rows_carry_the_default_flag() {
        let row = row_json(&host("lab", "http://x:7680", ConnectionMode::Regular), true);
        assert_eq!(row["name"], "lab");
        assert_eq!(row["mode"], "regular");
        assert_eq!(row["default"], true);
    }
}
