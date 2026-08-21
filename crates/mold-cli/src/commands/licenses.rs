//! `mold licenses` — third-party model licenses and their acceptance state.
//!
//! Acceptance is per Mold data root, so the only meaningful answer is "on
//! which machine?". This command asks the server the pull would go to, and
//! falls back to this machine's own root only when no server answers —
//! reporting which one it read, because a user staring at "accepted" for the
//! wrong host is exactly the confusion this command exists to prevent.

use anyhow::Result;
use colored::Colorize;

use crate::control::{is_loopback_host, CliContext};
use crate::output::status;
use crate::theme;
use crate::ui::col_width;
use crate::AlreadyReported;

fn local_statuses() -> Result<Vec<mold_core::ThirdPartyLicenseStatus>> {
    let Some(mold_home) = mold_core::Config::mold_dir() else {
        eprintln!(
            "{} could not resolve the Mold data directory",
            theme::prefix_error()
        );
        return Err(AlreadyReported.into());
    };
    Ok(mold_core::license_acceptance::license_statuses(&mold_home))
}

pub async fn run() -> Result<()> {
    let ctx = CliContext::new(None);
    let host = ctx.client().host().to_string();

    let (licenses, source) = match ctx.client().list_licenses().await {
        Ok(licenses) => (licenses, host.clone()),
        Err(_) if is_loopback_host(&host) => (local_statuses()?, "this machine".to_string()),
        Err(_) => {
            // A named remote that did not answer must not be papered over
            // with local state — that would report the wrong machine's
            // acceptances under the remote's name.
            crate::ui::print_server_fallback(&host, "showing this machine's licenses");
            (local_statuses()?, "this machine".to_string())
        }
    };

    if licenses.is_empty() {
        status!(
            "{} no third-party model licenses require acceptance on {}",
            theme::icon_info(),
            source.bold()
        );
        return Ok(());
    }

    status!("{} Licenses on {}", theme::icon_info(), source.bold());
    status!("");

    let nw = col_width(licenses.iter().map(|license| license.id.len()), 8, 2);
    for license in &licenses {
        let state = if license.accepted {
            "accepted".green().to_string()
        } else {
            "required".yellow().to_string()
        };
        println!("  {:<nw$} {:<10} {}", license.id, state, license.name);
        println!("  {:<nw$} {}", "", license.summary.dimmed(), nw = nw);
        if !license.required_by.is_empty() {
            println!(
                "  {:<nw$} {}",
                "",
                format!("needed by: {}", license.required_by.join(", ")).dimmed(),
                nw = nw
            );
        }
        println!("  {:<nw$} {}", "", license.url.dimmed(), nw = nw);
        if !license.accepted {
            println!(
                "  {:<nw$} {}",
                "",
                format!(
                    "accept with: mold pull {} --accept-license {}",
                    license
                        .required_by
                        .first()
                        .map(String::as_str)
                        .unwrap_or("<model>"),
                    license.id
                )
                .dimmed(),
                nw = nw
            );
        }
        println!();
    }

    Ok(())
}
