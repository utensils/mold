//! `mold licenses` — third-party model licenses and their acceptance state.
//!
//! Acceptance is per Mold data root, so the only meaningful answer is "on
//! which machine?". This command asks the server the pull would go to and
//! names the root it read, because a user staring at "accepted" for the wrong
//! host is exactly the confusion this command exists to prevent.
//!
//! It reads THIS machine's root only when there is genuinely no server to ask
//! — a connection failure, or the explicit `--local` flag. An authentication
//! failure, a 5xx, or a malformed body all mean a server IS there and did not
//! answer the question; substituting local state then would report one
//! machine's acceptances under another machine's name, which is the same class
//! of bug as accepting on the wrong host in the first place.

use anyhow::Result;
use colored::Colorize;

use crate::control::CliContext;
use crate::output::status;
use crate::theme;
use crate::ui::col_width;
use crate::AlreadyReported;

/// Where a listing came from, for the header and for error wording.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LicenseSource {
    Server(String),
    ThisMachine,
}

impl LicenseSource {
    fn label(&self) -> String {
        match self {
            Self::Server(host) => host.clone(),
            Self::ThisMachine => "this machine".to_string(),
        }
    }
}

/// Decide what a failed `GET /api/licenses` means.
///
/// Split out from the I/O so every arm is unit-testable: the whole finding is
/// that "the request failed" is not one outcome but two, and only one of them
/// may be answered with local state.
pub fn resolve_listing_failure(error: &anyhow::Error, host: &str) -> LicenseListingFailure {
    match mold_core::classify_server_error(error) {
        mold_core::ServerAvailability::FallbackLocal => LicenseListingFailure::NoServer,
        mold_core::ServerAvailability::SurfaceError => {
            LicenseListingFailure::Unusable(format!("{host} did not report its licenses: {error}"))
        }
    }
}

/// The two meanings of a failed listing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LicenseListingFailure {
    /// Nothing is listening. This machine's root is the honest answer.
    NoServer,
    /// A server answered, but not with a listing — auth, 5xx, or a body that
    /// would not parse. Reported against the host; never papered over.
    Unusable(String),
}

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

/// `mold licenses accept <ID>...` — agree to pinned terms without downloading.
///
/// Consent and acquisition are different acts. Before this existed the only
/// way to accept was `mold pull <model> --accept-license <id>`, so agreeing to
/// terms always started a multi-gigabyte transfer — and a licence that no
/// installed manifest required could not be accepted at all.
///
/// Recorded on the machine that would run the pull, for the same reason `run`
/// reads from there: acceptance is per Mold data root, and recording one
/// machine's consent under another machine's name is the bug this module
/// exists to prevent.
pub async fn accept(ids: &[String], local: bool) -> Result<()> {
    let ctx = CliContext::new(None);
    let host = ctx.client().host().to_string();

    // Show the terms the RECORDING machine pins, not merely the ones this
    // build ships: consent must be given for the document the user read.
    let server_terms = if local {
        None
    } else {
        match ctx.client().list_licenses().await {
            Ok(licenses) => Some(licenses),
            Err(error) => match resolve_listing_failure(&error, &host) {
                LicenseListingFailure::NoServer => None,
                LicenseListingFailure::Unusable(message) => {
                    eprintln!("{} {message}", theme::prefix_error());
                    return Err(AlreadyReported.into());
                }
            },
        }
    };

    let mut payload = Vec::with_capacity(ids.len());
    for id in ids {
        let terms = server_terms
            .as_ref()
            .and_then(|rows| rows.iter().find(|row| &row.id == id))
            .map(|row| mold_core::types::LicenseAcceptance {
                id: row.id.clone(),
                url: row.url.clone(),
                sha256: row.sha256.clone(),
            })
            .or_else(|| {
                mold_core::license_acceptance::license_by_id(id)
                    .map(mold_core::license_acceptance::acceptance_for)
            });
        let Some(terms) = terms else {
            eprintln!(
                "{} unknown license '{id}'. Run {} to list them.",
                theme::prefix_error(),
                "mold licenses".bold()
            );
            return Err(AlreadyReported.into());
        };
        status!("{} {}", theme::icon_info(), terms.id.bold());
        status!("  terms: {}", terms.url);
        payload.push(terms);
    }

    let recorded_on = if local || server_terms.is_none() {
        if !local {
            crate::ui::print_server_fallback(&host, "recording on this machine");
        }
        let mold_home = mold_core::Config::mold_dir()
            .ok_or_else(|| anyhow::anyhow!("could not resolve the Mold data directory"))?;
        mold_core::license_acceptance::record_acceptances(&mold_home, &payload).map_err(
            |error| {
                eprintln!("{} {error}", theme::prefix_error());
                anyhow::Error::from(AlreadyReported)
            },
        )?;
        LicenseSource::ThisMachine
    } else {
        match ctx.client().accept_licenses(&payload).await {
            Ok(_) => LicenseSource::Server(host.clone()),
            Err(error) => {
                // An older host records consent only as a side effect of a
                // pull, so name that path rather than silently writing here —
                // a local write would record the wrong machine's agreement.
                eprintln!("{} {host} did not accept: {error}", theme::prefix_error());
                eprintln!(
                    "  If it is an older build, accept as part of the pull instead: {}",
                    format!(
                        "mold pull <model> --accept-license {}",
                        ids.join(" --accept-license ")
                    )
                    .bold()
                );
                return Err(AlreadyReported.into());
            }
        }
    };

    status!("");
    status!(
        "{} accepted on {}",
        theme::icon_ok(),
        recorded_on.label().bold()
    );
    Ok(())
}

pub async fn run(local: bool) -> Result<()> {
    let ctx = CliContext::new(None);
    let host = ctx.client().host().to_string();

    let (licenses, source) = if local {
        (local_statuses()?, LicenseSource::ThisMachine)
    } else {
        match ctx.client().list_licenses().await {
            Ok(licenses) => (licenses, LicenseSource::Server(host.clone())),
            Err(error) => match resolve_listing_failure(&error, &host) {
                LicenseListingFailure::NoServer => {
                    crate::ui::print_server_fallback(&host, "showing this machine's licenses");
                    (local_statuses()?, LicenseSource::ThisMachine)
                }
                LicenseListingFailure::Unusable(message) => {
                    eprintln!("{} {message}", theme::prefix_error());
                    eprintln!(
                        "  Run {} to read this machine's own acceptances instead.",
                        "mold licenses --local".bold()
                    );
                    return Err(AlreadyReported.into());
                }
            },
        }
    };
    let source = source.label();

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

#[cfg(test)]
mod tests {
    use super::*;
    use mold_core::error::MoldError;

    /// Nothing is listening: this machine's root is the honest answer.
    #[test]
    fn a_connection_failure_falls_back_to_this_machine() {
        let error = anyhow::Error::new(MoldError::Client("connect failed".into()));
        assert_eq!(
            resolve_listing_failure(&error, "http://box:7680"),
            LicenseListingFailure::NoServer
        );
    }

    /// A server IS there and refused or failed. Reporting local acceptances
    /// under its name would answer a different question than the one asked.
    #[test]
    fn auth_and_server_errors_are_reported_against_the_host() {
        for error in [
            anyhow::anyhow!("HTTP 401 Unauthorized"),
            anyhow::anyhow!("HTTP 500 Internal Server Error"),
            anyhow::anyhow!("error decoding response body"),
        ] {
            match resolve_listing_failure(&error, "http://box:7680") {
                LicenseListingFailure::Unusable(message) => {
                    assert!(
                        message.contains("http://box:7680"),
                        "the failure must name the host: {message}"
                    );
                }
                other => panic!("expected an unusable listing, got {other:?}"),
            }
        }
    }

    #[test]
    fn source_labels_name_the_root_that_was_read() {
        assert_eq!(
            LicenseSource::Server("http://box:7680".into()).label(),
            "http://box:7680"
        );
        assert_eq!(LicenseSource::ThisMachine.label(), "this machine");
    }
}
