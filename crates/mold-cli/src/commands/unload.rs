use anyhow::Result;
use colored::Colorize;
use mold_core::{classify_server_error, ServerAvailability};

use crate::control::CliContext;
use crate::ui::print_server_unavailable;

pub async fn run() -> Result<()> {
    let ctx = CliContext::new(None);

    match ctx.client().unload_model().await {
        Ok(body) => {
            println!("{} {}", "●".green(), body);
            Ok(())
        }
        Err(e) => match classify_server_error(&e) {
            ServerAvailability::FallbackLocal => {
                print_server_unavailable(ctx.client().host(), &e);
                Err(crate::AlreadyReported.into())
            }
            ServerAvailability::SurfaceError => Err(e),
        },
    }
}
