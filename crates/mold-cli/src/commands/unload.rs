use anyhow::Result;
use mold_core::{classify_server_error, ServerAvailability};

use crate::control::CliContext;
use crate::theme;
use crate::ui::print_server_unavailable;

pub async fn run() -> Result<()> {
    let ctx = CliContext::new(None);

    match ctx.client().unload_model().await {
        Ok(body) => {
            println!("{} {}", theme::icon_ok(), body);
            Ok(())
        }
        Err(e) => match classify_server_error(&e) {
            ServerAvailability::FallbackLocal => {
                if crate::control::is_loopback_host(ctx.client().host()) {
                    println!("No mold server is running; no server-loaded model to unload.");
                    Ok(())
                } else {
                    print_server_unavailable(ctx.client().host(), &e);
                    Err(crate::AlreadyReported.into())
                }
            }
            ServerAvailability::SurfaceError => Err(e),
        },
    }
}
