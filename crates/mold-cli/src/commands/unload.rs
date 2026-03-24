use anyhow::Result;
use colored::Colorize;
use mold_core::MoldClient;

use crate::ui::print_server_unavailable;

pub async fn run() -> Result<()> {
    let client = crate::control::client_for_host(None);

    match client.unload_model().await {
        Ok(body) => {
            println!("{} {}", "●".green(), body);
            Ok(())
        }
        Err(e) => {
            if MoldClient::is_connection_error(&e) {
                print_server_unavailable(client.host(), &e);
                Err(crate::AlreadyReported.into())
            } else {
                Err(e)
            }
        }
    }
}
