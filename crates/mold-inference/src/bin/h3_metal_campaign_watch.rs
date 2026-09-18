//! Thin front end for the H3 Metal campaign watchdog.
//!
//! All supervision logic lives in `mold_inference::h3_metal_campaign_watch`
//! so it is unit-testable beside the in-process guard it mirrors. See that
//! module for the enforced gates and the fail-closed behavior.

use mold_inference::h3_metal_campaign_watch::{parse_args, run};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let parsed = parse_args(&args)?;
    let code = run(parsed)?;
    if code == 0 {
        Ok(())
    } else {
        std::process::exit(code);
    }
}
