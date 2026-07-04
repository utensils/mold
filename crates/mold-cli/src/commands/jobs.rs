#![allow(dead_code, reason = "Phase 3 placeholder — removed in Phase 6")]

use anyhow::Result;
use mold_core::{Config, MoldClient};

use crate::{JobsAction, RetakeModeArg};

pub async fn run(_action: JobsAction, _config: &Config) -> Result<()> {
    todo!("TODO(BR55 Phase 6): dispatch mold jobs subcommands to typed MoldClient methods")
}

struct RetakeArgs {
    stage: u32,
    mode: RetakeModeArg,
    seed_offset: Option<u64>,
    prompt: Option<String>,
}

async fn jobs_list(_client: &MoldClient, _json: bool) -> Result<()> {
    todo!("TODO(BR55 Phase 6): implement mold jobs list output")
}

async fn jobs_show(_client: &MoldClient, _id: &str, _json: bool) -> Result<()> {
    todo!("TODO(BR55 Phase 6): implement mold jobs show output")
}

async fn jobs_resume(_client: &MoldClient, _id: &str) -> Result<()> {
    todo!("TODO(BR55 Phase 6): implement mold jobs resume command")
}

async fn jobs_retake(_client: &MoldClient, _id: &str, _args: RetakeArgs) -> Result<()> {
    todo!("TODO(BR55 Phase 6): implement mold jobs retake command")
}

async fn jobs_cancel(_client: &MoldClient, _id: &str) -> Result<()> {
    todo!("TODO(BR55 Phase 6): implement mold jobs cancel command")
}

async fn jobs_delete(_client: &MoldClient, _id: &str, _yes: bool) -> Result<()> {
    todo!("TODO(BR55 Phase 6): implement mold jobs delete command with confirmation unless --yes")
}

async fn jobs_gc(_client: &MoldClient) -> Result<()> {
    todo!("TODO(BR55 Phase 6): implement mold jobs gc command")
}
