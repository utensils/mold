//! `mold lambda` — Lambda Cloud deployment helpers.

use anyhow::{anyhow, bail, Result};
use mold_core::config::Config;
use mold_core::lambda::{
    build_launch_request, filesystem_name, gpu_uses_unsupported_linux_arm64, image_tag_for_gpu,
    render_cloud_init, AvailabilityRow, CloudInitOptions, CreateFilesystemRequest,
    CreateSshKeyRequest, Filesystem, Instance, LambdaClient, LambdaSettings, LaunchRequestInput,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::fs;
use std::io::{IsTerminal, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::time::{sleep, Duration};

use crate::theme;

const STATE_FILE: &str = "lambda-state.json";
const REMOTE_PORT: u16 = 7680;

#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq)]
pub struct LambdaState {
    #[serde(default)]
    pub instance_id: Option<String>,
    #[serde(default)]
    pub region: Option<String>,
    #[serde(default)]
    pub instance_type: Option<String>,
    #[serde(default)]
    pub public_ip: Option<String>,
    #[serde(default)]
    pub ssh_key_name: Option<String>,
    #[serde(default)]
    pub ssh_private_key_path: Option<String>,
    #[serde(default)]
    pub filesystem_name: Option<String>,
    #[serde(default)]
    pub tunnel_pid: Option<u32>,
    #[serde(default)]
    pub local_port: Option<u16>,
    #[serde(default)]
    pub remote_port: Option<u16>,
    #[serde(default)]
    pub updated_at: Option<u64>,
}

impl LambdaState {
    pub fn has_live_tunnel(&self, pid_exists: impl FnOnce(u32) -> bool) -> bool {
        self.tunnel_pid.is_some_and(pid_exists)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SshKeyDecision {
    UseExisting {
        public_key_path: PathBuf,
        private_key_path: PathBuf,
    },
    Generate {
        public_key_path: PathBuf,
        private_key_path: PathBuf,
    },
}

pub fn choose_ssh_key(
    home: &Path,
    public_exists: impl Fn(&Path) -> bool,
    private_exists: impl Fn(&Path) -> bool,
) -> SshKeyDecision {
    for name in ["id_ed25519", "id_rsa"] {
        let private_key_path = home.join(".ssh").join(name);
        let public_key_path = private_key_path.with_extension("pub");
        if public_exists(&public_key_path) && private_exists(&private_key_path) {
            return SshKeyDecision::UseExisting {
                public_key_path,
                private_key_path,
            };
        }
        if public_exists(&public_key_path) {
            return SshKeyDecision::UseExisting {
                public_key_path,
                private_key_path,
            };
        }
    }
    let private_key_path = home.join(".ssh/mold_lambda_ed25519");
    SshKeyDecision::Generate {
        public_key_path: private_key_path.with_extension("pub"),
        private_key_path,
    }
}

pub fn is_mold_managed_instance(instance: &Instance) -> bool {
    instance
        .tags
        .iter()
        .any(|t| t.key == "managed-by" && t.value == "mold")
        || instance
            .name
            .as_deref()
            .is_some_and(|n| n.starts_with("mold-"))
}

pub fn is_mold_managed_filesystem(fs: &Filesystem, prefix: &str) -> bool {
    fs.name == prefix || fs.name.starts_with(&format!("{prefix}-"))
}

#[derive(Debug, Clone, Serialize)]
pub struct DeployEvent {
    phase: String,
    status: String,
    #[serde(flatten)]
    details: Map<String, Value>,
}

impl DeployEvent {
    pub fn new(phase: impl Into<String>, status: impl Into<String>) -> Self {
        Self {
            phase: phase.into(),
            status: status.into(),
            details: Map::new(),
        }
    }

    pub fn with_detail(mut self, key: impl Into<String>, value: impl Serialize) -> Self {
        self.details.insert(
            key.into(),
            serde_json::to_value(value).unwrap_or(Value::Null),
        );
        self
    }
}

#[derive(Debug, Clone, Default)]
pub struct DeployOptions {
    pub instance_type: Option<String>,
    pub region: Option<String>,
    pub new: bool,
    pub dry_run: bool,
    pub json: bool,
    pub forward_secrets: bool,
    pub model: Option<String>,
    pub open_browser: bool,
}

fn state_path() -> Result<PathBuf> {
    let home = Config::mold_dir().ok_or_else(|| anyhow!("could not resolve MOLD_HOME"))?;
    fs::create_dir_all(&home)?;
    Ok(home.join(STATE_FILE))
}

fn load_state() -> LambdaState {
    let Ok(path) = state_path() else {
        return LambdaState::default();
    };
    fs::read_to_string(path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

fn save_state(state: &LambdaState) -> Result<()> {
    let path = state_path()?;
    fs::write(path, serde_json::to_vec_pretty(state)?)?;
    Ok(())
}

fn now_epoch() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn client_from_config(config: &Config) -> Result<LambdaClient> {
    LambdaClient::from_settings(&config.lambda)
}

pub async fn run_doctor() -> Result<()> {
    let config = Config::load_or_default();
    let key_source = if std::env::var(mold_core::lambda::API_KEY_ENV).is_ok() {
        "LAMBDA_API_KEY"
    } else if config.lambda.api_key.is_some() {
        "config lambda.api_key"
    } else {
        bail!(
            "missing Lambda API key; set LAMBDA_API_KEY or `mold config set lambda.api_key <key>`"
        );
    };
    let client = client_from_config(&config)?;
    let types = client.list_instance_types().await?;
    let keys = client.list_ssh_keys().await?;
    println!("{} Lambda auth ok ({key_source})", theme::icon_ok());
    println!(
        "{} endpoint: {}",
        theme::icon_info(),
        config.lambda.endpoint()
    );
    println!("{} instance types: {}", theme::icon_info(), types.len());
    println!("{} ssh keys: {}", theme::icon_info(), keys.len());
    Ok(())
}

pub async fn run_availability(json: bool) -> Result<()> {
    let config = Config::load_or_default();
    let client = client_from_config(&config)?;
    let version = env!("CARGO_PKG_VERSION");
    let mut rows = Vec::new();
    for ty in client.list_instance_types().await? {
        if ty.regions_with_capacity_available.is_empty() {
            continue;
        }
        for region in &ty.regions_with_capacity_available {
            let mut row =
                AvailabilityRow::from_instance_type(&ty, config.lambda.image_repository(), version);
            row.region = region.name.clone();
            rows.push(row);
        }
    }
    if json {
        println!("{}", serde_json::to_string_pretty(&rows)?);
        return Ok(());
    }
    println!(
        "{:<22} {:<12} {:<24} {:>4} {:>7} {:>8} {:<32}",
        "INSTANCE", "REGION", "GPU", "GPU", "SLOTS", "$/HR", "IMAGE"
    );
    for row in rows {
        println!(
            "{:<22} {:<12} {:<24} {:>4} {:>7} {:>8.2} {:<32}",
            row.instance_type,
            row.region,
            row.gpu_description,
            row.gpu_count,
            row.generation_slots,
            row.price_per_hour_usd,
            row.image
        );
    }
    Ok(())
}

pub async fn run_filesystems(json: bool) -> Result<()> {
    let config = Config::load_or_default();
    let filesystems = client_from_config(&config)?.list_filesystems().await?;
    if json {
        println!("{}", serde_json::to_string_pretty(&filesystems)?);
        return Ok(());
    }
    println!("{:<28} {:<14} {:>14}", "NAME", "REGION", "BYTES USED");
    for fs in filesystems {
        let region = fs.region.map(|r| r.name).unwrap_or_default();
        let bytes = fs
            .bytes_used
            .map(|b| b.to_string())
            .unwrap_or_else(|| "-".into());
        println!("{:<28} {:<14} {:>14}", fs.name, region, bytes);
    }
    println!(
        "{} Filesystems keep billing while they exist; check Lambda billing for current rates.",
        theme::icon_info()
    );
    Ok(())
}

pub async fn run_status(json: bool) -> Result<()> {
    let config = Config::load_or_default();
    let state = load_state();
    let tunnel_live = state.has_live_tunnel(pid_exists);
    if let Some(id) = &state.instance_id {
        let instance = client_from_config(&config)?.get_instance(id).await?;
        if json {
            let mut value = serde_json::to_value(&instance)?;
            if let Some(obj) = value.as_object_mut() {
                obj.insert("tunnel_live".into(), Value::Bool(tunnel_live));
            }
            println!("{}", serde_json::to_string_pretty(&value)?);
        } else {
            println!("{} {} {}", theme::icon_info(), instance.id, instance.status);
            if let Some(ip) = instance.ip {
                println!("ip: {ip}");
            }
            println!("tunnel: {}", if tunnel_live { "live" } else { "stopped" });
        }
    } else if json {
        println!("{}", serde_json::to_string_pretty(&state)?);
    } else {
        println!("{} no Lambda instance recorded", theme::icon_info());
    }
    Ok(())
}

pub async fn run_deploy(opts: DeployOptions) -> Result<()> {
    let config = Config::load_or_default();
    let client = client_from_config(&config)?;
    if !opts.new {
        let existing = load_state();
        if let (Some(id), Some(ip), Some(key), Some(port)) = (
            existing.instance_id.clone(),
            existing.public_ip.clone(),
            existing.ssh_private_key_path.clone(),
            existing.local_port,
        ) {
            let mut state = existing;
            if !state.has_live_tunnel(pid_exists) {
                if let Ok(pid) = spawn_tunnel(&ip, &PathBuf::from(key), port) {
                    state.tunnel_pid = Some(pid);
                    state.updated_at = Some(now_epoch());
                    save_state(&state)?;
                }
            }
            if opts.json {
                print_event(DeployEvent::new("reuse", "ok").with_detail("instance_id", id))?;
            } else {
                println!("{} reusing Lambda instance: {id}", theme::icon_ok());
                println!("export MOLD_HOST=http://127.0.0.1:{port}");
            }
            return Ok(());
        }
    }
    let instance_type = opts
        .instance_type
        .clone()
        .ok_or_else(|| anyhow!("--instance-type is required in non-interactive deploy"))?;
    let region = opts
        .region
        .clone()
        .ok_or_else(|| anyhow!("--region is required in non-interactive deploy"))?;
    if gpu_uses_unsupported_linux_arm64(&instance_type) {
        bail!(
            "{instance_type} is not supported by the published mold Docker images yet; GH200 Lambda hosts are linux/arm64, while the current images are linux/amd64. Choose an x86 GPU instance such as gpu_1x_a100_sxm4 or gpu_1x_a10."
        );
    }

    let ssh = if opts.dry_run {
        plan_ssh_key(&config.lambda)?
    } else {
        ensure_ssh_key(&client, &config.lambda).await?
    };
    let fs_name = filesystem_name(&config.lambda, &region);
    let filesystem_id = if opts.dry_run {
        None
    } else {
        Some(ensure_filesystem(&client, &fs_name, &region).await?.id)
    };

    let gpu_desc = instance_type.as_str();
    let image_tag = image_tag_for_gpu(gpu_desc, env!("CARGO_PKG_VERSION"));
    let image = format!("{}:{image_tag}", config.lambda.image_repository());
    let user_data = render_cloud_init(&CloudInitOptions {
        image,
        mount_path: config.lambda.filesystem_mount_path.clone(),
        env_file: "/etc/mold/lambda.env".into(),
    });
    let request = build_launch_request(LaunchRequestInput {
        region_name: &region,
        instance_type_name: &instance_type,
        ssh_key_name: &ssh.key_name,
        filesystem_name: &fs_name,
        filesystem_id: filesystem_id.as_deref(),
        filesystem_mount_path: &config.lambda.filesystem_mount_path,
        instance_name: &format!("mold-{region}"),
        image_id: None,
        user_data: &user_data,
    });

    if opts.dry_run {
        println!("{}", serde_json::to_string_pretty(&request)?);
        return Ok(());
    }
    if opts.json {
        print_event(DeployEvent::new("launch", "starting"))?;
    }
    let launch = client.launch_instance(&request).await?;
    let instance_id = launch
        .instance_ids
        .first()
        .cloned()
        .ok_or_else(|| anyhow!("Lambda launch response did not include an instance id"))?;
    let instance = wait_for_instance_ip(&client, &instance_id).await?;
    let local_port = choose_local_port(config.lambda.local_port)?;
    let mut state = LambdaState {
        instance_id: Some(instance.id.clone()),
        region: Some(region),
        instance_type: Some(instance_type),
        public_ip: instance.ip.clone(),
        ssh_key_name: Some(ssh.key_name),
        ssh_private_key_path: Some(ssh.private_key_path.display().to_string()),
        filesystem_name: Some(fs_name),
        local_port: Some(local_port),
        remote_port: Some(REMOTE_PORT),
        updated_at: Some(now_epoch()),
        ..Default::default()
    };
    if let Some(ip) = instance.ip {
        if opts.forward_secrets {
            forward_secrets(&ip, &ssh.private_key_path)?;
        }
        if let Ok(pid) = spawn_tunnel(&ip, &ssh.private_key_path, local_port) {
            state.tunnel_pid = Some(pid);
            if opts.json {
                print_event(
                    DeployEvent::new("tunnel", "ready").with_detail("local_port", local_port),
                )?;
            }
            if opts.open_browser {
                open_browser(&format!("http://127.0.0.1:{local_port}"));
            }
            if let Some(model) = opts.model {
                enqueue_download(local_port, &model).await.ok();
            }
        }
    }
    save_state(&state)?;
    if !opts.json {
        println!(
            "{} Lambda instance launched: {}",
            theme::icon_ok(),
            instance.id
        );
        println!("export MOLD_HOST=http://127.0.0.1:{local_port}");
    }
    Ok(())
}

struct SshMaterial {
    key_name: String,
    private_key_path: PathBuf,
}

async fn ensure_ssh_key(client: &LambdaClient, settings: &LambdaSettings) -> Result<SshMaterial> {
    let configured_name = settings.ssh_key_name.clone();
    if let Some(material) = configured_ssh_material(settings) {
        return Ok(material);
    }
    let home = dirs::home_dir().ok_or_else(|| anyhow!("cannot determine home directory"))?;
    let decision = choose_ssh_key(&home, Path::exists, Path::exists);
    let (public_key_path, private_key_path) = match decision {
        SshKeyDecision::UseExisting {
            public_key_path,
            private_key_path,
        } => (public_key_path, private_key_path),
        SshKeyDecision::Generate {
            public_key_path,
            private_key_path,
        } => {
            fs::create_dir_all(private_key_path.parent().unwrap_or_else(|| Path::new(".")))?;
            let status = Command::new("ssh-keygen")
                .args(["-t", "ed25519", "-N", "", "-f"])
                .arg(&private_key_path)
                .status()?;
            if !status.success() {
                bail!("ssh-keygen failed with {status}");
            }
            (public_key_path, private_key_path)
        }
    };
    let public_key = fs::read_to_string(&public_key_path)?;
    let key_name = configured_name.unwrap_or_else(default_ssh_key_name);
    let keys = client.list_ssh_keys().await?;
    if !keys.iter().any(|k| k.name == key_name) {
        client
            .create_ssh_key(&CreateSshKeyRequest {
                name: key_name.clone(),
                public_key,
            })
            .await?;
    }
    Ok(SshMaterial {
        key_name,
        private_key_path,
    })
}

fn plan_ssh_key(settings: &LambdaSettings) -> Result<SshMaterial> {
    if let Some(material) = configured_ssh_material(settings) {
        return Ok(material);
    }
    let home = dirs::home_dir().ok_or_else(|| anyhow!("cannot determine home directory"))?;
    let decision = choose_ssh_key(&home, Path::exists, Path::exists);
    let private_key_path = match decision {
        SshKeyDecision::UseExisting {
            private_key_path, ..
        }
        | SshKeyDecision::Generate {
            private_key_path, ..
        } => private_key_path,
    };
    Ok(SshMaterial {
        key_name: settings
            .ssh_key_name
            .clone()
            .unwrap_or_else(default_ssh_key_name),
        private_key_path,
    })
}

fn configured_ssh_material(settings: &LambdaSettings) -> Option<SshMaterial> {
    match (
        settings.ssh_key_name.clone(),
        settings.ssh_private_key_path.clone(),
    ) {
        (Some(key_name), Some(path)) => Some(SshMaterial {
            key_name,
            private_key_path: expand_tilde(&path),
        }),
        _ => None,
    }
}

fn default_ssh_key_name() -> String {
    format!(
        "mold-{}",
        hostname::get()
            .ok()
            .and_then(|s| s.into_string().ok())
            .unwrap_or_else(|| "local".into())
    )
}

async fn ensure_filesystem(client: &LambdaClient, name: &str, region: &str) -> Result<Filesystem> {
    let filesystems = client.list_filesystems().await?;
    if let Some(filesystem) = filesystems.into_iter().find(|fs| fs.name == name) {
        return Ok(filesystem);
    }
    let filesystem = client
        .create_filesystem(&CreateFilesystemRequest {
            name: name.to_string(),
            region: region.to_string(),
        })
        .await?;
    Ok(filesystem)
}

async fn wait_for_instance_ip(client: &LambdaClient, instance_id: &str) -> Result<Instance> {
    let mut last = None;
    for _ in 0..90 {
        let instance = client.get_instance(instance_id).await?;
        if instance.ip.is_some() {
            return Ok(instance);
        }
        last = Some(instance);
        sleep(Duration::from_secs(5)).await;
    }
    match last {
        Some(instance) => Err(anyhow!(
            "Lambda instance {} did not report an IP before timeout (last status: {})",
            instance.id,
            instance.status
        )),
        None => Err(anyhow!(
            "Lambda instance {instance_id} was not visible after launch"
        )),
    }
}

pub async fn run_tunnel(local_port: Option<u16>) -> Result<()> {
    let mut state = load_state();
    let ip = state
        .public_ip
        .clone()
        .ok_or_else(|| anyhow!("no Lambda instance IP recorded"))?;
    let key = state
        .ssh_private_key_path
        .clone()
        .ok_or_else(|| anyhow!("no Lambda SSH private key recorded"))?;
    let port = choose_local_port(local_port.or(state.local_port).unwrap_or(7680))?;
    let pid = spawn_tunnel(&ip, &PathBuf::from(key), port)?;
    state.tunnel_pid = Some(pid);
    state.local_port = Some(port);
    state.remote_port = Some(REMOTE_PORT);
    state.updated_at = Some(now_epoch());
    save_state(&state)?;
    println!("export MOLD_HOST=http://127.0.0.1:{port}");
    Ok(())
}

pub async fn run_ssh() -> Result<()> {
    let state = load_state();
    let ip = state
        .public_ip
        .ok_or_else(|| anyhow!("no Lambda instance IP recorded"))?;
    let key = state
        .ssh_private_key_path
        .ok_or_else(|| anyhow!("no Lambda SSH private key recorded"))?;
    let status = Command::new("ssh")
        .args(ssh_common_options())
        .arg("-i")
        .arg(key)
        .arg(format!("ubuntu@{ip}"))
        .status()?;
    if !status.success() {
        bail!("ssh exited with {status}");
    }
    Ok(())
}

pub async fn run_logs(follow: bool) -> Result<()> {
    let state = load_state();
    let ip = state
        .public_ip
        .ok_or_else(|| anyhow!("no Lambda instance IP recorded"))?;
    let key = state
        .ssh_private_key_path
        .ok_or_else(|| anyhow!("no Lambda SSH private key recorded"))?;
    let mut remote = "sudo journalctl -u mold-lambda.service -n 200 --no-pager".to_string();
    if follow {
        remote.push_str(" -f");
    }
    let status = Command::new("ssh")
        .args(ssh_common_options())
        .arg("-i")
        .arg(key)
        .arg(format!("ubuntu@{ip}"))
        .arg(remote)
        .status()?;
    if !status.success() {
        bail!("ssh logs exited with {status}");
    }
    Ok(())
}

pub async fn run_terminate(json: bool) -> Result<()> {
    let config = Config::load_or_default();
    let mut state = load_state();
    if let Some(pid) = state.tunnel_pid.take() {
        kill_pid(pid).ok();
    }
    if let Some(id) = state.instance_id.take() {
        client_from_config(&config)?.terminate_instance(&id).await?;
        if json {
            print_event(DeployEvent::new("terminate", "ok").with_detail("instance_id", id))?;
        } else {
            println!("{} terminated Lambda instance", theme::icon_ok());
        }
    }
    state.updated_at = Some(now_epoch());
    save_state(&state)?;
    Ok(())
}

pub async fn run_reset_to_zero(confirm: Option<String>, json: bool) -> Result<()> {
    let config = Config::load_or_default();
    let client = client_from_config(&config)?;
    if confirm.as_deref() != Some("delete mold lambda resources") && std::io::stdin().is_terminal()
    {
        eprint!("Type 'delete mold lambda resources' to continue: ");
        std::io::stderr().flush().ok();
        let mut line = String::new();
        std::io::stdin().read_line(&mut line)?;
        if line.trim() != "delete mold lambda resources" {
            bail!("confirmation did not match");
        }
    }
    let state = load_state();
    if let Some(pid) = state.tunnel_pid {
        kill_pid(pid).ok();
    }
    for inst in client.list_instances().await? {
        if is_mold_managed_instance(&inst) {
            client.terminate_instance(&inst.id).await?;
        }
    }
    let prefix = config
        .lambda
        .filesystem_prefix
        .as_deref()
        .unwrap_or("mold")
        .to_string();
    for fs in client.list_filesystems().await? {
        if is_mold_managed_filesystem(&fs, &prefix) {
            client.delete_filesystem(&fs.id).await?;
        }
    }
    save_state(&LambdaState::default())?;
    if json {
        print_event(DeployEvent::new("reset", "ok"))?;
    } else {
        println!("{} removed mold-managed Lambda resources", theme::icon_ok());
    }
    Ok(())
}

fn spawn_tunnel(ip: &str, key: &Path, local_port: u16) -> Result<u32> {
    let mut command = Command::new("ssh");
    command
        .args(ssh_common_options())
        .arg("-o")
        .arg("ExitOnForwardFailure=yes")
        .arg("-i")
        .arg(key)
        .args([
            "-N",
            "-L",
            &format!("127.0.0.1:{local_port}:127.0.0.1:{REMOTE_PORT}"),
            &format!("ubuntu@{ip}"),
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }
    let child = command.spawn()?;
    Ok(child.id())
}

fn choose_local_port(preferred: u16) -> Result<u16> {
    if TcpListener::bind(("127.0.0.1", preferred)).is_ok() {
        return Ok(preferred);
    }
    let listener = TcpListener::bind(("127.0.0.1", 0))?;
    Ok(listener.local_addr()?.port())
}

fn kill_pid(pid: u32) -> Result<()> {
    let status = Command::new("kill").arg(pid.to_string()).status()?;
    if !status.success() {
        bail!("kill {pid} exited with {status}");
    }
    Ok(())
}

fn pid_exists(pid: u32) -> bool {
    Command::new("kill")
        .args(["-0", &pid.to_string()])
        .status()
        .is_ok_and(|s| s.success())
}

fn expand_tilde(path: &str) -> PathBuf {
    if let Some(rest) = path.strip_prefix("~/") {
        if let Some(home) = dirs::home_dir() {
            return home.join(rest);
        }
    }
    PathBuf::from(path)
}

fn print_event(event: DeployEvent) -> Result<()> {
    println!("{}", serde_json::to_string(&event)?);
    Ok(())
}

fn open_browser(url: &str) {
    let program = if cfg!(target_os = "macos") {
        "open"
    } else if cfg!(target_os = "windows") {
        "cmd"
    } else {
        "xdg-open"
    };
    let mut cmd = Command::new(program);
    if cfg!(target_os = "windows") {
        cmd.args(["/C", "start", url]);
    } else {
        cmd.arg(url);
    }
    let _ = cmd.stdout(Stdio::null()).stderr(Stdio::null()).spawn();
}

fn forward_secrets(ip: &str, key: &Path) -> Result<()> {
    let mut env_lines = Vec::new();
    for name in ["HF_TOKEN", "CIVITAI_TOKEN"] {
        if let Ok(value) = std::env::var(name) {
            env_lines.push(format!("{name}={value}"));
        }
    }
    if env_lines.is_empty() {
        return Ok(());
    }
    let payload = env_lines.join("\n");
    let remote = "sudo install -m 600 /dev/stdin /etc/mold/lambda.env && sudo systemctl restart mold-lambda.service";
    let mut child = Command::new("ssh")
        .args(ssh_common_options())
        .arg("-i")
        .arg(key)
        .arg(format!("ubuntu@{ip}"))
        .arg(remote)
        .stdin(Stdio::piped())
        .spawn()?;
    if let Some(stdin) = child.stdin.as_mut() {
        stdin.write_all(payload.as_bytes())?;
    }
    let status = child.wait()?;
    if !status.success() {
        bail!("secret forwarding failed with {status}");
    }
    Ok(())
}

fn ssh_common_options() -> [&'static str; 4] {
    [
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ServerAliveInterval=30",
    ]
}

async fn enqueue_download(local_port: u16, model: &str) -> Result<()> {
    let url = format!("http://127.0.0.1:{local_port}/api/downloads");
    reqwest::Client::new()
        .post(url)
        .json(&serde_json::json!({ "model": model }))
        .send()
        .await?
        .error_for_status()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ssh_key_discovery_prefers_ed25519_then_rsa_then_generated() {
        let home = std::path::Path::new("/home/me");
        let ed = home.join(".ssh/id_ed25519.pub");
        let rsa = home.join(".ssh/id_rsa.pub");
        assert_eq!(
            choose_ssh_key(
                home,
                |p| p == ed.as_path() || p == rsa.as_path(),
                |_p| false
            ),
            SshKeyDecision::UseExisting {
                public_key_path: ed.clone(),
                private_key_path: home.join(".ssh/id_ed25519")
            }
        );
        assert_eq!(
            choose_ssh_key(home, |p| p == rsa.as_path(), |_p| false),
            SshKeyDecision::UseExisting {
                public_key_path: rsa,
                private_key_path: home.join(".ssh/id_rsa")
            }
        );
        assert!(matches!(
            choose_ssh_key(home, |_p| false, |_p| false),
            SshKeyDecision::Generate { .. }
        ));
    }

    #[test]
    fn lambda_state_serde_preserves_tunnel_and_instance() {
        let state = LambdaState {
            instance_id: Some("inst-1".into()),
            region: Some("us-west-1".into()),
            instance_type: Some("gpu_1x_a10".into()),
            public_ip: Some("203.0.113.10".into()),
            ssh_key_name: Some("mold-host".into()),
            ssh_private_key_path: Some("/home/me/.ssh/id_ed25519".into()),
            filesystem_name: Some("mold-us-west-1".into()),
            tunnel_pid: Some(1234),
            local_port: Some(7681),
            remote_port: Some(7680),
            updated_at: Some(1_700_000_000),
        };
        let json = serde_json::to_string(&state).unwrap();
        let decoded: LambdaState = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.instance_id.as_deref(), Some("inst-1"));
        assert_eq!(decoded.tunnel_pid, Some(1234));
        assert_eq!(decoded.local_port, Some(7681));
    }

    #[test]
    fn stale_tunnel_is_detected_from_pid_probe() {
        let state = LambdaState {
            tunnel_pid: Some(42),
            ..Default::default()
        };
        assert!(state.has_live_tunnel(|pid| pid == 42));
        assert!(!state.has_live_tunnel(|_| false));
    }

    #[test]
    fn reset_filter_only_targets_mold_managed_resources() {
        let owned = mold_core::lambda::Instance {
            id: "inst-owned".into(),
            name: Some("mold-us-west-1".into()),
            tags: vec![mold_core::lambda::Tag {
                key: "managed-by".into(),
                value: "mold".into(),
            }],
            ..Default::default()
        };
        let other = mold_core::lambda::Instance {
            id: "inst-other".into(),
            name: Some("not-mold".into()),
            ..Default::default()
        };
        assert!(is_mold_managed_instance(&owned));
        assert!(!is_mold_managed_instance(&other));

        let fs = mold_core::lambda::Filesystem {
            id: "fs-1".into(),
            name: "mold-us-west-1".into(),
            ..Default::default()
        };
        assert!(is_mold_managed_filesystem(&fs, "mold"));
        assert!(!is_mold_managed_filesystem(&fs, "other"));
    }

    #[test]
    fn deploy_event_json_shape_is_stable() {
        let event = DeployEvent::new("tunnel", "ready").with_detail("local_port", 7680);
        let json = serde_json::to_value(event).unwrap();
        assert_eq!(json["phase"], "tunnel");
        assert_eq!(json["status"], "ready");
        assert_eq!(json["local_port"], 7680);
    }
}
