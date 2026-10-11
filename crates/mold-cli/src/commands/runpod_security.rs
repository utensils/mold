use anyhow::{bail, Context, Result};
use mold_core::runpod::{CreatePodRequest, Pod, RunPodClient};
use mold_secure_client::{PairedHost, PreparedBootstrap};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

#[derive(Serialize, Deserialize)]
struct Pending {
    name: String,
    profile: PairedHost,
    volume: Option<String>,
}
fn root() -> Result<PathBuf> {
    Ok(super::runpod_state_dir()?.join("secure/runpod"))
}
fn volume_path(root: &Path, volume: &str) -> PathBuf {
    root.join(format!(
        "volume-{}.json",
        hex::encode(Sha256::digest(volume.as_bytes()))
    ))
}
fn write(path: &Path, value: &impl Serialize, replace: bool) -> Result<()> {
    let bytes = zeroize::Zeroizing::new(serde_json::to_vec(value)?);
    mold_secure_client::identity_file::write_private_file(path, &bytes, replace)?;
    Ok(())
}
pub(super) fn mark_fresh_volume(volume: &str) -> Result<()> {
    write(
        &volume_path(&root()?, volume),
        &serde_json::json!({"fresh":true}),
        false,
    )
}
pub(super) fn remember_volume(pod: &Pod) -> Result<()> {
    let Some(volume) = pod.attached_network_volume_id() else {
        return Ok(());
    };
    let origin = format!("https://{}-7680.proxy.runpod.net", pod.id);
    mold_secure_client::load_pairing(&origin)?;
    write(
        &volume_path(&root()?, volume),
        &serde_json::json!({"origin":origin}),
        true,
    )
}
pub(super) fn preflight(request: &CreatePodRequest) -> Result<()> {
    if let Some(volume) = &request.network_volume_id {
        let saved: serde_json::Value = serde_json::from_slice(&mold_secure_client::read_private(&volume_path(&root()?,volume)).context("Pair the existing network volume host before replacement; no paid resource was changed")?)?;
        if saved["fresh"] != true {
            let origin = saved["origin"].as_str().context(
                "network volume has an unresolved bootstrap; recover the pending pod first",
            )?;
            mold_secure_client::load_pairing(origin)?;
        }
    }
    Ok(())
}
fn prepare(
    root: &Path,
    request: &mut CreatePodRequest,
) -> Result<(Pending, Option<PreparedBootstrap>)> {
    let stored = match &request.network_volume_id {
        None => None,
        Some(volume) => {
            let path = volume_path(root, volume);
            let raw = mold_secure_client::read_private(&path).context("This existing network volume has no trusted Mold pairing. Connect and pair its current host before replacement; no paid pod was created")?;
            let saved: serde_json::Value = serde_json::from_slice(&raw)?;
            if saved["fresh"] == true {
                None
            } else {
                let origin = saved["origin"]
                    .as_str()
                    .context("invalid private volume pairing")?;
                Some(mold_secure_client::load_pairing(origin)?)
            }
        }
    };
    let bootstrap = if stored.is_none() {
        Some(mold_secure_client::prepare_bootstrap("RunPod CLI", "cli")?)
    } else {
        None
    };
    let id = bootstrap
        .as_ref()
        .map(|v| v.id().to_owned())
        .unwrap_or_else(|| uuid::Uuid::new_v4().simple().to_string());
    request.name = format!(
        "{}-{}",
        request.name.chars().take(40).collect::<String>(),
        id
    );
    if let Some(bootstrap) = &bootstrap {
        for (key, value) in bootstrap.environment() {
            request.env.insert(key, value.into());
        }
    } else {
        // Operator auth remains enabled, but existing host/device authority is untouched.
        request.env.insert(
            "MOLD_API_KEY".into(),
            format!(
                "mold_operator_{}{}",
                uuid::Uuid::new_v4().simple(),
                uuid::Uuid::new_v4().simple()
            )
            .into(),
        );
    }
    let profile = match stored {
        Some(profile) => profile,
        None => bootstrap
            .as_ref()
            .unwrap()
            .profile("https://pending.invalid")?,
    };
    Ok((
        Pending {
            name: request.name.clone(),
            profile,
            volume: request.network_volume_id.clone(),
        },
        bootstrap,
    ))
}
fn finalize(root: &Path, pending: &Pending, pod: &Pod) -> Result<()> {
    if pod.name.as_deref().is_some_and(|name| name != pending.name) {
        bail!("provider returned a different bootstrap pod")
    }
    let mut profile = pending.profile.clone();
    profile.origin = format!("https://{}-7680.proxy.runpod.net", pod.id);
    profile.websocket = format!("wss://{}-7680.proxy.runpod.net/_mold/secure", pod.id);
    mold_secure_client::save_pairing(&profile)?;
    if let Some(volume) = &pending.volume {
        write(
            &volume_path(root, volume),
            &serde_json::json!({"origin":profile.origin}),
            true,
        )?;
    }
    std::fs::remove_file(root.join("pending.json"))?;
    Ok(())
}
pub(super) async fn recover(client: &RunPodClient) -> Result<Option<Pod>> {
    let root = root()?;
    let path = root.join("pending.json");
    if path.try_exists()? {
        let pending: Pending = serde_json::from_slice(&mold_secure_client::read_private(&path)?)?;
        let mut matches = client
            .list_pods()
            .await?
            .into_iter()
            .filter(|p| p.name.as_deref() == Some(&pending.name));
        let pod = matches.next().context("Previous pod creation outcome is uncertain. Inspect the RunPod console; no second paid request was sent. Keep the private pending journal for recovery")?;
        if matches.next().is_some() {
            bail!(
                "multiple provider pods match the pending bootstrap; refusing automatic selection"
            )
        }
        finalize(&root, &pending, &pod)?;
        return Ok(Some(pod));
    }
    Ok(None)
}

/// Reconcile uncertain owner API responses before another paid create. The
/// journal holds only client authority and public host pins, never host secrets.
pub(super) async fn create(client: &RunPodClient, request: &mut CreatePodRequest) -> Result<Pod> {
    struct SecretRequest<'a>(&'a mut CreatePodRequest);
    impl Drop for SecretRequest<'_> {
        fn drop(&mut self) {
            use zeroize::Zeroize;
            for key in ["MOLD_API_KEY", "MOLD_SECURE_BOOTSTRAP"] {
                if let Some(serde_json::Value::String(mut secret)) = self.0.env.remove(key) {
                    secret.zeroize();
                }
            }
        }
    }
    let guarded = SecretRequest(request);
    let root = root()?;
    let path = root.join("pending.json");
    if let Some(pod) = recover(client).await? {
        return Ok(pod);
    }
    let (pending, _bootstrap) = prepare(&root, guarded.0)?;
    write(&path, &pending, false)?;
    // Consumed once, before provider mutation: an unknown outcome cannot reuse
    // a supposedly empty volume for a different host identity.
    if let Some(volume) = &pending.volume {
        write(
            &volume_path(&root, volume),
            &serde_json::json!({"pending":pending.name}),
            true,
        )?;
    }
    let result = client.create_pod(guarded.0).await;
    drop(guarded);
    let pod = result.context("Pod create did not return a confirmed result. Client recovery authority was retained; rerun to reconcile the exact provider pod, never retry the paid POST blindly")?;
    finalize(&root, &pending, &pod)?;
    Ok(pod)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn unknown_volume_refused_before_bootstrap_or_paid_mutation() {
        let temp = tempfile::tempdir().unwrap();
        let mut request = CreatePodRequest {
            name: "test".into(),
            network_volume_id: Some("unknown".into()),
            ..Default::default()
        };
        assert!(prepare(temp.path(), &mut request).is_err());
        assert!(!request.env.contains_key("MOLD_SECURE_BOOTSTRAP"));
    }
    #[test]
    fn fresh_container_journal_contains_client_authority_but_no_host_secret() {
        let temp = tempfile::tempdir().unwrap();
        let mut request = CreatePodRequest {
            name: "test".into(),
            ..Default::default()
        };
        let (pending, bootstrap) = prepare(temp.path(), &mut request).unwrap();
        assert!(request.env.contains_key("MOLD_SECURE_BOOTSTRAP"));
        assert!(request.name.ends_with(bootstrap.unwrap().id()));
        let json = serde_json::to_value(pending).unwrap();
        assert!(json["profile"]["device_identity"].is_string());
        assert!(json.to_string().find("host_identity").is_none());
    }
    struct HomeGuard(Option<std::ffi::OsString>);
    impl Drop for HomeGuard {
        fn drop(&mut self) {
            match self.0.take() {
                Some(value) => std::env::set_var("MOLD_HOME", value),
                None => std::env::remove_var("MOLD_HOME"),
            }
        }
    }
    #[cfg(unix)]
    #[allow(clippy::await_holding_lock)]
    #[tokio::test]
    async fn private_journal_failure_clears_bootstrap_before_returning() {
        use std::os::unix::fs::PermissionsExt;
        let _lock = crate::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let home = tempfile::tempdir().unwrap();
        let _guard = HomeGuard(std::env::var_os("MOLD_HOME"));
        std::env::set_var("MOLD_HOME", home.path());
        let root = root().unwrap();
        std::fs::create_dir_all(&root).unwrap();
        std::fs::set_permissions(&root, std::fs::Permissions::from_mode(0o755)).unwrap();
        let mut request = CreatePodRequest {
            name: "test".into(),
            ..Default::default()
        };
        let client = RunPodClient::new("http://127.0.0.1:1", "test");
        assert!(create(&client, &mut request).await.is_err());
        assert!(!request.env.contains_key("MOLD_SECURE_BOOTSTRAP"));
        assert!(!request.env.contains_key("MOLD_API_KEY"));
    }
    #[test]
    fn known_volume_reuses_existing_authority_without_importing_identity() {
        let _lock = crate::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let home = tempfile::tempdir().unwrap();
        let _guard = HomeGuard(std::env::var_os("MOLD_HOME"));
        std::env::set_var("MOLD_HOME", home.path());
        let original = mold_secure_client::prepare_bootstrap("test", "cli")
            .unwrap()
            .profile("https://old-7680.proxy.runpod.net")
            .unwrap();
        mold_secure_client::save_pairing(&original).unwrap();
        let root = root().unwrap();
        write(
            &volume_path(&root, "known"),
            &serde_json::json!({"origin":original.origin}),
            false,
        )
        .unwrap();
        let mut request = CreatePodRequest {
            name: "replacement".into(),
            network_volume_id: Some("known".into()),
            ..Default::default()
        };
        let (pending, bootstrap) = prepare(&root, &mut request).unwrap();
        assert!(bootstrap.is_none());
        assert!(!request.env.contains_key("MOLD_SECURE_BOOTSTRAP"));
        assert_eq!(pending.profile.host_key, original.host_key);
        assert_eq!(pending.profile.device_identity, original.device_identity);
        assert_eq!(pending.profile.api_key, original.api_key);
        assert_eq!(pending.profile.grant.signature, original.grant.signature);
    }
    #[allow(clippy::await_holding_lock)]
    #[tokio::test]
    async fn uncertain_create_is_recovered_without_replaying_paid_post() {
        use wiremock::{
            matchers::{method, path},
            Mock, MockServer, ResponseTemplate,
        };
        let _lock = crate::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let home = tempfile::tempdir().unwrap();
        let _guard = HomeGuard(std::env::var_os("MOLD_HOME"));
        std::env::set_var("MOLD_HOME", home.path());
        let server = MockServer::start().await;
        let journal = home.path().join("secure/runpod/pending.json");
        let checked = journal.clone();
        Mock::given(method("POST"))
            .and(path("/pods"))
            .respond_with(move |req: &wiremock::Request| {
                assert!(
                    checked.exists(),
                    "authority must be durable before paid create"
                );
                let value: serde_json::Value = serde_json::from_slice(&req.body).unwrap();
                assert!(value["env"]["MOLD_SECURE_BOOTSTRAP"].is_string());
                ResponseTemplate::new(500).set_body_string("unknown outcome")
            })
            .expect(1)
            .mount(&server)
            .await;
        let client = RunPodClient::new(server.uri(), "test");
        let mut request = CreatePodRequest {
            name: "mold-test".into(),
            ..Default::default()
        };
        assert!(create(&client, &mut request).await.is_err());
        let pending: Pending =
            serde_json::from_slice(&mold_secure_client::read_private(&journal).unwrap()).unwrap();
        Mock::given(method("GET")).and(path("/pods")).respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!([{"id":"recovered","name":pending.name,"desiredStatus":"RUNNING"}]))).expect(1).mount(&server).await;
        let recovered = create(&client, &mut request).await.unwrap();
        assert_eq!(recovered.id, "recovered");
        assert!(!journal.exists());
        let profile =
            mold_secure_client::load_pairing("https://recovered-7680.proxy.runpod.net").unwrap();
        assert_eq!(profile.host_key, pending.profile.host_key);
        assert_eq!(profile.device_identity, pending.profile.device_identity);
        assert!(!request.env.contains_key("MOLD_SECURE_BOOTSTRAP"));
    }
}
