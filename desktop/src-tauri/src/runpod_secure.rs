//! Durable recovery for an owner-authorized paid create. Never journal the host seed.
use super::{CreatePodRequest, Pod, RunPodLaunchClient};
use crate::secrets::SecretStore;
use mold_secure_client::{Binding, BootstrapProfile};
use serde::{Deserialize, Serialize};

#[derive(Serialize)]
pub struct RunPodLaunch {
    pub pod: Pod,
    pub profile: serde_json::Value,
}
#[derive(Clone, Serialize, Deserialize)]
struct Journal {
    id: String,
    name: String,
    profile: BootstrapProfile,
    #[serde(default)]
    pod_id: Option<String>,
    #[serde(default)]
    volume_id: Option<String>,
}
fn origin(id: &str) -> Result<String, String> {
    if id.is_empty()
        || id.len() > 128
        || !id.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-')
    {
        return Err("Invalid RunPod instance identity.".into());
    }
    Ok(format!("https://{id}-7680.proxy.runpod.net"))
}
fn public_profile(
    profile: &BootstrapProfile,
    id: &str,
    device: &Binding,
) -> Result<serde_json::Value, String> {
    profile
        .validate()
        .map_err(|_| "This saved connection needs a fresh pairing from the host.".to_string())?;
    if device
        .bytes()
        .map_err(|_| "Invalid device identity.".to_string())?
        != profile
            .device_binding
            .bytes()
            .map_err(|_| "Invalid saved device identity.".to_string())?
    {
        return Err(
            "This instance was paired with another installation. Pair it again from the host."
                .into(),
        );
    }
    let profile = profile
        .bind_origin(&origin(id)?)
        .map_err(|_| "Invalid secure instance address.".to_string())?;
    Ok(serde_json::json!({
        "namespace": profile.namespace, "host_key": profile.host_key,
        "host_noise_key": profile.host_noise_key, "host_binding_signature": profile.host_binding_signature,
        "websocket_url": null, "instance_id": profile.instance_id, "grant": profile.grant,
        "origins": [profile.origin], "endpoints": [{"url":profile.origin, "kind":"lan"}],
        "apiKey": profile.api_key,
    }))
}
fn journal(secrets: &SecretStore, value: &Journal) -> Result<(), String> {
    let encoded = serde_json::to_string(value)
        .map_err(|_| "Could not prepare secure recovery state.".to_string())?;
    secrets
        .set(&format!("runpod-bootstrap.{}", value.id), &encoded)
        .map_err(|_| "Could not save secure recovery state. No instance was requested.".to_string())
}
fn publish(
    secrets: &SecretStore,
    value: &Journal,
    id: &str,
    device: &Binding,
) -> Result<serde_json::Value, String> {
    let profile = public_profile(&value.profile, id, device)?;
    secrets.set(&format!("runpod-pairing.{id}"), &value.id)
        .map_err(|_| "The instance was created. Use in Mold to recover its secure connection; do not rent another instance.".to_string())?;
    let mut completed = value.clone();
    completed.pod_id = Some(id.into());
    journal(secrets, &completed)?;
    if let Some(volume) = &value.volume_id {
        save_volume(secrets, volume, &value.profile.namespace, &origin(id)?)?;
    }
    Ok(profile)
}
pub(super) fn pending_names(secrets: &SecretStore) -> Vec<String> {
    secrets
        .runpod_bootstraps()
        .into_iter()
        .filter_map(|raw| serde_json::from_str::<Journal>(&raw).ok())
        .filter(|j| j.pod_id.is_none())
        .map(|j| j.name)
        .collect()
}
pub(super) fn clear_unconfirmed(secrets: &SecretStore, owned: &[Pod]) -> Result<(), String> {
    let pending: Vec<Journal> = secrets
        .runpod_bootstraps()
        .into_iter()
        .map(|raw| {
            serde_json::from_str(&raw).map_err(|_| "Unconfirmed request is unreadable.".to_string())
        })
        .collect::<Result<_, _>>()?;
    if pending
        .iter()
        .any(|j| j.pod_id.is_none() && owned.iter().any(|p| p.name.as_deref() == Some(&j.name)))
    {
        return Err("RunPod lists an instance for this request. Choose Use in Mold to recover it; the request was not cleared.".into());
    }
    for value in pending.into_iter().filter(|j| j.pod_id.is_none()) {
        secrets
            .clear(&format!("runpod-bootstrap.{}", value.id))
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}
pub(super) fn volume_origin(secrets: &SecretStore, id: &str) -> Result<Option<String>, String> {
    let Some(raw) = secrets
        .get(&format!("runpod-volume.{id}"))
        .map_err(|e| e.to_string())?
    else {
        return Ok(None);
    };
    let profile: serde_json::Value =
        serde_json::from_str(&raw).map_err(|_| "Saved volume pairing is unreadable".to_string())?;
    Ok(profile["origin"].as_str().map(str::to_string))
}
pub(super) fn current_profile(
    value: serde_json::Value,
    device: &Binding,
    origin: &str,
) -> Result<BootstrapProfile, String> {
    let profile: BootstrapProfile = serde_json::from_value(serde_json::json!({
        "protocol":3,"instance_id":value["instance_id"],"origin":"","websocket":"","endpoints":[],
        "namespace":value["namespace"],"host_key":value["host_key"],"host_noise_key":value["host_noise_key"],
        "host_binding_signature":value["host_binding_signature"],"device_binding":device,"api_key":value["apiKey"],"grant":value["grant"]
    })).map_err(|_| "Current secure pairing is incomplete.".to_string())?;
    profile
        .validate()
        .map_err(|_| "Current secure pairing is invalid or expired.".to_string())?;
    profile.bind_origin(origin).map_err(|e| e.to_string())
}
fn save_volume(
    secrets: &SecretStore,
    id: &str,
    namespace: &str,
    origin: &str,
) -> Result<(), String> {
    secrets
        .set(
            &format!("runpod-volume.{id}"),
            &serde_json::json!({"namespace":namespace,"origin":origin}).to_string(),
        )
        .map_err(|e| e.to_string())
}
pub(super) fn associate(
    secrets: &SecretStore,
    pod: &Pod,
    profile: BootstrapProfile,
) -> Result<(), String> {
    if let Some(id) = &pod.network_volume_id {
        save_volume(secrets, id, &profile.namespace, &profile.origin)?;
    }
    Ok(())
}
#[cfg(test)]
pub(super) async fn launch<C: RunPodLaunchClient>(
    client: &C,
    secrets: &SecretStore,
    request: CreatePodRequest,
    device: &Binding,
) -> Result<RunPodLaunch, String> {
    launch_with_profile(client, secrets, request, device, None).await
}
pub(super) async fn launch_with_profile<C: RunPodLaunchClient>(
    client: &C,
    secrets: &SecretStore,
    mut request: CreatePodRequest,
    device: &Binding,
    resume: Option<serde_json::Value>,
) -> Result<RunPodLaunch, String> {
    for raw in secrets.runpod_bootstraps() {
        let value: Journal = serde_json::from_str(&raw)
            .map_err(|_| "Unconfirmed provisioning state is unreadable".to_string())?;
        if value.pod_id.is_none() {
            return Err("A previous GPU request is unconfirmed. Refresh the instance list and choose Use in Mold before renting again.".into());
        }
    }
    if secrets.runpod_bootstraps().len() >= 128 {
        return Err("Remove old RunPod connections before renting another instance.".into());
    }
    let resume = if let Some(volume) = &request.network_volume_id {
        if let Some(raw) = secrets
            .get(&format!("runpod-volume.{volume}"))
            .map_err(|e| e.to_string())?
        {
            let saved: serde_json::Value = serde_json::from_str(&raw).map_err(|e| e.to_string())?;
            let current = current_profile(resume.ok_or("Pair or reconnect this volume's current Mold host before renting replacement compute.")?, device, saved["origin"].as_str().ok_or("Invalid volume association")?)?;
            if Some(current.namespace.as_str()) != saved["namespace"].as_str() {
                return Err("The volume's host identity does not match this pairing.".into());
            }
            Some(current)
        } else {
            if secrets
                .get(&format!("runpod-fresh-volume.{volume}"))
                .map_err(|e| e.to_string())?
                .is_none()
            {
                return Err("This existing volume must be paired through its current Mold host before renting compute. No instance was requested.".into());
            }
            None
        }
    } else {
        None
    };
    let prepared = if resume.is_none() {
        Some(
            mold_secure_client::prepare_bootstrap_for_binding("Mold Studio", "desktop", device)
                .map_err(|_| "Could not prepare an encrypted instance connection.".to_string())?,
        )
    } else {
        None
    };
    let request_id = prepared
        .as_ref()
        .map(|p| p.id().to_string())
        .unwrap_or_else(|| uuid::Uuid::new_v4().simple().to_string());
    // The full random ID lets authenticated ListPods reconcile a lost create reply.
    let label: String = request
        .name
        .chars()
        .filter(|c| !c.is_control())
        .take(24)
        .collect();
    request.name = format!("{label}-{request_id}");
    let value = Journal {
        id: request_id,
        name: request.name.clone(),
        profile: resume.unwrap_or_else(|| prepared.as_ref().unwrap().public_profile()),
        pod_id: None,
        volume_id: request.network_volume_id.clone(),
    };
    journal(secrets, &value)?;
    if let Some(volume) = &request.network_volume_id {
        secrets
            .clear(&format!("runpod-fresh-volume.{volume}"))
            .map_err(|e| e.to_string())?;
    }
    if let Some(prepared) = &prepared {
        for (name, value) in prepared.environment() {
            request.env.insert(name, value.into());
        }
    } else {
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
    struct SecretRequest<'a>(&'a mut CreatePodRequest);
    impl Drop for SecretRequest<'_> {
        fn drop(&mut self) {
            use zeroize::Zeroize;
            for name in ["MOLD_API_KEY", "MOLD_SECURE_BOOTSTRAP"] {
                if let Some(serde_json::Value::String(mut secret)) = self.0.env.remove(name) {
                    secret.zeroize();
                }
            }
        }
    }
    let guarded = SecretRequest(&mut request);
    let result = client.create_pod(guarded.0).await;
    drop(guarded);
    let pod = result.map_err(|_| format!(
        "RunPod did not confirm the request for {}. Check the RunPod instance list before renting again; Use in Mold recovers its secure connection if creation succeeded.", value.name))?;
    let profile = publish(secrets, &value, &pod.id, device)?;
    Ok(RunPodLaunch { pod, profile })
}
pub(super) fn saved(
    secrets: &SecretStore,
    id: &str,
    device: &Binding,
) -> Result<Option<serde_json::Value>, String> {
    origin(id)?;
    let Some(key) = secrets
        .get(&format!("runpod-pairing.{id}"))
        .map_err(|_| "Could not read saved RunPod connection.".to_string())?
    else {
        return Ok(None);
    };
    let Some(raw) = secrets
        .get(&format!("runpod-bootstrap.{key}"))
        .map_err(|_| "Could not read saved RunPod connection.".to_string())?
    else {
        return Ok(None);
    };
    let value: Journal = serde_json::from_str(&raw)
        .map_err(|_| "Saved RunPod pairing is unreadable. Pair the host again.".to_string())?;
    public_profile(&value.profile, id, device).map(Some)
}
pub(super) fn recover(
    secrets: &SecretStore,
    id: &str,
    device: &Binding,
    owned: &[Pod],
) -> Result<Option<serde_json::Value>, String> {
    origin(id)?;
    let Some(pod) = owned.iter().find(|p| p.id == id) else {
        return Ok(None);
    };
    for raw in secrets.runpod_bootstraps() {
        let Ok(value) = serde_json::from_str::<Journal>(&raw) else {
            continue;
        };
        if pod.name.as_deref() == Some(&value.name) {
            return publish(secrets, &value, id, device).map(Some);
        }
    }
    Ok(None)
}
pub(super) fn forget(secrets: &SecretStore, id: &str) -> Result<(), String> {
    origin(id)?;
    let name = format!("runpod-pairing.{id}");
    let key = secrets
        .get(&name)
        .map_err(|_| "Could not read saved RunPod connection.".to_string())?;
    if let Some(key) = key {
        secrets
            .clear(&format!("runpod-bootstrap.{key}"))
            .map_err(|_| {
                "Instance deleted, but its saved secure connection could not be cleared."
                    .to_string()
            })?;
    }
    secrets.clear(&name).map_err(|_| {
        "Instance deleted, but its saved secure connection could not be cleared.".to_string()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    struct Client {
        calls: AtomicUsize,
        fail: bool,
    }
    impl RunPodLaunchClient for Client {
        async fn gpu_types(&self) -> anyhow::Result<Vec<super::super::GpuType>> {
            unreachable!()
        }
        async fn supported_pod_gpu_type_ids(
            &self,
        ) -> anyhow::Result<std::collections::HashSet<String>> {
            unreachable!()
        }
        async fn create_pod(&self, request: &CreatePodRequest) -> anyhow::Result<Pod> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            assert!(request.env.contains_key("MOLD_API_KEY"));
            let bootstrap: serde_json::Value =
                serde_json::from_str(request.env["MOLD_SECURE_BOOTSTRAP"].as_str().unwrap())
                    .unwrap();
            assert_eq!(bootstrap["client"]["kind"], "desktop");
            assert!(request.name.ends_with(bootstrap["id"].as_str().unwrap()));
            if self.fail {
                anyhow::bail!("lost provider reply")
            }
            Ok(
                serde_json::from_value(serde_json::json!({"id":"pod123", "name":request.name}))
                    .unwrap(),
            )
        }
    }
    fn device() -> Binding {
        // The common fixture holds public binding/signature only; no production identity.
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../relay/aws/test/fixtures/admission-v3.json"
        ))
        .unwrap();
        serde_json::from_value(serde_json::json!({"signingKey":fixture["binding"]["value"]["signingKey"],"noiseKey":fixture["binding"]["value"]["noiseKey"],"signature":fixture["binding"]["signature"]})).unwrap()
    }
    fn request() -> CreatePodRequest {
        super::super::build_request(super::super::tests::input(), None, "fixture-image".into())
            .unwrap()
    }
    #[tokio::test]
    async fn launch_saves_only_recoverable_client_authority_before_returning() {
        let dir = tempfile::tempdir().unwrap();
        let secrets = SecretStore::new(dir.path().into());
        let client = Client {
            calls: AtomicUsize::new(0),
            fail: false,
        };
        let result = launch(&client, &secrets, request(), &device())
            .await
            .unwrap();
        assert_eq!(result.pod.id, "pod123");
        assert_eq!(client.calls.load(Ordering::SeqCst), 1);
        assert_eq!(result.profile["websocket_url"], serde_json::Value::Null);
        assert_eq!(
            result.profile["origins"][0],
            "https://pod123-7680.proxy.runpod.net"
        );
        assert_eq!(
            saved(&secrets, "pod123", &device()).unwrap(),
            Some(result.profile)
        );
        let bytes = std::fs::read_to_string(dir.path().join("secrets.json")).unwrap();
        assert!(!bytes.contains("host_identity"));
        assert!(!bytes.contains("device_identity"));
        forget(&secrets, "pod123").unwrap();
        assert!(secrets.runpod_bootstraps().is_empty());
    }
    #[tokio::test]
    async fn a_lost_paid_reply_recovers_from_owned_pod_name_without_retrying_create() {
        let dir = tempfile::tempdir().unwrap();
        let secrets = SecretStore::new(dir.path().into());
        let client = Client {
            calls: AtomicUsize::new(0),
            fail: true,
        };
        assert!(launch(&client, &secrets, request(), &device())
            .await
            .is_err());
        let records = secrets.runpod_bootstraps();
        assert_eq!(records.len(), 1);
        let record: Journal = serde_json::from_str(&records[0]).unwrap();
        let pod: Pod =
            serde_json::from_value(serde_json::json!({"id":"recovered", "name":record.name}))
                .unwrap();
        assert!(recover(&secrets, "recovered", &device(), &[])
            .unwrap()
            .is_none());
        assert!(recover(&secrets, "recovered", &device(), &[pod])
            .unwrap()
            .is_some());
        assert_eq!(client.calls.load(Ordering::SeqCst), 1);
    }
    #[tokio::test]
    async fn an_unresolved_paid_create_blocks_another_charge() {
        let dir = tempfile::tempdir().unwrap();
        let secrets = SecretStore::new(dir.path().into());
        let client = Client {
            calls: AtomicUsize::new(0),
            fail: true,
        };
        assert!(launch(&client, &secrets, request(), &device())
            .await
            .is_err());
        assert!(launch(&client, &secrets, request(), &device())
            .await
            .is_err());
        assert_eq!(client.calls.load(Ordering::SeqCst), 1);
    }
    #[tokio::test]
    async fn an_unknown_existing_network_volume_is_rejected_before_charging() {
        let dir = tempfile::tempdir().unwrap();
        let secrets = SecretStore::new(dir.path().into());
        let client = Client {
            calls: AtomicUsize::new(0),
            fail: false,
        };
        let mut input = request();
        input.network_volume_id = Some("unknown".into());
        assert!(launch(&client, &secrets, input, &device()).await.is_err());
        assert_eq!(client.calls.load(Ordering::SeqCst), 0);
    }
    #[tokio::test]
    async fn a_fresh_volume_bootstraps_once_and_replacement_reuses_current_pairing() {
        let dir = tempfile::tempdir().unwrap();
        let secrets = SecretStore::new(dir.path().into());
        secrets.set("runpod-fresh-volume.volume1", "fresh").unwrap();
        let client = Client {
            calls: AtomicUsize::new(0),
            fail: false,
        };
        let mut input = request();
        input.network_volume_id = Some("volume1".into());
        let first = launch(&client, &secrets, input.clone(), &device())
            .await
            .unwrap();
        assert!(secrets
            .get("runpod-fresh-volume.volume1")
            .unwrap()
            .is_none());
        assert_eq!(
            volume_origin(&secrets, "volume1").unwrap().as_deref(),
            Some("https://pod123-7680.proxy.runpod.net")
        );
        let association = secrets.get("runpod-volume.volume1").unwrap().unwrap();
        assert!(!association.contains("api_key") && !association.contains("grant"));
        forget(&secrets, "pod123").unwrap();
        struct ResumeClient;
        impl RunPodLaunchClient for ResumeClient {
            async fn gpu_types(&self) -> anyhow::Result<Vec<super::super::GpuType>> {
                unreachable!()
            }
            async fn supported_pod_gpu_type_ids(
                &self,
            ) -> anyhow::Result<std::collections::HashSet<String>> {
                unreachable!()
            }
            async fn create_pod(&self, request: &CreatePodRequest) -> anyhow::Result<Pod> {
                assert!(!request.env.contains_key("MOLD_SECURE_BOOTSTRAP"));
                assert!(request.env.contains_key("MOLD_API_KEY"));
                Ok(serde_json::from_value(
                    serde_json::json!({"id":"replacement","name":request.name}),
                )
                .unwrap())
            }
        }
        let replacement = launch_with_profile(
            &ResumeClient,
            &secrets,
            input,
            &device(),
            Some(first.profile.clone()),
        )
        .await
        .unwrap();
        assert_eq!(replacement.profile["host_key"], first.profile["host_key"]);
        assert_eq!(replacement.profile["grant"], first.profile["grant"]);
        assert_eq!(
            replacement.profile["origins"][0],
            "https://replacement-7680.proxy.runpod.net"
        );
    }
    #[tokio::test]
    async fn explicit_clear_requires_a_fresh_owned_list_without_the_pending_pod() {
        let dir = tempfile::tempdir().unwrap();
        let secrets = SecretStore::new(dir.path().into());
        let client = Client {
            calls: AtomicUsize::new(0),
            fail: true,
        };
        assert!(launch(&client, &secrets, request(), &device())
            .await
            .is_err());
        let name = pending_names(&secrets).pop().unwrap();
        let pod = serde_json::from_value(serde_json::json!({"id":"late-pod","name":name})).unwrap();
        assert!(clear_unconfirmed(&secrets, &[pod]).is_err());
        assert_eq!(pending_names(&secrets).len(), 1);
        clear_unconfirmed(&secrets, &[]).unwrap();
        assert!(pending_names(&secrets).is_empty());
        assert_eq!(client.calls.load(Ordering::SeqCst), 1);
    }
    #[tokio::test]
    async fn failing_the_recovery_write_makes_zero_paid_requests() {
        let dir = tempfile::tempdir().unwrap();
        let bad = dir.path().join("file");
        std::fs::write(&bad, "not a directory").unwrap();
        let client = Client {
            calls: AtomicUsize::new(0),
            fail: false,
        };
        assert!(
            launch(&client, &SecretStore::new(bad), request(), &device())
                .await
                .is_err()
        );
        assert_eq!(client.calls.load(Ordering::SeqCst), 0);
    }
    #[test]
    fn provider_ids_and_device_bindings_are_not_trusted_as_urls_or_authority() {
        for id in ["", "../secret", "user@host", "id?token=secret", "id/path"] {
            assert!(origin(id).is_err());
        }
        let prepared =
            mold_secure_client::prepare_bootstrap_for_binding("Mold Studio", "desktop", &device())
                .unwrap();
        let mut wrong = device();
        wrong.signature = "0".repeat(128);
        assert!(public_profile(&prepared.public_profile(), "pod123", &wrong).is_err());
    }
}
