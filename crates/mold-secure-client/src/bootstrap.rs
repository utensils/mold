//! Owner-controlled initial provisioning. The private environment belongs only
//! in the authenticated provider create request; it is never a discovery reply.
use crate::{now, Endpoint, PairedHost};
use anyhow::{bail, Result};
use mold_secure::{decode_hex, grant_bytes, namespace, Binding, DeviceGrant, Identity};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use zeroize::{Zeroize, Zeroizing};

/// Recovery metadata for a client already authorized by the bootstrap. This
/// includes its API credential, so persist it only in the platform secret store.
/// Neither endpoint private key is present in this public-key profile.
#[derive(Clone, Serialize, Deserialize)]
pub struct BootstrapProfile {
    pub protocol: u8,
    pub instance_id: String,
    pub origin: String,
    pub websocket: String,
    pub endpoints: Vec<Endpoint>,
    pub namespace: String,
    pub host_key: String,
    pub host_noise_key: String,
    pub host_binding_signature: String,
    pub device_binding: Binding,
    pub api_key: String,
    pub grant: DeviceGrant,
}
impl Drop for BootstrapProfile {
    fn drop(&mut self) {
        self.api_key.zeroize();
    }
}
impl BootstrapProfile {
    /// Attach a provider-confirmed origin without accepting identity from it.
    pub fn bind_origin(&self, origin: &str) -> Result<Self> {
        let mut url = reqwest::Url::parse(origin)?;
        let numeric = url.host_str().is_some_and(|host| {
            host == "localhost"
                || host
                    .trim_matches(['[', ']'])
                    .parse::<std::net::IpAddr>()
                    .is_ok()
        });
        if url.origin().ascii_serialization() != origin
            || !url.username().is_empty()
            || url.password().is_some()
            || url.query().is_some()
            || url.fragment().is_some()
            || (url.scheme() != "https" && !(url.scheme() == "http" && numeric))
        {
            bail!("secure bootstrap requires a canonical HTTPS or numeric local origin")
        }
        let scheme = if url.scheme() == "https" { "wss" } else { "ws" };
        url.set_scheme(scheme)
            .map_err(|_| anyhow::anyhow!("invalid secure origin"))?;
        url.set_path("/_mold/secure");
        let mut result = self.clone();
        result.origin = origin.to_string();
        result.websocket = url.to_string();
        Ok(result)
    }
    pub fn validate(&self) -> Result<()> {
        if self.protocol != 3 || uuid::Uuid::parse_str(&self.instance_id).is_err() {
            bail!("invalid bootstrap profile version or instance identity")
        }
        let host = decode_hex::<32>(&self.host_key)?;
        if namespace(&host) != self.namespace {
            bail!("invalid bootstrap namespace")
        }
        Binding {
            signing_key: self.host_key.clone(),
            noise_key: self.host_noise_key.clone(),
            signature: self.host_binding_signature.clone(),
        }
        .bytes()?;
        self.grant
            .verify(&host, &self.device_binding, now(), self.grant.epoch)?;
        if !self.origin.is_empty() {
            let rebound = self.bind_origin(&self.origin)?;
            if rebound.websocket != self.websocket {
                bail!("bootstrap endpoint mismatch")
            }
        }
        Ok(())
    }
}

pub struct PreparedBootstrap {
    id: String,
    profile: BootstrapProfile,
    environment: BTreeMap<String, String>,
    device_identity: Option<Zeroizing<String>>,
}
impl Drop for PreparedBootstrap {
    fn drop(&mut self) {
        for value in self.environment.values_mut() {
            value.zeroize();
        }
    }
}
impl PreparedBootstrap {
    pub fn id(&self) -> &str {
        &self.id
    }
    pub fn public_profile(&self) -> BootstrapProfile {
        self.profile.clone()
    }
    /// Secrets: send only through the owner-authenticated provider API. Never
    /// log this map or include it in a command line or a client-facing response.
    pub fn environment(&self) -> BTreeMap<String, String> {
        self.environment.clone()
    }
    pub fn profile(&self, origin: &str) -> Result<PairedHost> {
        let identity = self.device_identity.as_ref().ok_or_else(|| {
            anyhow::anyhow!("this bootstrap uses the existing installation identity")
        })?;
        let p = self.profile.bind_origin(origin)?;
        let host = PairedHost {
            protocol: p.protocol,
            origin: p.origin.clone(),
            websocket: p.websocket.clone(),
            endpoints: p.endpoints.clone(),
            namespace: p.namespace.clone(),
            host_key: p.host_key.clone(),
            host_noise_key: p.host_noise_key.clone(),
            host_binding_signature: p.host_binding_signature.clone(),
            device_identity: identity.to_string(),
            api_key: p.api_key.clone(),
            grant: p.grant.clone(),
        };
        host.validate()?;
        Ok(host)
    }
}

pub fn prepare_bootstrap(name: &str, kind: &str) -> Result<PreparedBootstrap> {
    let device = Identity::generate()?;
    let mut prepared = prepare_bootstrap_for_binding(name, kind, &device.public_binding())?;
    let bytes = Zeroizing::new(device.to_bytes());
    prepared.device_identity = Some(Zeroizing::new(hex::encode(&*bytes)));
    Ok(prepared)
}

pub fn prepare_bootstrap_for_binding(
    name: &str,
    kind: &str,
    device: &Binding,
) -> Result<PreparedBootstrap> {
    device.bytes()?;
    if name.is_empty()
        || name.len() > 128
        || name.chars().any(char::is_control)
        || !matches!(kind, "cli" | "desktop" | "ios" | "android" | "browser")
    {
        bail!("invalid bootstrap client name or kind")
    }
    let host = Identity::generate()?;
    let binding = host.public_binding();
    let namespace = namespace(&host.signing_public());
    let id = uuid::Uuid::new_v4().simple().to_string();
    let instance_id = uuid::Uuid::new_v4().to_string();
    let grant_id = uuid::Uuid::new_v4().simple().to_string();
    let expires_at = now() + 30 * 24 * 3600;
    let grant = DeviceGrant {
        id: grant_id.clone(),
        expires_at,
        epoch: 0,
        signature: hex::encode(host.sign(&grant_bytes(
            &namespace,
            &decode_hex::<32>(&device.signing_key)?,
            &decode_hex::<32>(&device.noise_key)?,
            &grant_id,
            expires_at,
            0,
        )?)),
    };
    let api_key = format!(
        "mold_pair_{}{}",
        uuid::Uuid::new_v4().simple(),
        uuid::Uuid::new_v4().simple()
    );
    let host_bytes = Zeroizing::new(host.to_bytes());
    #[derive(Serialize)]
    struct Client<'a> {
        binding: &'a Binding,
        grant: &'a DeviceGrant,
        api_key: &'a str,
        name: &'a str,
        kind: &'a str,
    }
    #[derive(Serialize)]
    struct Bootstrap<'a> {
        version: u8,
        id: &'a str,
        instance_id: &'a str,
        host_identity: &'a str,
        client: Client<'a>,
    }
    let host_secret = Zeroizing::new(hex::encode(&*host_bytes));
    let serialized = serde_json::to_string(&Bootstrap {
        version: 1,
        id: &id,
        instance_id: &instance_id,
        host_identity: &host_secret,
        client: Client {
            binding: device,
            grant: &grant,
            api_key: &api_key,
            name,
            kind,
        },
    })?;
    // The serialized environment is zeroized by PreparedBootstrap::drop.
    let environment = BTreeMap::from([
        ("MOLD_SECURE_BOOTSTRAP".to_string(), serialized),
        (
            "MOLD_API_KEY".to_string(),
            format!(
                "mold_operator_{}{}",
                uuid::Uuid::new_v4().simple(),
                uuid::Uuid::new_v4().simple()
            ),
        ),
    ]);
    Ok(PreparedBootstrap {
        id,
        profile: BootstrapProfile {
            protocol: 3,
            instance_id,
            origin: String::new(),
            websocket: String::new(),
            endpoints: vec![],
            namespace,
            host_key: binding.signing_key,
            host_noise_key: binding.noise_key,
            host_binding_signature: binding.signature,
            device_binding: device.clone(),
            api_key,
            grant,
        },
        environment,
        device_identity: None,
    })
}
