//! Family-neutral durable admission-authority adapter.
//!
//! Generic queue code persists and restores only opaque envelopes plus a
//! typed disposition. Family policy remains behind this module.

use crate::durable_disposition::DurableDisposition;

/// Durable outcome for failures discovered only after acknowledgement.
pub(crate) fn preparation_disposition(error: &crate::routes::ApiError) -> DurableDisposition {
    match error.code.as_str() {
        "SERVER_RESTARTING" => DurableDisposition::Retain,
        // These name dependencies the operator/user can install or make
        // reachable without changing the accepted request.
        "UNKNOWN_MODEL" | "MODEL_NOT_FOUND" => DurableDisposition::Hold { retryable: true },
        _ if error.status() == axum::http::StatusCode::TOO_MANY_REQUESTS
            || error.status().is_server_error() =>
        {
            DurableDisposition::Hold { retryable: true }
        }
        _ => DurableDisposition::Hold { retryable: false },
    }
}

pub(crate) struct Failure {
    pub disposition: DurableDisposition,
    pub message: String,
}

pub(crate) struct CapturedAuthority {
    pub envelope: Vec<u8>,
    pub replaces: crate::queue_media::ProcessPrivateAuthority,
}

#[cfg(any(feature = "h3", feature = "h3-private-uat"))]
const ENVELOPE_VERSION: u16 = 1;
#[cfg(any(feature = "h3", feature = "h3-private-uat"))]
const H3_PRIVATE_KIND: &str = "minimax_h3_private";

#[cfg(any(feature = "h3", feature = "h3-private-uat"))]
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct AuthorityEnvelope {
    version: u16,
    kind: String,
    payload: Vec<u8>,
}

#[derive(Clone, Default)]
pub(crate) enum RuntimeAuthority {
    #[default]
    None,
    #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
    H3(crate::h3_private_bridge::H3PrivateIngressGrant),
}

impl RuntimeAuthority {
    #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
    pub(crate) fn h3_grant(&self) -> Option<&crate::h3_private_bridge::H3PrivateIngressGrant> {
        match self {
            Self::H3(grant) => Some(grant),
            Self::None => None,
        }
    }
}

/// Whether this request is admitted through a private ingress grant rather
/// than public model activation. Pure and cheap: it is the ONE predicate
/// admission branches on before the grant exists, and the one `restore`
/// demands an envelope for, so the two can never disagree about which rows
/// carry authority.
pub(crate) fn claims_private_ingress(request: &mold_core::GenerateRequest) -> bool {
    #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
    {
        crate::h3_private_bridge::claims_private_ingress(request)
    }
    #[cfg(not(any(feature = "h3", feature = "h3-private-uat")))]
    {
        let _ = request;
        false
    }
}

/// The identity-only subject a private ingress binds client-operation
/// idempotency to. It is what `capture` will record on the grant, asked
/// BEFORE any child is resolved so a duplicate operation can be recognised
/// without spending anything.
pub(crate) fn idempotency_subject_sha256(
    request: &mold_core::GenerateRequest,
    authenticated: Option<&crate::auth::ApiKeyAuthenticated>,
    instance_id: &str,
) -> Result<Option<String>, crate::routes::ApiError> {
    #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
    {
        crate::h3_private_bridge::idempotency_subject_sha256(request, authenticated, instance_id)
    }
    #[cfg(not(any(feature = "h3", feature = "h3-private-uat")))]
    {
        let _ = (request, authenticated, instance_id);
        Ok(None)
    }
}

pub(crate) fn capture(
    request: &mold_core::GenerateRequest,
    authenticated: Option<&crate::auth::ApiKeyAuthenticated>,
    instance_id: &str,
) -> Result<Option<CapturedAuthority>, crate::routes::ApiError> {
    #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
    if let Some(grant) = crate::h3_private_bridge::capture_durable_h3_private_ingress(
        request,
        authenticated,
        instance_id,
    )? {
        let payload = grant
            .durable_replay_envelope()
            .map_err(crate::routes::ApiError::internal)?;
        return Ok(Some(CapturedAuthority {
            envelope: serde_json::to_vec(&AuthorityEnvelope {
                version: ENVELOPE_VERSION,
                kind: H3_PRIVATE_KIND.to_string(),
                payload,
            })
            .map_err(|error| crate::routes::ApiError::internal(error.to_string()))?,
            replaces: crate::queue_media::ProcessPrivateAuthority::H3PrivateIngressGrant,
        }));
    }
    let _ = (request, authenticated, instance_id);
    Ok(None)
}

pub(crate) fn restore(
    request: &mold_core::GenerateRequest,
    envelope: Option<&[u8]>,
    instance_id: &str,
) -> Result<RuntimeAuthority, Failure> {
    #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
    {
        // Must ask the SAME question `capture` asked. `capture` routes through
        // `classify_h3_private_ingress`, which returns `Ok(None)` for a pinned
        // unrunnable identity and therefore binds no envelope; asking the
        // broader `capability_contract_for_model` here made restore demand an
        // envelope capture never wrote, so a download-only H3 row parked as a
        // permanent hold where its own `/api/models` entry promised
        // `MINIMAX_H3_RUNTIME_UNAVAILABLE` / 501.
        let requires_h3 = claims_private_ingress(request);
        let Some(envelope) = envelope else {
            return if requires_h3 {
                Err(Failure {
                    disposition: DurableDisposition::Hold { retryable: false },
                    message: "durable MiniMax H3 admission authority is missing".into(),
                })
            } else {
                Ok(RuntimeAuthority::default())
            };
        };
        let envelope: AuthorityEnvelope =
            serde_json::from_slice(envelope).map_err(|_| Failure {
                disposition: DurableDisposition::Hold { retryable: false },
                message: "durable admission authority envelope is invalid".into(),
            })?;
        if envelope.version != ENVELOPE_VERSION || envelope.kind != H3_PRIVATE_KIND || !requires_h3
        {
            return Err(Failure {
                disposition: DurableDisposition::Hold { retryable: false },
                message: "durable admission authority is attached to an unsupported request".into(),
            });
        }
        crate::h3_private_bridge::restore_durable_h3_private_ingress(
            request,
            &envelope.payload,
            instance_id,
        )
        .and_then(|h3| {
            h3.map(RuntimeAuthority::H3).ok_or_else(|| {
                crate::routes::ApiError::with_code(
                    "durable admission authority did not restore a runtime grant",
                    crate::h3_private_bridge::H3_PRIVATE_PARTITION_REJECTED,
                    axum::http::StatusCode::UNPROCESSABLE_ENTITY,
                )
            })
        })
        .map_err(|error| Failure {
            disposition: DurableDisposition::Hold {
                retryable: error.code == crate::h3_private_bridge::H3_PRIVATE_RUNTIME_UNAVAILABLE,
            },
            message: error.error,
        })
    }
    #[cfg(not(any(feature = "h3", feature = "h3-private-uat")))]
    {
        let _ = (request, instance_id);
        if envelope.is_some() {
            Err(Failure {
                disposition: DurableDisposition::Hold { retryable: false },
                message: "durable admission authority is unsupported by this build".into(),
            })
        } else {
            Ok(RuntimeAuthority::default())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The grant is captured on the DESCRIPTOR request — the form every
    /// consumer re-hashes after hydration — so restore accepts exactly that
    /// request and refuses one whose descriptors were reordered.
    #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
    #[test]
    fn h3_authority_is_captured_on_the_descriptor_request_and_restores_against_it() {
        let descriptor = |name: &str, byte: u8| {
            serde_json::json!({
                "kind": "image",
                "media": { "authority": "descriptor" },
                "provenance": { "name": name, "sha256": format!("{byte:02x}").repeat(32) },
                "mime_type": "image/png",
                "width": 1024,
                "height": 768
            })
        };
        let request: mold_core::GenerateRequest = serde_json::from_value(serde_json::json!({
            "prompt": "descriptor-bound authority",
            "model": mold_core::minimax_h3::REF2VA_COMFY,
            "width": mold_core::minimax_h3::DEFAULT_WIDTH,
            "height": mold_core::minimax_h3::DEFAULT_HEIGHT,
            "steps": 4,
            "guidance": 0.0,
            "seed": 7,
            "batch_size": 1,
            "output_format": "mp4",
            "references": [descriptor("subject.png", 1), descriptor("style.png", 2)]
        }))
        .unwrap();
        let authenticated = crate::auth::ApiKeyAuthenticated {
            identity: "process-local".to_string(),
            durable_identity: "restart-stable".to_string(),
        };
        assert!(claims_private_ingress(&request));
        let captured = capture(&request, Some(&authenticated), "instance-a")
            .unwrap()
            .expect("a reviewed Ref2VA request captures private authority");
        assert_eq!(
            idempotency_subject_sha256(&request, Some(&authenticated), "instance-a").unwrap(),
            Some(
                crate::h3_private_bridge::capture_durable_h3_private_ingress(
                    &request,
                    Some(&authenticated),
                    "instance-a",
                )
                .unwrap()
                .unwrap()
                .idempotency_subject_sha256()
                .to_string()
            )
        );

        let restored = restore(&request, Some(&captured.envelope), "instance-a")
            .unwrap_or_else(|failure| panic!("{}", failure.message));
        let grant = restored.h3_grant().expect("restored H3 grant");
        grant.validate_bound_request(&request).unwrap();

        let mut reordered = request.clone();
        reordered.references.as_mut().unwrap().reverse();
        assert!(
            restore(&reordered, Some(&captured.envelope), "instance-a").is_err(),
            "a reordered descriptor list is a different request"
        );

        let mut plain = request.clone();
        plain.model = "flux-dev".to_string();
        plain.references = None;
        assert!(!claims_private_ingress(&plain));
        assert!(capture(&plain, Some(&authenticated), "instance-a")
            .unwrap()
            .is_none());
        assert!(matches!(
            restore(&plain, None, "instance-a"),
            Ok(RuntimeAuthority::None)
        ));
    }

    #[test]
    fn deferred_preparation_disposition_is_typed_by_recoverability() {
        assert_eq!(
            preparation_disposition(&crate::routes::ApiError::unknown_model("missing")),
            DurableDisposition::Hold { retryable: true }
        );
        assert_eq!(
            preparation_disposition(&crate::routes::ApiError::not_found("missing")),
            DurableDisposition::Hold { retryable: true }
        );
        assert_eq!(
            preparation_disposition(&crate::routes::ApiError::server_restarting("restart")),
            DurableDisposition::Retain
        );
        assert_eq!(
            preparation_disposition(&crate::routes::ApiError::validation("invalid")),
            DurableDisposition::Hold { retryable: false }
        );
    }
}
