//! Family-neutral durable admission-authority adapter.
//!
//! Generic queue code persists and restores only opaque envelopes plus a
//! typed disposition. Family policy remains behind this module.

/// Durable outcome for failures discovered only after acknowledgement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PreparationDisposition {
    Hold,
    HoldRetryable,
    Retain,
}

pub(crate) fn preparation_disposition(error: &crate::routes::ApiError) -> PreparationDisposition {
    match error.code.as_str() {
        "SERVER_RESTARTING" => PreparationDisposition::Retain,
        // These name dependencies the operator/user can install or make
        // reachable without changing the accepted request.
        "UNKNOWN_MODEL" | "MODEL_NOT_FOUND" => PreparationDisposition::HoldRetryable,
        _ if error.status() == axum::http::StatusCode::TOO_MANY_REQUESTS
            || error.status().is_server_error() =>
        {
            PreparationDisposition::HoldRetryable
        }
        _ => PreparationDisposition::Hold,
    }
}

pub(crate) struct Failure {
    pub disposition: PreparationDisposition,
    pub message: String,
}

pub(crate) struct CapturedAuthority {
    pub envelope: Vec<u8>,
    pub idempotency_subject_sha256: String,
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
            idempotency_subject_sha256: grant.idempotency_subject_sha256().to_string(),
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
        let requires_h3 = !mold_core::is_pinned_unrunnable_minimax_h3_identity(&request.model)
            && mold_core::minimax_h3::capability_contract_for_model(&request.model).is_some();
        let Some(envelope) = envelope else {
            return if requires_h3 {
                Err(Failure {
                    disposition: PreparationDisposition::Hold,
                    message: "durable MiniMax H3 admission authority is missing".into(),
                })
            } else {
                Ok(RuntimeAuthority::default())
            };
        };
        let envelope: AuthorityEnvelope =
            serde_json::from_slice(envelope).map_err(|_| Failure {
                disposition: PreparationDisposition::Hold,
                message: "durable admission authority envelope is invalid".into(),
            })?;
        if envelope.version != ENVELOPE_VERSION || envelope.kind != H3_PRIVATE_KIND || !requires_h3
        {
            return Err(Failure {
                disposition: PreparationDisposition::Hold,
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
            disposition: if error.code == crate::h3_private_bridge::H3_PRIVATE_RUNTIME_UNAVAILABLE {
                PreparationDisposition::HoldRetryable
            } else {
                PreparationDisposition::Hold
            },
            message: error.error,
        })
    }
    #[cfg(not(any(feature = "h3", feature = "h3-private-uat")))]
    {
        let _ = (request, instance_id);
        if envelope.is_some() {
            Err(Failure {
                disposition: PreparationDisposition::Hold,
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

    #[test]
    fn deferred_preparation_disposition_is_typed_by_recoverability() {
        assert_eq!(
            preparation_disposition(&crate::routes::ApiError::unknown_model("missing")),
            PreparationDisposition::HoldRetryable
        );
        assert_eq!(
            preparation_disposition(&crate::routes::ApiError::not_found("missing")),
            PreparationDisposition::HoldRetryable
        );
        assert_eq!(
            preparation_disposition(&crate::routes::ApiError::server_restarting("restart")),
            PreparationDisposition::Retain
        );
        assert_eq!(
            preparation_disposition(&crate::routes::ApiError::validation("invalid")),
            PreparationDisposition::Hold
        );
    }
}
