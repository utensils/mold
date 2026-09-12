//! What a failure is ABOUT: this model and this request, or the device.
//!
//! The server's worker breaker degrades a GPU after three consecutive
//! failures and stops scheduling anything on it for sixty seconds. That is
//! the right answer for a card that is wedged or faulting, and the wrong
//! answer for a checkpoint that produces a NaN: on a single-GPU host three
//! bad renders of ONE model took every OTHER model out of service, and the
//! client got "no enabled, healthy GPU device is available" for work the card
//! could have run.
//!
//! An engine that KNOWS its failure is about the model or the request says so
//! by ending the message with [`MODEL_SPECIFIC_FAILURE_MARKER`]. The server
//! keys on the marker rather than on prose — the same discipline
//! `ADMISSION_PRESSURE_MARKER` uses for memory pressure — so rewording a
//! message cannot silently start counting a model's fault as the card's.
//!
//! The default is deliberately the other way: an UNMARKED failure still
//! counts against the device. A classifier that had to recognise every device
//! fault would fail open on the one error nobody anticipated, and failing
//! open here means continuing to schedule onto a broken card.

/// Ends the message of a failure that is about the model or the request
/// rather than about the device.
///
/// It is a readable sentence rather than a token because it reaches the user
/// on the failing request itself.
pub const MODEL_SPECIFIC_FAILURE_MARKER: &str =
    "this is a model-specific failure, not a device fault";

/// Whether this message was marked model-specific by the engine that raised it.
pub fn message_is_model_specific_failure(message: &str) -> bool {
    message.contains(MODEL_SPECIFIC_FAILURE_MARKER)
}

/// Whether this error was marked model-specific by the engine that raised it.
///
/// Formatted with `{:#}` so a marked error still classifies after being
/// wrapped in context by a caller.
pub fn is_model_specific_failure(error: &anyhow::Error) -> bool {
    message_is_model_specific_failure(&format!("{error:#}"))
}

/// Build an error whose message carries the marker.
///
/// Engines should prefer this to writing the sentence by hand, so the marker
/// has exactly one spelling.
pub fn model_specific_error(message: impl std::fmt::Display) -> anyhow::Error {
    anyhow::anyhow!("{message}; {MODEL_SPECIFIC_FAILURE_MARKER}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_marked_error_classifies_as_model_specific() {
        let error = model_specific_error("non-finite prediction at denoise step 3");
        assert!(is_model_specific_failure(&error));
        assert!(
            error.to_string().starts_with("non-finite prediction"),
            "the marker must not displace the engine's own sentence: {error:#}"
        );
    }

    #[test]
    fn a_marked_error_survives_being_wrapped_in_context() {
        let error = model_specific_error("bad checkpoint tensor").context("loading transformer");
        assert!(
            is_model_specific_failure(&error),
            "the classifier reads the whole chain: {error:#}"
        );
    }

    /// The default is device-class, because failing open here means
    /// continuing to schedule onto a broken card.
    #[test]
    fn an_unmarked_error_is_not_model_specific() {
        for message in [
            "CUDA_ERROR_ILLEGAL_ADDRESS",
            "DeviceMemory: CUDA_ERROR_OUT_OF_MEMORY",
            "cudarc driver error",
            "something nobody anticipated",
        ] {
            assert!(
                !is_model_specific_failure(&anyhow::anyhow!("{message}")),
                "{message} must keep counting against the device"
            );
        }
    }
}
