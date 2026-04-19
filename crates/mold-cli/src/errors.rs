//! Error types shared across CLI commands.
//!
//! `RemoteInferenceError` tags an error as originating from a remote server so the
//! top-level error handler can distinguish remote failures (e.g. a CUDA OOM on the
//! server) from local-device failures and produce the right diagnostic message.

use std::fmt;

/// Wraps an `anyhow::Error` with the host URL of the remote server it came from.
///
/// Construct with [`RemoteInferenceError::wrap`]. Retrieve at the top-level error
/// handler via `err.downcast_ref::<RemoteInferenceError>()`.
#[derive(Debug)]
pub struct RemoteInferenceError {
    pub host: String,
    pub inner: anyhow::Error,
}

impl RemoteInferenceError {
    /// Wrap an existing error with a remote-server host tag.
    pub fn wrap(host: impl Into<String>, inner: anyhow::Error) -> anyhow::Error {
        anyhow::Error::new(RemoteInferenceError {
            host: host.into(),
            inner,
        })
    }
}

impl fmt::Display for RemoteInferenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Transparent: the wrapper exists to carry host metadata, not to rewrite
        // the user-visible error string. The top-level handler consults the host
        // via downcast_ref and formats a proper remote-context message.
        write!(f, "{}", self.inner)
    }
}

impl std::error::Error for RemoteInferenceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(AsRef::<dyn std::error::Error + 'static>::as_ref(
            &self.inner,
        ))
    }
}

/// Whether the generation that produced an error ran locally or on a remote server.
#[derive(Debug, Clone, Copy)]
pub enum OomContext<'a> {
    Local,
    Remote { host: &'a str },
}

/// True if `msg` looks like a GPU OOM error from candle/CUDA/Metal.
pub fn is_oom_message(msg: &str) -> bool {
    msg.contains("CUDA_ERROR_OUT_OF_MEMORY")
        || msg.contains("out of memory")
        || msg.contains("exceeds available VRAM")
        || msg.contains("Failed to create metal resource")
}

/// Build the human-readable OOM label + hint lines for the given context.
///
/// `macos_client` should be `cfg!(target_os = "macos")` at the call site — we take
/// it as a parameter so tests can exercise both platforms without cross-compiling.
/// `fallback` is the already-cleaned error text to use when we can't confidently
/// pick a backend label (e.g. unknown local error).
pub fn format_oom_message(
    msg: &str,
    ctx: OomContext<'_>,
    macos_client: bool,
    fallback: &str,
) -> (String, Vec<String>) {
    let label = match ctx {
        OomContext::Remote { host } => format!("Remote server GPU out of memory ({host})"),
        OomContext::Local => {
            // candle wraps Metal allocation failures as CUDA_ERROR_OUT_OF_MEMORY
            // on macOS, so we gate the "Metal" label on the client platform.
            let is_metal = macos_client
                && (msg.contains("CUDA_ERROR_OUT_OF_MEMORY")
                    || msg.contains("Failed to create metal resource"));
            let is_cuda = !macos_client && msg.contains("CUDA_ERROR_OUT_OF_MEMORY");
            if is_metal {
                "Metal out of memory".to_string()
            } else if is_cuda {
                "CUDA out of memory".to_string()
            } else {
                fallback.to_string()
            }
        }
    };

    let hints = match ctx {
        OomContext::Remote { .. } => vec![
            "The remote GPU ran out of memory during generation.".to_string(),
            "Try these fixes:".to_string(),
            String::new(),
            "    Reduce batch size:    --batch 1".to_string(),
            "    Use a smaller model:  mold run <model>:q4 \"...\"".to_string(),
            "    Lower resolution:     --width 512 --height 512".to_string(),
            String::new(),
            "  The server keeps models resident between requests; ask the".to_string(),
            "  operator to free VRAM (`mold unload`) if other workloads share".to_string(),
            "  the GPU. Use `--local` to force local inference instead.".to_string(),
        ],
        OomContext::Local => vec![
            "GPU ran out of memory during generation.".to_string(),
            "Try these fixes:".to_string(),
            String::new(),
            "    Reduce resolution:  --width 512 --height 512".to_string(),
            "    Use a smaller model: mold run <model>:q4 \"...\"".to_string(),
            String::new(),
            "  For img2img, the source image resolution is used by default.".to_string(),
            "  Override with --width/--height to reduce VRAM usage.".to_string(),
            "  Run 'mold list' to see available models and sizes.".to_string(),
        ],
    };

    (label, hints)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remote_oom_labels_host_regardless_of_client_platform() {
        // Simulate a macOS client talking to a remote CUDA server.
        let msg = "CUDA_ERROR_OUT_OF_MEMORY: out of memory";
        let (label, hints) = format_oom_message(
            msg,
            OomContext::Remote {
                host: "http://hal9000:7680",
            },
            true, // macOS client
            msg,
        );
        assert_eq!(
            label,
            "Remote server GPU out of memory (http://hal9000:7680)"
        );
        assert!(!label.contains("Metal"));
        let hints_blob = hints.join("\n");
        assert!(hints_blob.contains("--batch 1"));
        assert!(hints_blob.contains("--local"));
    }

    #[test]
    fn remote_oom_labels_host_on_linux_client() {
        let (label, _) = format_oom_message(
            "CUDA_ERROR_OUT_OF_MEMORY",
            OomContext::Remote {
                host: "http://gpu-box:7680",
            },
            false,
            "fallback",
        );
        assert_eq!(
            label,
            "Remote server GPU out of memory (http://gpu-box:7680)"
        );
    }

    #[test]
    fn local_metal_oom_on_macos_client() {
        let msg = "CUDA_ERROR_OUT_OF_MEMORY";
        let (label, hints) = format_oom_message(msg, OomContext::Local, true, msg);
        assert_eq!(label, "Metal out of memory");
        assert!(hints.iter().any(|h| h.contains("--width 512")));
    }

    #[test]
    fn local_metal_resource_failure_on_macos_client() {
        let msg = "Failed to create metal resource";
        let (label, _) = format_oom_message(msg, OomContext::Local, true, msg);
        assert_eq!(label, "Metal out of memory");
    }

    #[test]
    fn local_cuda_oom_on_linux_client() {
        let msg = "CUDA_ERROR_OUT_OF_MEMORY";
        let (label, _) = format_oom_message(msg, OomContext::Local, false, msg);
        assert_eq!(label, "CUDA out of memory");
    }

    #[test]
    fn local_cuda_oom_on_macos_client_not_mislabeled_as_cuda() {
        // When the client is macOS and the error came from local Metal (candle
        // reports it as CUDA_ERROR_OUT_OF_MEMORY), we must NOT say "CUDA".
        let msg = "CUDA_ERROR_OUT_OF_MEMORY";
        let (label, _) = format_oom_message(msg, OomContext::Local, true, msg);
        assert_ne!(label, "CUDA out of memory");
        assert_eq!(label, "Metal out of memory");
    }

    #[test]
    fn unknown_local_oom_falls_back_to_raw_message() {
        let msg = "exceeds available VRAM: need 10 GB, have 8 GB";
        let (label, _) = format_oom_message(msg, OomContext::Local, false, msg);
        assert_eq!(label, msg);
    }

    #[test]
    fn is_oom_message_detects_all_variants() {
        assert!(is_oom_message("CUDA_ERROR_OUT_OF_MEMORY"));
        assert!(is_oom_message("the GPU reports out of memory"));
        assert!(is_oom_message("exceeds available VRAM"));
        assert!(is_oom_message("Failed to create metal resource"));
        assert!(!is_oom_message("network timeout"));
    }

    #[test]
    fn remote_inference_error_preserves_display_transparently() {
        let inner = anyhow::anyhow!("CUDA_ERROR_OUT_OF_MEMORY: device ran out of memory");
        let wrapped = RemoteInferenceError::wrap("http://server:7680", inner);
        // Wrapper must not alter the user-visible error text.
        let rendered = format!("{wrapped}");
        assert!(rendered.contains("CUDA_ERROR_OUT_OF_MEMORY"));
        assert!(!rendered.contains("http://server:7680"));
    }

    #[test]
    fn remote_inference_error_downcasts_by_type() {
        let inner = anyhow::anyhow!("boom");
        let wrapped = RemoteInferenceError::wrap("http://server:7680", inner);
        let got = wrapped.downcast_ref::<RemoteInferenceError>();
        assert!(got.is_some());
        assert_eq!(got.unwrap().host, "http://server:7680");
    }

    #[test]
    fn remote_inference_error_inner_chain_does_not_duplicate_top_message() {
        // Regression for the duplicate `error:` / `cause:` lines on the generic
        // remote-error path. `main` prints the top-level display and then walks
        // `chain().skip(1)` for "caused by" lines; when the error came from the
        // remote path, we must iterate the *inner* chain so the wrapper (which
        // transparently forwards Display to `inner`) doesn't appear as both the
        // header and the first cause.
        let inner = anyhow::anyhow!("server error 500: boom");
        let wrapped = RemoteInferenceError::wrap("http://server:7680", inner);

        let top = format!("{wrapped}");
        assert_eq!(top, "server error 500: boom");

        // Simulate what main.rs does: grab the inner from the wrapper and walk
        // the inner's chain.
        let inner_ref = &wrapped
            .downcast_ref::<RemoteInferenceError>()
            .expect("wrapped")
            .inner;
        let causes: Vec<String> = inner_ref.chain().skip(1).map(|c| c.to_string()).collect();

        assert!(
            !causes.iter().any(|c| c == &top),
            "top message must not appear as its own cause: causes={causes:?}",
        );
    }

    #[test]
    fn remote_inference_error_downcasts_through_context() {
        // Simulate a caller adding `.context()` above our wrapper as the error
        // propagates up through `?`. We must still be able to recover the host.
        use anyhow::Context;
        let inner = anyhow::anyhow!("CUDA_ERROR_OUT_OF_MEMORY: out of memory");
        let wrapped = RemoteInferenceError::wrap("http://server:7680", inner);
        let with_ctx = Err::<(), _>(wrapped)
            .context("batch 2/5 failed")
            .unwrap_err();
        let got = with_ctx.downcast_ref::<RemoteInferenceError>();
        assert!(got.is_some(), "downcast must traverse context chain");
        assert_eq!(got.unwrap().host, "http://server:7680");
    }
}
