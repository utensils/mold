//! Which convolution backend a family renders on.
//!
//! candle has two CUDA convolution implementations:
//!
//! * **im2col** — materialise the column buffer, then one GEMM. Always
//!   available, and what every mold build used before this module existed.
//! * **cuDNN** — NVIDIA's own kernels. Compiled in only with the `cudnn`
//!   feature, and chosen per convolution: the fork skips it for shapes where
//!   its uncached per-call setup costs more than it saves, and falls back to
//!   im2col if it errors.
//!
//! The two sum in a different order, so they do not agree bit-for-bit. That
//! makes the choice the same product decision [`crate::attention`] already
//! faces, and it is answered the same way (#736, #1483):
//!
//! * [`ConvPolicy::Image`] — im2col, in every build. An archived still seed is
//!   expected to re-render the same bytes forever.
//! * [`ConvPolicy::Video`] — cuDNN wherever it is compiled in. A clip is
//!   re-rendered for its content, and cuDNN is worth a measured **4.4x** on
//!   the convolutions of a Wan VAE decode (845 ms -> 192 ms per latent frame,
//!   `wan22-t2v-a14b:q5` 832x480 on an RTX 4090; see `website/models/wan.md`).
//!
//! `MOLD_CONV={cudnn,im2col}` overrides both directions. It shapes output, so
//! it is registered in [`crate::runtime_env::ENGINE_SHAPING_VARIABLES`].
//!
//! # Why a scope guard and not a per-call argument
//!
//! candle's switch is one process-global flag, because a convolution deep
//! inside a VAE has no idea which model family invoked it. That is sound here
//! for a reason mold already relies on elsewhere: at most one engine is
//! GPU-resident and generation is serialised behind `AppState.model_cache`
//! (CLAUDE.md decision 4), so exactly one family is convolving at a time.
//! [`ConvScope`] makes that explicit — it applies the policy for a render and
//! restores the previous value on drop, so a panic or an early return cannot
//! leave the next family running on the wrong backend.
//!
//! The process default is im2col ([`install_process_default`]), not candle's
//! own default of "on". Anything that has not deliberately opted in stays on
//! the byte-stable path.

use candle_core::cudnn_policy;
use std::sync::{Once, OnceLock};

/// Which family is asking, and therefore which default applies when the
/// operator has expressed no preference through `MOLD_CONV`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvPolicy {
    /// Stills. Always im2col, in every build.
    Image,
    /// Clips. cuDNN wherever the feature is compiled in.
    Video,
}

/// The resolved convolution backend for a render.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvBackend {
    /// candle's column-buffer + GEMM path.
    Im2Col,
    /// candle's cuDNN path (shape-gated and fallback-guarded inside candle).
    Cudnn,
}

impl ConvBackend {
    /// The provenance token for this backend, for plan fingerprints and logs.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Im2Col => "im2col",
            Self::Cudnn => "cudnn",
        }
    }
}

/// Whether this build can take the cuDNN path at all.
pub fn cudnn_compiled() -> bool {
    cudnn_policy::is_compiled()
}

/// The family-scoped default, mirroring [`crate::attention::policy_for_family`].
///
/// Deliberately the same family list: the argument for accepting different
/// bytes is about clips versus stills, not about which operator is doing the
/// arithmetic.
pub fn policy_for_family(family: &str) -> ConvPolicy {
    match family {
        "wan" | "ltx2" | "ltx-2" | "ltx-2.3" => ConvPolicy::Video,
        _ => ConvPolicy::Image,
    }
}

/// Parse `MOLD_CONV`. Public because `mold-server`'s execution-equivalence
/// classifier calls it: every other entry in that table hand-mirrors its
/// engine's parser and carries a comment promising the two stay in step, which
/// is a promise a refactor can quietly break. Sharing the function cannot drift.
///
/// `None` means "the operator said nothing", which is a
/// different answer from "the operator asked for im2col" and is what the
/// per-family default keys on. An unparseable value is also `None` — a typo
/// must not silently pin a family to the other backend.
pub fn parse_backend_env(raw: Option<&str>) -> Option<ConvBackend> {
    let value = raw?;
    match value.trim().to_ascii_lowercase().as_str() {
        "cudnn" => Some(ConvBackend::Cudnn),
        "im2col" | "math" => Some(ConvBackend::Im2Col),
        other if !other.is_empty() => {
            tracing::warn!(
                "MOLD_CONV={other} is not one of cudnn/im2col; using the family default"
            );
            None
        }
        _ => None,
    }
}

/// Resolve the backend a policy renders on, honouring `MOLD_CONV`.
///
/// Asking for `cudnn` in a build without the feature yields [`ConvBackend::Im2Col`]:
/// the request is not silently honoured, because there is nothing to honour it
/// with.
pub fn resolve_for(policy: ConvPolicy) -> ConvBackend {
    let requested = requested_backend_env();
    let wanted = requested.unwrap_or(match policy {
        ConvPolicy::Image => ConvBackend::Im2Col,
        ConvPolicy::Video => ConvBackend::Cudnn,
    });
    match wanted {
        ConvBackend::Cudnn if cudnn_compiled() => ConvBackend::Cudnn,
        _ => ConvBackend::Im2Col,
    }
}

/// Read and cache the `MOLD_CONV` request, disclosing the resolved policy once.
///
/// Caching the *request* rather than a resolved backend is what lets one
/// `OnceLock` serve both policies, mirroring `attention::requested_backend_env`.
/// The log line matters more here than it looks: which convolution backend ran
/// is otherwise invisible in a render's output, so without it the only evidence
/// is the wall clock — and inferring a code path from a timing is how a change
/// that never fired gets believed.
fn requested_backend_env() -> Option<ConvBackend> {
    static CACHED: OnceLock<Option<ConvBackend>> = OnceLock::new();
    *CACHED.get_or_init(|| {
        let requested = parse_backend_env(crate::runtime_env::value("MOLD_CONV").as_deref());
        tracing::info!(
            requested = ?requested,
            compiled = cudnn_compiled(),
            image_default = ?ConvBackend::Im2Col,
            video_default = ?if cudnn_compiled() { ConvBackend::Cudnn } else { ConvBackend::Im2Col },
            "convolution backend policy resolved"
        );
        requested
    })
}

/// Pin the process default to im2col.
///
/// candle defaults its own switch to "on", which would put every family on
/// cuDNN the moment the feature is compiled — including the still families
/// whose bytes must not move. Idempotent; called from engine construction.
pub fn install_process_default() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        cudnn_policy::set_enabled(false);
    });
}

/// Applies a convolution policy for as long as it is alive, then restores
/// whatever was in effect before.
///
/// Restoring on drop rather than resetting to a fixed value is what makes this
/// safe to nest and safe across an error return: a Wan render that fails
/// halfway cannot leave cuDNN enabled for the next still.
#[must_use = "the policy is only in effect while the scope is alive"]
pub struct ConvScope {
    previous: bool,
    backend: ConvBackend,
}

impl ConvScope {
    /// Apply the resolved backend for `family`.
    pub fn for_family(family: &str) -> Self {
        Self::apply(resolve_for(policy_for_family(family)))
    }

    /// Apply an explicitly resolved backend.
    pub fn apply(backend: ConvBackend) -> Self {
        install_process_default();
        let previous = cudnn_policy::set_enabled(backend == ConvBackend::Cudnn);
        tracing::debug!(backend = backend.as_str(), "convolution scope applied");
        Self { previous, backend }
    }

    /// The backend this scope put in effect.
    pub fn backend(&self) -> ConvBackend {
        self.backend
    }
}

impl Drop for ConvScope {
    fn drop(&mut self) {
        cudnn_policy::set_enabled(self.previous);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The switch these tests assert on is one process-global flag, so they
    /// have to run one at a time or they read each other's writes. That is not
    /// a flaw in the design — production has the same single global, and the
    /// serialisation there is the scheduler running one engine at a time.
    static GLOBAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn serially<T>(f: impl FnOnce() -> T) -> T {
        let guard = GLOBAL.lock().unwrap_or_else(|e| e.into_inner());
        let out = f();
        drop(guard);
        out
    }

    #[test]
    fn video_families_match_the_attention_policy_list() {
        // The two policies answer the same clips-versus-stills question, so a
        // family that is Video for attention and Image for convolutions (or
        // the reverse) is a bug, not a design.
        for family in [
            "wan",
            "ltx2",
            "ltx-2",
            "ltx-2.3",
            "flux",
            "flux2",
            "sd15",
            "sdxl",
            "sd3",
            "z-image",
            "qwen-image",
            "wuerstchen",
            "ltx-video",
            "minimax-h3",
        ] {
            let conv = matches!(policy_for_family(family), ConvPolicy::Video);
            let attn = matches!(
                crate::attention::policy_for_family(family),
                crate::attention::AttentionPolicy::Video
            );
            assert_eq!(
                conv, attn,
                "{family} disagrees between conv and attention policy"
            );
        }
    }

    #[test]
    fn stills_never_default_to_cudnn() {
        // The #736 contract: compiling a feature must never silently change
        // the bytes an archived image seed renders.
        assert_eq!(resolve_for(ConvPolicy::Image), ConvBackend::Im2Col);
    }

    #[test]
    fn clips_take_cudnn_exactly_when_it_is_compiled() {
        let expected = if cudnn_compiled() {
            ConvBackend::Cudnn
        } else {
            ConvBackend::Im2Col
        };
        assert_eq!(resolve_for(ConvPolicy::Video), expected);
    }

    #[test]
    fn env_parsing_distinguishes_silence_from_a_choice() {
        assert_eq!(parse_backend_env(None), None);
        assert_eq!(parse_backend_env(Some("")), None);
        assert_eq!(
            parse_backend_env(Some("  CuDNN ")),
            Some(ConvBackend::Cudnn)
        );
        assert_eq!(parse_backend_env(Some("im2col")), Some(ConvBackend::Im2Col));
        // `math` is the word MOLD_ATTN uses for "the portable path"; accepting
        // it here costs nothing and is what an operator will try first.
        assert_eq!(parse_backend_env(Some("math")), Some(ConvBackend::Im2Col));
        // A typo falls back to the family default rather than pinning a family
        // to the backend the operator did not ask for.
        assert_eq!(parse_backend_env(Some("cudnnn")), None);
    }

    #[test]
    fn a_scope_restores_the_previous_backend_on_drop() {
        serially(|| {
            install_process_default();
            let before = cudnn_policy::is_enabled();
            {
                let scope = ConvScope::apply(ConvBackend::Cudnn);
                assert_eq!(scope.backend(), ConvBackend::Cudnn);
                assert_eq!(cudnn_policy::is_enabled(), cudnn_compiled());
            }
            assert_eq!(cudnn_policy::is_enabled(), before);
        });
    }

    #[test]
    fn a_scope_restores_even_when_the_render_unwinds() {
        serially(|| {
            install_process_default();
            let before = cudnn_policy::is_enabled();
            let hook = std::panic::take_hook();
            std::panic::set_hook(Box::new(|_| {}));
            let unwound = std::panic::catch_unwind(|| {
                let _scope = ConvScope::apply(ConvBackend::Cudnn);
                panic!("render failed halfway");
            });
            std::panic::set_hook(hook);
            assert!(unwound.is_err());
            assert_eq!(
                cudnn_policy::is_enabled(),
                before,
                "a failed render left the next family on the wrong convolution backend"
            );
        });
    }

    #[test]
    fn the_process_default_is_im2col_not_candles_own_default() {
        serially(|| {
            install_process_default();
            assert!(
                !cudnn_policy::is_enabled(),
                "candle defaults its switch on; mold must pin it off so image \
                 families stay byte-stable"
            );
        });
    }
}
