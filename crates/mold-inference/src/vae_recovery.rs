//! Decode recovery ordering, independent of tensors and GPU devices.

/// Resolved mode for the `MOLD_VAE_TILED` override.
///
/// `Auto` attempts whole decode and tiles only after OOM. `Force` starts with
/// tiles. `Off` surfaces whole-decode errors without a tiled retry.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum TiledMode {
    #[default]
    Auto,
    Force,
    Off,
}

/// Select the Metal-only policy without constructing accelerator devices.
pub(crate) fn decode_for_backend<T>(
    on_metal: bool,
    legacy: impl FnOnce() -> T,
    recovery: impl FnOnce() -> T,
) -> T {
    if on_metal {
        recovery()
    } else {
        legacy()
    }
}

/// Complete a submitted decode before exposing its output to postprocessing.
pub(crate) fn complete_decode<T, E>(
    decoded: Result<T, E>,
    complete: impl FnOnce() -> Result<(), E>,
) -> Result<T, E> {
    let output = decoded?;
    complete()?;
    Ok(output)
}

pub(crate) fn decode_with_recovery<T, E>(
    mode: TiledMode,
    whole: impl FnOnce() -> Result<T, E>,
    tiled: impl FnOnce() -> Result<T, E>,
    cleanup: impl FnOnce() -> Result<(), E>,
    is_oom: impl Fn(&E) -> bool,
) -> Result<T, E> {
    if mode == TiledMode::Force {
        return tiled();
    }
    match whole() {
        Err(error) if mode == TiledMode::Auto && is_oom(&error) => {
            cleanup_after_oom(cleanup, &is_oom)?;
            tiled()
        }
        result => result,
    }
}

/// Retry an exhausted GPU decode once on CPU. Callers gate this on the VAE device.
pub(crate) fn retry_on_oom<T, E>(
    result: Result<T, E>,
    on_metal: bool,
    cleanup: impl FnOnce() -> Result<(), E>,
    fallback: impl FnOnce() -> Result<T, E>,
    is_oom: impl Fn(&E) -> bool,
) -> Result<T, E> {
    match result {
        Err(error) if is_oom(&error) => {
            if on_metal {
                cleanup_after_oom(cleanup, &is_oom)?;
            } else {
                cleanup()?;
            }
            fallback()
        }
        result => result,
    }
}

fn cleanup_after_oom<E>(
    cleanup: impl FnOnce() -> Result<(), E>,
    is_oom: &impl Fn(&E) -> bool,
) -> Result<(), E> {
    match cleanup() {
        // Synchronization may report the same asynchronous failure that led here.
        Err(error) if is_oom(&error) => Ok(()),
        result => result,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Fault {
        Oom,
        Invalid,
    }

    fn is_oom(error: &Fault) -> bool {
        *error == Fault::Oom
    }

    fn attempt<T: Copy>(
        events: &RefCell<Vec<&'static str>>,
        name: &'static str,
        result: Result<T, Fault>,
    ) -> Result<T, Fault> {
        events.borrow_mut().push(name);
        result
    }

    #[test]
    fn whole_success_never_retries_or_cleans_up() {
        let events = RefCell::new(vec![]);
        let result = decode_with_recovery(
            TiledMode::Auto,
            || attempt(&events, "whole", Ok(7)),
            || attempt(&events, "tiles", Ok(8)),
            || attempt(&events, "cleanup", Ok(())),
            is_oom,
        );
        assert_eq!(result, Ok(7));
        assert_eq!(*events.borrow(), ["whole"]);
    }

    #[test]
    fn force_skips_whole_and_off_skips_tiles() {
        for (mode, expected) in [
            (TiledMode::Force, vec!["tiles"]),
            (TiledMode::Off, vec!["whole"]),
        ] {
            let events = RefCell::new(vec![]);
            let result: Result<(), Fault> = decode_with_recovery(
                mode,
                || attempt(&events, "whole", Err(Fault::Oom)),
                || attempt(&events, "tiles", Err(Fault::Oom)),
                || attempt(&events, "cleanup", Ok(())),
                is_oom,
            );
            assert_eq!(result, Err(Fault::Oom));
            assert_eq!(*events.borrow(), expected);
        }
    }

    #[test]
    fn whole_oom_cleans_up_then_retries_tiles_once() {
        for cleanup_result in [Ok(()), Err(Fault::Oom)] {
            let events = RefCell::new(vec![]);
            let result = decode_with_recovery(
                TiledMode::Auto,
                || attempt(&events, "whole", Err(Fault::Oom)),
                || attempt(&events, "tiles", Ok(8)),
                || attempt(&events, "cleanup", cleanup_result),
                is_oom,
            );
            assert_eq!(result, Ok(8));
            assert_eq!(*events.borrow(), ["whole", "cleanup", "tiles"]);
        }
    }

    #[test]
    fn non_oom_decode_and_cleanup_errors_propagate() {
        for (decode_error, expected) in [
            (Fault::Invalid, vec!["whole"]),
            (Fault::Oom, vec!["whole", "cleanup"]),
        ] {
            let events = RefCell::new(vec![]);
            let result = decode_with_recovery(
                TiledMode::Auto,
                || attempt(&events, "whole", Err(decode_error)),
                || attempt(&events, "tiles", Ok(8)),
                || attempt(&events, "cleanup", Err(Fault::Invalid)),
                is_oom,
            );
            assert_eq!(result, Err(Fault::Invalid));
            assert_eq!(*events.borrow(), expected);
        }
    }

    #[test]
    fn eager_cpu_recovery_survives_repeated_cleanup_oom_in_every_mode() {
        for mode in [TiledMode::Auto, TiledMode::Force, TiledMode::Off] {
            let events = RefCell::new(vec![]);
            let gpu = decode_with_recovery(
                mode,
                || attempt(&events, "whole", Err(Fault::Oom)),
                || attempt(&events, "tiles", Err(Fault::Oom)),
                || attempt(&events, "tile cleanup", Ok(())),
                is_oom,
            );
            let result = retry_on_oom(
                gpu,
                true,
                || attempt(&events, "cpu cleanup", Err(Fault::Oom)),
                || attempt(&events, "cpu", Ok(9)),
                is_oom,
            );
            assert_eq!(result, Ok(9));
            let expected = match mode {
                TiledMode::Auto => vec!["whole", "tile cleanup", "tiles", "cpu cleanup", "cpu"],
                TiledMode::Force => vec!["tiles", "cpu cleanup", "cpu"],
                TiledMode::Off => vec!["whole", "cpu cleanup", "cpu"],
            };
            assert_eq!(*events.borrow(), expected);
        }
    }

    #[test]
    fn cpu_recovery_preserves_non_oom_and_does_not_retry_cpu_failure() {
        let events = RefCell::new(vec![]);
        let result = retry_on_oom(
            Err(Fault::Invalid),
            true,
            || attempt(&events, "cleanup", Ok(())),
            || attempt(&events, "cpu", Ok(9)),
            is_oom,
        );
        assert_eq!(result, Err(Fault::Invalid));
        assert!(events.borrow().is_empty());
        let result: Result<(), Fault> = retry_on_oom(
            Err(Fault::Oom),
            true,
            || attempt(&events, "cleanup", Ok(())),
            || attempt(&events, "cpu", Err(Fault::Oom)),
            is_oom,
        );
        assert_eq!(result, Err(Fault::Oom));
        assert_eq!(*events.borrow(), ["cleanup", "cpu"]);
    }

    #[test]
    fn unrelated_cpu_cleanup_error_prevents_fallback() {
        let events = RefCell::new(vec![]);
        let result = retry_on_oom(
            Err(Fault::Oom),
            true,
            || attempt(&events, "cleanup", Err(Fault::Invalid)),
            || attempt(&events, "cpu", Ok(9)),
            is_oom,
        );
        assert_eq!(result, Err(Fault::Invalid));
        assert_eq!(*events.borrow(), ["cleanup"]);
    }
    #[test]
    fn non_metal_retains_legacy_decode_even_when_tiling_is_forced() {
        for on_metal in [false, true] {
            let events = RefCell::new(vec![]);
            let result = decode_for_backend(
                on_metal,
                || attempt(&events, "legacy whole", Ok(7)),
                || {
                    decode_with_recovery(
                        TiledMode::Force,
                        || attempt(&events, "whole", Ok(7)),
                        || attempt(&events, "tiles", Ok(8)),
                        || attempt(&events, "cleanup", Ok(())),
                        is_oom,
                    )
                },
            );
            assert_eq!(result, Ok(if on_metal { 8 } else { 7 }));
            assert_eq!(
                *events.borrow(),
                if on_metal {
                    vec!["tiles"]
                } else {
                    vec!["legacy whole"]
                }
            );
        }
    }

    #[test]
    fn cuda_keeps_direct_cpu_fallback_and_propagates_cleanup_failure() {
        for cleanup_result in [Ok(()), Err(Fault::Oom), Err(Fault::Invalid)] {
            let events = RefCell::new(vec![]);
            let gpu = decode_for_backend(
                false,
                || attempt(&events, "legacy whole", Err(Fault::Oom)),
                || attempt(&events, "new recovery", Ok(8)),
            );
            let result = retry_on_oom(
                gpu,
                false,
                || attempt(&events, "legacy cleanup", cleanup_result),
                || attempt(&events, "cpu", Ok(9)),
                is_oom,
            );
            match cleanup_result {
                Ok(()) => {
                    assert_eq!(result, Ok(9));
                    assert_eq!(*events.borrow(), ["legacy whole", "legacy cleanup", "cpu"]);
                }
                Err(error) => {
                    assert_eq!(result, Err(error));
                    assert_eq!(*events.borrow(), ["legacy whole", "legacy cleanup"]);
                }
            }
        }
    }

    #[test]
    fn delayed_completion_oom_enters_tiled_recovery_before_output_is_returned() {
        let events = RefCell::new(vec![]);
        let result = decode_with_recovery(
            TiledMode::Auto,
            || {
                complete_decode(attempt(&events, "submit whole", Ok(7)), || {
                    attempt(&events, "whole completion", Err(Fault::Oom))
                })
            },
            || {
                complete_decode(attempt(&events, "submit tiles", Ok(8)), || {
                    attempt(&events, "tile completion", Ok(()))
                })
            },
            || attempt(&events, "cleanup", Err(Fault::Oom)),
            is_oom,
        );
        assert_eq!(result, Ok(8));
        assert_eq!(
            *events.borrow(),
            [
                "submit whole",
                "whole completion",
                "cleanup",
                "submit tiles",
                "tile completion"
            ]
        );
    }

    #[test]
    fn failed_decode_never_completes_or_replaces_original_error() {
        let events = RefCell::new(vec![]);
        let result: Result<(), Fault> = complete_decode(Err(Fault::Invalid), || {
            attempt(&events, "completion", Err(Fault::Oom))
        });
        assert_eq!(result, Err(Fault::Invalid));
        assert!(events.borrow().is_empty());
    }
}
