//! Qwen Image 2.1's transparent-background (RGBA) prompt recipe.
//!
//! The model card's "Transparent Image Generation (RGBA)" section
//! (`Qwen/Qwen-Image-2.1` README, revision
//! `b3179ad355be050328e483a9dfdd9e60cd62adfa`, lines 96-108) gives the
//! recommended prompt format verbatim:
//!
//! > This is an RGBA image with transparency. A cute cartoon dragon sticker.
//! > The image has alpha channel and the background is transparent.
//!
//! i.e. a fixed prefix sentence, the description as its own sentence, and a
//! fixed suffix sentence. [`apply_rgba_prompt_recipe`] reproduces exactly
//! that shape.
//!
//! **Only the engine calls it**, on the positive prompt, when
//! `GenerateRequest.transparent_background == Some(true)`. The stored prompt
//! is never rewritten: Reuse would wrap it twice, Expand/Remix would rewrite
//! the boilerplate, Library search and titles would show it, and a client's
//! conditioning fingerprint would read a toggle flip as a prompt edit. The
//! expander is told about the toggle through `ExpandContext` instead.

use std::borrow::Cow;

/// The recipe's leading sentence, verbatim from the model card.
pub const RGBA_PROMPT_PREFIX: &str = "This is an RGBA image with transparency.";

/// The recipe's trailing sentence, verbatim from the model card.
pub const RGBA_PROMPT_SUFFIX: &str =
    "The image has alpha channel and the background is transparent.";

/// Wrap a description in the RGBA prompt recipe:
/// `"{PREFIX} {description}. {SUFFIX}"`.
///
/// The description is trimmed and ONE trailing `.`, `!` or `?` is removed so
/// the recipe's own sentence break is not doubled. Idempotent: a prompt that
/// already opens with the prefix (case-insensitively) is returned unchanged,
/// so a user who typed the recipe themselves is not wrapped twice.
pub fn apply_rgba_prompt_recipe(description: &str) -> Cow<'_, str> {
    let trimmed = description.trim();
    let already = trimmed
        .get(..RGBA_PROMPT_PREFIX.len())
        .is_some_and(|head| head.eq_ignore_ascii_case(RGBA_PROMPT_PREFIX));
    if already {
        return Cow::Borrowed(description);
    }
    let body = trimmed
        .strip_suffix(['.', '!', '?'])
        .unwrap_or(trimmed)
        .trim_end();
    if body.is_empty() {
        return Cow::Owned(format!("{RGBA_PROMPT_PREFIX} {RGBA_PROMPT_SUFFIX}"));
    }
    Cow::Owned(format!("{RGBA_PROMPT_PREFIX} {body}. {RGBA_PROMPT_SUFFIX}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_recipe_reproduces_the_model_card_example_exactly() {
        assert_eq!(
            apply_rgba_prompt_recipe("A cute cartoon dragon sticker"),
            "This is an RGBA image with transparency. A cute cartoon dragon sticker. \
             The image has alpha channel and the background is transparent."
        );
    }

    #[test]
    fn one_trailing_terminator_and_surrounding_space_are_dropped() {
        for input in [
            "A cute cartoon dragon sticker.",
            "  A cute cartoon dragon sticker!  ",
            "A cute cartoon dragon sticker?",
        ] {
            assert_eq!(
                apply_rgba_prompt_recipe(input),
                apply_rgba_prompt_recipe("A cute cartoon dragon sticker"),
                "{input:?}"
            );
        }
        // Only ONE terminator is replaced by the recipe's own full stop, so
        // an ellipsis survives as an ellipsis.
        assert_eq!(
            apply_rgba_prompt_recipe("a fox..."),
            format!("{RGBA_PROMPT_PREFIX} a fox... {RGBA_PROMPT_SUFFIX}")
        );
    }

    #[test]
    fn the_recipe_is_idempotent() {
        let once = apply_rgba_prompt_recipe("a red paper lantern").into_owned();
        assert_eq!(apply_rgba_prompt_recipe(&once), once);
        let shouted = "THIS IS AN RGBA IMAGE WITH TRANSPARENCY. a lantern";
        assert_eq!(apply_rgba_prompt_recipe(shouted), shouted);
    }

    #[test]
    fn an_empty_description_still_yields_the_recipe() {
        assert_eq!(
            apply_rgba_prompt_recipe("   "),
            format!("{RGBA_PROMPT_PREFIX} {RGBA_PROMPT_SUFFIX}")
        );
    }
}
