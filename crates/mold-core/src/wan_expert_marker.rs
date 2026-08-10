//! Wan 2.2 A14B high/low expert markers in published names.
//!
//! One authority, shared by two consumers that must agree: the catalog pairs
//! Civitai's separately-published experts by these markers, and LoRA routing
//! infers which expert a user adapter belongs to from the same conventions.
//! A second, weaker tokenizer in either place would classify the same file
//! differently depending on which path reached it.
//!
//! Promoted from `mold-catalog::wan_a14b` (#784) when LoRA routing (#780)
//! became the second consumer.

/// What one name says about the expert role.
///
/// `NoMarker` and `Ambiguous` are deliberately distinct: an unmarked name may
/// still be resolved from another source, but a name carrying BOTH markers is
/// an explicit high+low bundle, and falling back to a one-sided reading would
/// classify a bundle as a single expert.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NameMarker {
    High,
    Low,
    NoMarker,
    Ambiguous,
}

/// Token spans of `name` (char indices): maximal alphanumeric runs split
/// at non-alphanumeric separators (space, underscore, hyphen, dot, …),
/// letter↔digit transitions, and camel-case boundaries. Markers are
/// matched against whole tokens so "Glow" never reads as a low marker
/// and "highlight" never reads as a high one.
pub fn token_spans(chars: &[char]) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let mut start: Option<usize> = None;
    for (i, &c) in chars.iter().enumerate() {
        if !c.is_ascii_alphanumeric() {
            if let Some(s) = start.take() {
                spans.push((s, i));
            }
            continue;
        }
        if let Some(s) = start {
            let p = chars[i - 1];
            let boundary = p.is_ascii_digit() != c.is_ascii_digit()
                || (p.is_ascii_lowercase() && c.is_ascii_uppercase())
                || (p.is_ascii_uppercase()
                    && c.is_ascii_uppercase()
                    && chars.get(i + 1).is_some_and(|n| n.is_ascii_lowercase()));
            if boundary {
                spans.push((s, i));
                start = Some(i);
            }
        } else {
            start = Some(i);
        }
    }
    if let Some(s) = start {
        spans.push((s, chars.len()));
    }
    spans
}

/// Marker classification for one token. A token IS a marker when it
/// equals `high`/`low`. One published convention glues the marker onto a
/// parameter-size suffix with no case boundary (`…_t2vA14BHIGH` /
/// `…_t2vA14BLOW`), so a token immediately preceded by a digit may also
/// END with a marker when at most two characters remain before it —
/// "14B" + "HIGH". Space-delimited words like "SLOW" or "Glow" are never
/// digit-preceded, so they stay non-markers. The returned prefix length
/// is what [`expert_key`] preserves ahead of the placeholder.
pub fn token_marker_suffix(chars: &[char], span: (usize, usize)) -> Option<(usize, NameMarker)> {
    let token: String = chars[span.0..span.1]
        .iter()
        .map(|c| c.to_ascii_lowercase())
        .collect();
    let marker = |s: &str| match s {
        "high" => Some(NameMarker::High),
        "low" => Some(NameMarker::Low),
        _ => None,
    };
    if let Some(m) = marker(&token) {
        return Some((0, m));
    }
    let digit_preceded = span.0 > 0 && chars[span.0 - 1].is_ascii_digit();
    if !digit_preceded {
        return None;
    }
    for suffix in ["high", "low"] {
        if let Some(prefix) = token.strip_suffix(suffix) {
            if prefix.len() <= 2 {
                return Some((prefix.len(), marker(suffix).expect("marker suffix")));
            }
        }
    }
    None
}

pub fn token_is_marker(chars: &[char], span: (usize, usize)) -> Option<NameMarker> {
    token_marker_suffix(chars, span).map(|(_, marker)| marker)
}

pub fn classify_name(name: &str) -> NameMarker {
    let chars: Vec<char> = name.chars().collect();
    let mut has_high = false;
    let mut has_low = false;
    for span in token_spans(&chars) {
        match token_is_marker(&chars, span) {
            Some(NameMarker::High) => has_high = true,
            Some(NameMarker::Low) => has_low = true,
            _ => {}
        }
    }
    match (has_high, has_low) {
        (true, false) => NameMarker::High,
        (false, true) => NameMarker::Low,
        // Merged republications, plain "v2.0-fp8" — no marker at all.
        (false, false) => NameMarker::NoMarker,
        // "high+low bundle" — explicitly both experts, never one.
        (true, true) => NameMarker::Ambiguous,
    }
}
