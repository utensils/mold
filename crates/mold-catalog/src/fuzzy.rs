//! Bounded candidate matching for live discovery. Never used for model identity.
use unicode_normalization_alignments::{char::is_combining_mark, UnicodeNormalization};

pub(crate) fn fold(value: &str) -> String {
    value
        .nfkd()
        .map(|(c, _)| c)
        .filter(|c| !is_combining_mark(*c))
        .flat_map(char::to_lowercase)
        .map(|c| if c.is_alphanumeric() { c } else { ' ' })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn distance(a: &str, b: &str) -> usize {
    let x: Vec<_> = a.chars().collect();
    let y: Vec<_> = b.chars().collect();
    let mut before: Vec<_> = (0..=y.len()).collect();
    let mut previous = before.clone();
    for i in 1..=x.len() {
        let mut row = vec![i];
        for j in 1..=y.len() {
            let mut cost = (row[j - 1] + 1)
                .min(previous[j] + 1)
                .min(previous[j - 1] + usize::from(x[i - 1] != y[j - 1]));
            if i > 1 && j > 1 && x[i - 1] == y[j - 2] && x[i - 2] == y[j - 1] {
                cost = cost.min(before[j - 2] + 1);
            }
            row.push(cost);
        }
        before = previous;
        previous = row;
    }
    previous[y.len()]
}

fn token_score(token: &str, text: &str) -> usize {
    let words: Vec<_> = text.split_whitespace().collect();
    if words.contains(&token) {
        return 100;
    }
    if words.iter().any(|word| word.starts_with(token)) {
        return 80;
    }
    if text.contains(token) {
        return 60;
    }
    if text.replace(' ', "").contains(token) {
        return 55;
    }
    if token.chars().any(char::is_numeric) {
        return 0;
    }
    let letters: Vec<_> = token.chars().collect();
    let limit = if letters.len() >= 8 {
        2
    } else {
        usize::from(letters.len() >= 4)
    };
    for word in &words {
        let len = word.chars().count();
        if limit > 0
            && len <= 64
            && len.abs_diff(letters.len()) <= limit
            && distance(token, word) <= limit
        {
            return 40;
        }
    }
    if letters.len() >= 2 {
        for word in words {
            let chars: Vec<_> = word.chars().collect();
            if chars.first() != letters.first()
                || chars.len() > letters.len() * 2
                || chars.len().saturating_sub(letters.len()) > 3
            {
                continue;
            }
            let mut next = 0;
            for c in chars {
                if letters.get(next) == Some(&c) {
                    next += 1;
                }
            }
            if next == letters.len() {
                return 20;
            }
        }
    }
    0
}

pub(crate) fn score(query: &str, fields: &[&str], metadata: &[&str]) -> usize {
    if query.encode_utf16().count() > 128 {
        return 0;
    }
    let q = fold(query);
    if q.is_empty() {
        return 1;
    }
    let tokens: Vec<_> = q.split_whitespace().collect();
    if tokens.len() > 12 {
        return 0;
    }
    let primary: Vec<_> = fields
        .iter()
        .map(|f| fold(&f.chars().take(1024).collect::<String>()))
        .collect();
    let secondary: Vec<_> = metadata
        .iter()
        .map(|f| fold(&f.chars().take(1024).collect::<String>()))
        .collect();
    let mut result = 0;
    for token in tokens {
        let best = primary
            .iter()
            .map(|text| token_score(token, text))
            .chain(
                secondary
                    .iter()
                    .map(|text| token_score(token, text).saturating_sub(15)),
            )
            .max()
            .unwrap_or(0);
        if best == 0 {
            return 0;
        }
        result += best;
    }
    result + if primary.contains(&q) { 1000 } else { 0 }
}

/// One bounded broadened query, never an empty/unfiltered provider request.
/// A two-letter prefix survives a typo after the first two characters; the
/// original tokens still have to match each returned candidate locally.
pub(crate) fn candidate_query(query: &str) -> Option<String> {
    if query.encode_utf16().count() > 128 {
        return None;
    }
    let q = fold(query);
    let token = q
        .split_whitespace()
        .find(|token| token.chars().count() >= 3 && token.chars().all(char::is_alphabetic))?;
    let prefix: String = token.chars().take(2).collect();
    (prefix != query.trim()).then_some(prefix)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn matches_the_native_and_studio_contract() {
        let fields = [
            "FLUX.1 Dev Q4",
            "flux-dev:q4",
            "black-forest-labs/FLUX.1-dev",
        ];
        for q in [
            "FLUX",
            "dev flux",
            "flux_dev:q4",
            "fluxdevq4",
            "flx dev",
            "flxu dev",
            "fluz dev",
            "fluxx dev",
            "flx q4",
            "forest q4",
        ] {
            assert!(score(q, &fields, &[]) > 0, "{q}");
        }
        for q in ["q8", "banana", "flux banana", "dx", "4q"] {
            assert_eq!(score(q, &fields, &[]), 0, "{q}");
        }
        assert!(score("cafe video", &["Café", "Video"], &[]) > 0);
        assert!(score("猫", &["猫のモデル"], &[]) > 0);
        assert_eq!(score(&"x".repeat(10000), &fields, &[]), 0);
        assert!(score("flux", &["flux"], &[]) > score("flux", &["fluxx"], &[]));
    }
    #[test]
    fn fallback_is_never_unfiltered() {
        assert_eq!(candidate_query("flxu dev"), Some("fl".into()));
        for query in ["", "q4", "fl", "42", "x", " --- "] {
            assert_eq!(candidate_query(query), None);
        }
    }
}
