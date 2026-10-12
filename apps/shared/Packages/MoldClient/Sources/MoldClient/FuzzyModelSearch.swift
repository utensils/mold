import Foundation

/// Model visibility/ranking shared by native Mac and iOS. Fixtures mirror Studio.
public enum FuzzyModelSearch {
    private static func fold(_ value: String) -> String {
        let folded = value.precomposedStringWithCompatibilityMapping
            .folding(options: [.diacriticInsensitive, .caseInsensitive], locale: Locale(identifier: "en_US_POSIX"))
            .lowercased()
        return folded.components(separatedBy: CharacterSet.alphanumerics.inverted)
            .filter { !$0.isEmpty }.joined(separator: " ")
    }

    private static func distance(_ a: String, _ b: String) -> Int {
        let x = Array(a), y = Array(b)
        var before = Array(0...y.count), previous = Array(0...y.count)
        guard !x.isEmpty else { return y.count }
        for i in 1...x.count {
            var row = [i]
            if !y.isEmpty {
                for j in 1...y.count {
                    var cost = min(row[j - 1] + 1, previous[j] + 1, previous[j - 1] + (x[i - 1] == y[j - 1] ? 0 : 1))
                    if i > 1, j > 1, x[i - 1] == y[j - 2], x[i - 2] == y[j - 1] {
                        cost = min(cost, before[j - 2] + 1)
                    }
                    row.append(cost)
                }
            }
            before = previous; previous = row
        }
        return previous[y.count]
    }

    private static func tokenScore(_ token: String, _ text: String) -> Int {
        let words = text.split(separator: " ").map(String.init)
        if words.contains(token) { return 100 }
        if words.contains(where: { $0.hasPrefix(token) }) { return 80 }
        if text.contains(token) { return 60 }
        if text.replacingOccurrences(of: " ", with: "").contains(token) { return 55 }
        if token.unicodeScalars.contains(where: CharacterSet.decimalDigits.contains) { return 0 }
        let letters = Array(token)
        let limit = letters.count >= 8 ? 2 : letters.count >= 4 ? 1 : 0
        if limit > 0 {
            for word in words where word.count <= 64 && abs(word.count - letters.count) <= limit {
                if distance(token, word) <= limit { return 40 }
            }
        }
        if letters.count >= 2 {
            for word in words {
                let chars = Array(word)
                guard chars.first == letters.first, chars.count <= letters.count * 2,
                      chars.count - letters.count <= 3 else { continue }
                var next = 0
                for c in chars where next < letters.count {
                    if c == letters[next] { next += 1 }
                }
                if next == letters.count { return 20 }
            }
        }
        return 0
    }

    /// Every token must match. Names outrank metadata; zero means no match.
    public static func score(_ query: String, fields: [String], metadata: [String] = []) -> Int {
        guard query.utf16.count <= 128 else { return 0 }
        let q = fold(query)
        if q.isEmpty { return 1 }
        let tokens = q.split(separator: " ").map(String.init)
        guard tokens.count <= 12 else { return 0 }
        let primary = fields.map { fold(String($0.prefix(1024))) }
        let secondary = metadata.map { fold(String($0.prefix(1024))) }
        var result = 0
        for token in tokens {
            let best = max(primary.map { tokenScore(token, $0) }.max() ?? 0,
                           secondary.map { max(0, tokenScore(token, $0) - 15) }.max() ?? 0)
            guard best > 0 else { return 0 }
            result += best
        }
        return result + (primary.contains(q) ? 1000 : 0)
    }

    public static func ranked<T>(_ rows: [T], query: String, fields: (T) -> [String], metadata: (T) -> [String] = { _ in [] }) -> [T] {
        if query.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { return rows }
        // Explicit intermediate types keep Swift's generic constraint solver
        // bounded on both native targets (a chained tuple map exceeded its budget).
        var matches: [(index: Int, row: T, score: Int)] = []
        for (index, row) in rows.enumerated() {
            let primary = fields(row)
            let secondary = metadata(row)
            let value = score(query, fields: primary, metadata: secondary)
            if value > 0 { matches.append((index: index, row: row, score: value)) }
        }
        matches.sort { lhs, rhs in
            if lhs.score == rhs.score { return lhs.index < rhs.index }
            return lhs.score > rhs.score
        }
        return matches.map { $0.row }
    }

    /// Rank first, then group by the original family headings. An exact hit in
    /// a later family must precede an earlier family's fuzzy hit.
    public static func groups(_ families: [(family: String, models: [Model])], query: String) -> [(family: String, models: [Model])] {
        if query.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { return families }
        var familyFor: [String: String] = [:]
        for group in families {
            for model in group.models { familyFor[model.name] = group.family }
        }
        let matches = models(families.flatMap(\.models), query: query)
        var order: [String] = [], rows: [String: [Model]] = [:]
        for model in matches {
            let family = familyFor[model.name] ?? model.family
            if rows[family] == nil { order.append(family) }
            rows[family, default: []].append(model)
        }
        return order.map { (family: $0, models: rows[$0] ?? []) }
    }

    public static func models(_ models: [Model], query: String) -> [Model] {
        ranked(models, query: query, fields: { $0.searchFields }, metadata: { $0.searchMetadata })
    }
}

public extension Model {
    var searchFields: [String] { [name, headline, title, variantLabel ?? "", family, familyLabel, hfRepo ?? ""] }
    var searchMetadata: [String] { [summary ?? "", description] }
    func searchScore(_ query: String) -> Int { FuzzyModelSearch.score(query, fields: searchFields, metadata: searchMetadata) }
}
