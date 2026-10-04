import Foundation

public extension RetainedSourceMedia {
    /// Mirrors keep the exact recipe. Archive-only completion facts may be
    /// absent from embedded metadata written before the generation completed.
    /// Conflicting facts present on both sides still refuse the copy.
    static func mirrorMetadataMatches(_ first: Data?, _ second: Data?) -> Bool {
        guard let first, let second,
              var left = try? JSONSerialization.jsonObject(with: first) as? [String: Any],
              var right = try? JSONSerialization.jsonObject(with: second) as? [String: Any]
        else { return false }
        for key in ["job_id", "generation_time_ms"] {
            let leftMissing = left[key] == nil || left[key] is NSNull
            let rightMissing = right[key] == nil || right[key] is NSNull
            if leftMissing || rightMissing {
                left.removeValue(forKey: key)
                right.removeValue(forKey: key)
            }
        }
        if let a = left["version"] as? String, let b = right["version"] as? String,
           a != b, shortVersion(a) == shortVersion(b),
           a == shortVersion(a) || b == shortVersion(b) {
            left["version"] = shortVersion(a)
            right["version"] = shortVersion(b)
        }
        return NSDictionary(dictionary: left).isEqual(to: right)
    }

    static func mirrorMetadataMatches(_ first: OutputMetadata?, _ second: OutputMetadata?) -> Bool {
        guard let first, let second else { return false }
        return mirrorMetadataMatches(try? MoldJSON.encoder.encode(first), try? MoldJSON.encoder.encode(second))
    }

    private static func shortVersion(_ version: String) -> String {
        // Only Mold's documented `version (revision date)` decoration is
        // removable; prerelease identifiers remain part of the version.
        guard let range = version.range(of: #" \([0-9a-f]{7,40} [0-9]{4}-[0-9]{2}-[0-9]{2}\)$"#,
                                        options: .regularExpression) else { return version }
        return String(version[..<range.lowerBound])
    }
}
