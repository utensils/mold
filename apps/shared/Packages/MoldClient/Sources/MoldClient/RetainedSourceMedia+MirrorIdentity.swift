import Foundation

public extension RetainedSourceMedia {
    /// Mirrors keep the exact recipe. Archive-only completion facts may be
    /// absent from embedded metadata written before the generation completed.
    /// Alpha is an output-container fact added by imports of older prints;
    /// it may be absent on one side when the output digest and size agree.
    /// Conflicting facts present on both sides still refuse the copy.
    static func mirrorMetadataMatches(_ first: Data?, _ second: Data?) -> Bool {
        mirrorMetadataMatches(first, second, verifiedEmbeddedRecipe: nil)
    }

    internal static func mirrorMetadataMatches(_ first: Data?, _ second: Data?,
                                               verifiedEmbeddedRecipe: Data?) -> Bool {
        guard let first, let second,
              var left = try? JSONSerialization.jsonObject(with: first) as? [String: Any],
              var right = try? JSONSerialization.jsonObject(with: second) as? [String: Any]
        else { return false }
        for key in ["job_id", "generation_time_ms", "has_alpha"] {
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
        if let verifiedEmbeddedRecipe,
           let embedded = try? JSONSerialization.jsonObject(with: verifiedEmbeddedRecipe) as? [String: Any] {
            // Older archives can omit these generation facts even though the
            // exact output's embedded recipe recorded them. Absence alone is
            // insufficient: the hashed output must corroborate the other side.
            for key in ["scheduler", "transparent_background"] {
                let leftMissing = left[key] == nil || left[key] is NSNull
                let rightMissing = right[key] == nil || right[key] is NSNull
                guard leftMissing != rightMissing, let recorded = embedded[key], !(recorded is NSNull),
                      let present = leftMissing ? right[key] : left[key],
                      NSDictionary(dictionary: [key: recorded]).isEqual(to: [key: present]) else { continue }
                left.removeValue(forKey: key)
                right.removeValue(forKey: key)
            }
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
