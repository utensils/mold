import Foundation

/// `URLSearchParams`' own serializer (WHATWG `application/x-www-form-urlencoded`):
/// alphanumerics plus `*-._` bare, space -> `+`, everything else
/// percent-encoded in uppercase hex. `URLComponents`' percent-encoding does
/// not match this -- a Swift-built pairing URL would spell `base_url` and any
/// hostname with a space differently from every code the studio has ever
/// printed (design fact 8).
enum FormURLEncoded {
    private static let unreserved = CharacterSet(
        charactersIn: "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._*")

    /// One value, WHATWG form-encoded.
    static func encode(_ value: String) -> String {
        var result = ""
        for scalar in value.unicodeScalars {
            if scalar == " " {
                result.append("+")
            } else if unreserved.contains(scalar) {
                result.unicodeScalars.append(scalar)
            } else {
                for byte in String(scalar).utf8 {
                    result += "%" + String(format: "%02X", byte)
                }
            }
        }
        return result
    }

    /// `name=value&name=value…`, in the order given -- callers control
    /// ordering because the wire's own field order matters for a
    /// byte-identical vector.
    static func queryString(_ pairs: [(String, String)]) -> String {
        pairs.map { "\(encode($0.0))=\(encode($0.1))" }.joined(separator: "&")
    }

    /// `URLSearchParams`' parser: `&`-separated, the first `=` splits name
    /// from value, `+` is a space, then percent-decoding. `URLComponents`'
    /// `queryItems` leaves `+` alone, which read "Studio Mac" as "Studio+Mac".
    static func decode(_ query: String) -> [(String, String)] {
        query.split(separator: "&", omittingEmptySubsequences: true).map { pair in
            let parts = pair.split(separator: "=", maxSplits: 1, omittingEmptySubsequences: false)
            func text(_ raw: Substring?) -> String {
                let spaced = (raw ?? "").replacingOccurrences(of: "+", with: " ")
                return spaced.removingPercentEncoding ?? spaced
            }
            return (text(parts.first), text(parts.count > 1 ? parts[1] : nil))
        }
    }
}
