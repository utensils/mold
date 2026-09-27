import Foundation

/// Navigating the glTF JSON, one checked field at a time.
///
/// Port of the JSON helpers in `studio/lib/glb.ts:70-149`. The document is
/// untrusted `Any` until each field has been checked, so a missing or wrongly
/// typed field is a sentence rather than a crash.
enum GLBDocument {

    static func object(_ parent: [String: Any], _ key: String,
                       _ what: String) throws -> [String: Any] {
        guard let value = parent[key] as? [String: Any] else {
            throw GLBParseError("GLB \(what) is missing the \"\(key)\" object")
        }
        return value
    }

    static func array(_ parent: [String: Any], _ key: String,
                      _ what: String) throws -> [Any] {
        guard let value = parent[key] as? [Any] else {
            throw GLBParseError("GLB \(what) is missing the \"\(key)\" array")
        }
        return value
    }

    static func objectAt(_ array: [Any], _ index: Int,
                         _ what: String) throws -> [String: Any] {
        guard index >= 0, index < array.count else {
            throw GLBParseError(
                "GLB \(what) index \(index) is out of range (\(array.count) entries)")
        }
        guard let value = array[index] as? [String: Any] else {
            throw GLBParseError("GLB \(what) \(index) is not an object")
        }
        return value
    }

    /// A non-negative integer field, defaulted when absent — glTF's own rule.
    static func int(_ parent: [String: Any], _ key: String, _ what: String,
                    fallback: Int? = nil) throws -> Int {
        guard let raw = present(parent[key]) else {
            if let fallback { return fallback }
            throw GLBParseError("GLB \(what) is missing \"\(key)\"")
        }
        guard let value = integer(raw), value >= 0 else {
            throw GLBParseError(
                "GLB \(what) has a non-integer \"\(key)\": \(describe(raw))")
        }
        return value
    }

    static func string(_ parent: [String: Any], _ key: String,
                       _ what: String) throws -> String {
        guard let value = parent[key] as? String else {
            throw GLBParseError("GLB \(what) is missing the \"\(key)\" string")
        }
        return value
    }

    /// `nil` for an absent key AND for an explicit JSON `null`, which is what
    /// `value === undefined || value === null` means on the other side.
    static func present(_ value: Any?) -> Any? {
        guard let value, !(value is NSNull) else { return nil }
        return value
    }

    /// A JSON number that is a whole number, or nil.
    ///
    /// `true` is a `__NSCFBoolean` and bridges to a Swift integer, so it is
    /// ruled out by type id rather than by an `as?` that would accept it —
    /// `typeof value === "number"` refuses it on the TypeScript side.
    static func integer(_ value: Any) -> Int? {
        guard let number = number(value) else { return nil }
        guard number.rounded() == number, number.magnitude < 9_007_199_254_740_992 else {
            return nil
        }
        return Int(number)
    }

    /// A JSON number, never a boolean.
    static func number(_ value: Any) -> Double? {
        guard !isBoolean(value), let number = value as? NSNumber else { return nil }
        return number.doubleValue
    }

    static func isBoolean(_ value: Any) -> Bool {
        CFGetTypeID(value as AnyObject) == CFBooleanGetTypeID()
    }

    private static func describe(_ value: Any) -> String {
        if let number = value as? NSNumber { return number.stringValue }
        return String(describing: value)
    }
}
