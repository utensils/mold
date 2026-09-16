import Foundation

/// The result of a conditional GET.
///
/// `GET /api/gallery` has no pagination -- a host answers with its whole index
/// in one array (1,536 rows and 1.2 MB on a well-used machine). Re-fetching
/// that on every refresh is the wasteful path, so the app keeps the ETag and
/// lets the server say "nothing changed" in a few bytes.
public enum Fetched<Value: Sendable>: Sendable {
    case fresh(Value, etag: String?)
    case notModified

    public var value: Value? {
        if case let .fresh(value, _) = self { return value }
        return nil
    }

    public var etag: String? {
        if case let .fresh(_, etag) = self { return etag }
        return nil
    }
}
