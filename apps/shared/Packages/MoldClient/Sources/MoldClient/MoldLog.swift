import Foundation
import os

/// Where this package's diagnostics go.
///
/// One subsystem and a category per concern, so
/// `log stream --predicate 'subsystem == "io.utensils.mold.native"'` is the
/// whole of it and a category can be followed on its own.
///
/// NOTHING a person owns and nothing that is a credential is ever logged: not
/// an API key, not a media ticket, not a prompt, not a filename, not a
/// response body. What a diagnostic says is WHERE something failed -- a
/// route, a status, a type, a coding path -- which is exactly the fact an
/// error that collapses to `.malformedResponse` throws away. Everything is
/// interpolated `.public` BECAUSE of that rule: a log redacted down to
/// `<private>` is a log nobody can read, and there is nothing here to redact.
public enum MoldLog {
    public static let subsystem = "io.utensils.mold.native"

    /// Requests and their refusals: the route FAMILY (`RouteTemplate`, never
    /// the concrete path, whose components are filenames and ids), method,
    /// status, mold's error code. Never a body and never a header.
    static let transport = Logger(subsystem: subsystem, category: "transport")

    /// Event-stream lifecycle: opened, refused, ended.
    static let stream = Logger(subsystem: subsystem, category: "stream")

    /// A reply this build could not read, and where in it.
    static let decoding = Logger(subsystem: subsystem, category: "decoding")
}

/// A `DecodingError` reduced to the fact that makes it findable.
///
/// Built from the error's STRUCTURED parts rather than its
/// `debugDescription`: the description is written for a developer looking at
/// the data and can quote it, and this goes to a log that must never carry a
/// prompt or a filename. A key name, a type name and a coding path name the
/// shape, never the content.
enum DecodingFailure {
    static func summary(_ error: DecodingError) -> String {
        switch error {
        case let .typeMismatch(type, context):
            "type mismatch, expected \(type) at \(path(context))"
        case let .valueNotFound(type, context):
            "no value for \(type) at \(path(context))"
        case let .keyNotFound(key, context):
            "no key \(name(key)) at \(path(context))"
        case let .dataCorrupted(context):
            "corrupt data at \(path(context))"
        @unknown default:
            "unreadable"
        }
    }

    /// `children.0.state`, with an array index as its number rather than
    /// Foundation's "Index 0" prose.
    private static func path(_ context: DecodingError.Context) -> String {
        let steps = context.codingPath.map(name)
        return steps.isEmpty ? "the root" : steps.joined(separator: ".")
    }

    private static func name(_ key: any CodingKey) -> String {
        key.intValue.map(String.init) ?? key.stringValue
    }
}
