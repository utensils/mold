import Foundation

/// Percent-encoding for the two positions a route has, which take DIFFERENT
/// rules. `RouteEscapingContractTests` reads the source to check every dynamic
/// interpolation takes the one its position needs.
enum RouteEscaping {
    /// One path component, whatever a person put in it.
    ///
    /// A tag can hold a slash, a space or a `#`, and a print's filename folds
    /// in a title slug. `/` is subtracted from the allowed set BECAUSE it is
    /// allowed in a path: left alone it would silently split one component
    /// into two and address a different route.
    static func escaped(_ component: String) -> String {
        component.addingPercentEncoding(
            withAllowedCharacters: .urlPathAllowed.subtracting(CharacterSet(charactersIn: "/"))
        ) ?? component
    }

    /// One QUERY VALUE, whatever a person put in it.
    ///
    /// A DIFFERENT rule from `escaped(_:)`, and the difference is silent:
    /// `.urlPathAllowed` includes `&`, `=`, `+`, `;`, `$` and `,`, every one
    /// of which is structure in a query string. Searching Discover for
    /// `cats & dogs` sent `q=cats%20&%20dogs`, which the host reads as
    /// `q = "cats "` plus an unrelated empty parameter -- the wrong search,
    /// answered, with no error anywhere. A literal `+` is worse still: axum's
    /// `Query` parses through `serde_urlencoded`, which decodes it as a space,
    /// so `C++` arrives as `C  `.
    ///
    /// `#` and space are already outside `.urlQueryAllowed`. Everything else
    /// legal in a query value is left readable -- a `:` in a model tag stays
    /// a `:`, because over-encoding is noise in every proxy log between here
    /// and the machine.
    ///
    /// NOT `URLComponents.queryItems`: its setter leaves `+` alone, which is
    /// the same bug with more ceremony.
    static func escapedQueryValue(_ value: String) -> String {
        value.addingPercentEncoding(withAllowedCharacters: queryValueAllowed) ?? value
    }

    private static let queryValueAllowed: CharacterSet =
        .urlQueryAllowed.subtracting(CharacterSet(charactersIn: "&+=;,$"))
}
