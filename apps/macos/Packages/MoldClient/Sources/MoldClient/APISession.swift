import Foundation

/// The one `URLSession` every `HTTPBackend` talks through.
///
/// NOT `URLSession.shared`: that session keeps `URLCache.shared`, a SQLite
/// store at `~/Library/Caches/<bundle>/Cache.db`, and nothing this app sends
/// wants it -- a listing carries its own ETag (`Fetched`), an event stream
/// cannot be cached, and every picture lands in a cache the app owns and
/// bounds. Worse, any second `URLCache` built without a `directory` opens
/// that SAME file, and two caches on one file is a state Foundation does not
/// support: the console fills with `stepSQLStatement … result=1` while one
/// purges under the other. So the API session carries no URL cache at all.
public enum APISession {
    public static let api: URLSession = {
        let configuration = URLSessionConfiguration.default
        configuration.urlCache = nil
        configuration.requestCachePolicy = .reloadIgnoringLocalCacheData
        return URLSession(configuration: configuration)
    }()
}
