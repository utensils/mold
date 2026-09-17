import Foundation

// The one place a reply becomes a value.
extension HTTPBackend {
    /// Decodes a reply, or says WHERE it could not.
    ///
    /// `.malformedResponse` is still the right thing to hand a caller --
    /// there is nothing an app can do about a field this build cannot read,
    /// and retrying will not help. But collapsing to it threw away the only
    /// fact that makes such a bug findable: which key, at which path, on
    /// which route. That answer goes to the log; the BODY never does, and
    /// neither does the concrete route, whose components ARE filenames and
    /// ids (`RouteTemplate`, applied by `TransportLog`).
    func decoded<T: Decodable>(_ type: T.Type, from data: Data, route: String) throws -> T {
        do {
            return try MoldJSON.decoder.decode(type, from: data)
        } catch let error as DecodingError {
            TransportLog.decodeFailure(route: route, type: type, failure: DecodingFailure.summary(error))
            throw MoldClientError.malformedResponse
        } catch {
            TransportLog.decodeFailure(route: route, type: type, failure: "unreadable")
            throw MoldClientError.malformedResponse
        }
    }
}
