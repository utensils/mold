import Foundation

public extension HTTPBackend {
    /// What this machine will chain for one model.
    ///
    /// `fps` is sent because LTX-2's per-clip cap is a runtime duration and
    /// moves with it (`routes.rs:8497-8498`); omitting it takes the model's
    /// own default. A 404 is an unknown or unsupported model and reaches the
    /// caller as a refusal, which is the honest answer -- the client's own
    /// constants are for a host too OLD to publish the route, not for a model
    /// this host does not have.
    func chainLimits(model: String, fps: Int?) async throws -> ChainLimits {
        var path = "/api/capabilities/chain-limits?model=\(RouteEscaping.escapedQueryValue(model))"
        if let fps { path += "&fps=\(fps)" }
        return try await get(path)
    }
}
