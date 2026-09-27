import Foundation

public extension HTTPBackend {
    /// `GET /api/activity` (`routes_activity.rs:180-396`).
    ///
    /// Read on a timer while the app is frontmost, so it takes the ordinary
    /// idle timeout: a machine that cannot answer inside it has something
    /// more wrong than a slow snapshot, and the next tick asks again.
    func activity() async throws -> ActiveWorkSnapshot {
        try await get("/api/activity")
    }
}
