import Foundation
import MoldClient
@testable import Mold

// `/api/activity` on the fake. The state is in `FakeExtras`, for the reason
// that type's own comment gives.
extension FakeBackend {
    func activity() async throws -> ActiveWorkSnapshot {
        try record("activity")
        await pause("activity")
        guard let snapshot = extras.activitySnapshot else { throw notPlanted() }
        return snapshot
    }
}
