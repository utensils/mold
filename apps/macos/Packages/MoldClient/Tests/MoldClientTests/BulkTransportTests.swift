import Foundation
import Testing
@testable import MoldClient

private final class BulkTransport: StubTransport {
    nonisolated(unsafe) static var timeouts: [TimeInterval] = []
    override class func response(for path: String) -> (status: Int, body: Data)? {
        (204, Data())
    }
    override func startLoading() {
        Self.timeouts.append(request.timeoutInterval)
        super.startLoading()
    }
}

@Suite(.serialized)
struct BulkTransportTests {
    @Test func destructiveRequestsAllowFilesystemWorkWithoutChangingProbeTimeouts() async throws {
        BulkTransport.timeouts = []
        let backend = BulkTransport.backend()
        try await backend.trash(["a.png"])
        try await backend.restoreFromTrash(["a.png"])
        try await backend.deleteForever(["a.png"])
        try await backend.emptyTrash()
        #expect(BulkTransport.timeouts == [300, 300, 300, 300])
        #expect(backend.request("/api/status").timeoutInterval == 10)
    }

    @Test func busyLifecycleActionsAreDisabledInBothMenuScopes() {
        for scope in [LibraryScopeKind.prints, .trash] {
            let plan = LibraryMenuPlan(scope: scope, count: 3, trashCount: 10, lifecycleBusy: true)
            let removals = plan.items.filter {
                switch $0.kind {
                case .trash?, .putBack?, .deleteForever?, .emptyTrash?: true
                default: false
                }
            }
            #expect(!removals.isEmpty)
            #expect(removals.allSatisfy { $0.isDisabled })
        }
    }
}
