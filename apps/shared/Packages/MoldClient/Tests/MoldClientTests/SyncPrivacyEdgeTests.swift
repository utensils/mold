import Foundation
import Testing
@testable import MoldClient

struct SyncPrivacyEdgeTests {
    private let local = UUID(), remote = UUID()

    private func edge(host: UUID?, file: String = "copy.png", sourceRoute: String? = "src",
                      destinationRoute: String? = "dst", instance: String? = "i") -> SyncPrivacyEdge {
        SyncPrivacyEdge(origin: PrintID(host: remote, filename: "a.png"), destinationHost: host,
                        destinationFilename: file, localHost: local, sourceRoute: sourceRoute,
                        destinationRoute: destinationRoute, destinationInstance: instance)
    }

    /// Reconcile stamps the local host on records written without one; the
    /// copy is the same print, so the identity clock must not move per print.
    @Test func anUnstampedDestinationIsTheLocalHost() {
        #expect(edge(host: nil) == edge(host: local))
        #expect(edge(host: nil) != edge(host: remote))
    }

    @Test func routeOrFilenameChangesAreIdentityChanges() {
        #expect(edge(host: local) != edge(host: local, file: "other.png"))
        #expect(edge(host: local) != edge(host: local, destinationRoute: "elsewhere"))
        #expect(edge(host: local) != edge(host: local, instance: "j"))
        #expect(edge(host: local) != edge(host: local, sourceRoute: nil))
    }

    @Test func authorityAdmitsOnlyCurrentRoutesAndInstances() {
        let current = [remote: "src", local: [local.uuidString, "dst", "i"].joined(separator: "\n")]
        #expect(edge(host: nil).admitted(by: current))
        #expect(edge(host: nil).admitted(by: nil))
        #expect(!edge(host: nil, sourceRoute: "stale").admitted(by: current))
        #expect(!edge(host: nil, sourceRoute: nil).admitted(by: current))
        #expect(!edge(host: nil, destinationRoute: nil).admitted(by: current))
        #expect(!edge(host: nil, instance: "replaced").admitted(by: current))
    }

    @Test func componentsJoinEveryAdmittedEdge() {
        let current = [remote: "src", local: [local.uuidString, "dst", "i"].joined(separator: "\n")]
        let edges = [edge(host: nil, file: "one.png"), edge(host: local, file: "two.png"),
                     edge(host: nil, file: "stale.png", sourceRoute: "old")]
        let links = SyncPrivacyEdge.components(edges, authority: current)
        let origin = PrintID(host: remote, filename: "a.png")
        #expect(links[PrintID(host: local, filename: "one.png")] == links[origin])
        #expect(links[PrintID(host: local, filename: "two.png")] == links[origin])
        #expect(links[PrintID(host: local, filename: "stale.png")] == nil)
    }
}
