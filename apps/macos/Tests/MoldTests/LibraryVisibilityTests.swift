import Foundation
import MoldClient
import Testing
@testable import Mold

@MainActor struct LibraryVisibilityTests {
    @Test func mixedHiddenFlagsProtectAndRepairEveryReplica() async {
        let firstHost = MoldHost(name: "This Mac", baseURL: URL(string: "http://local")!)
        let secondHost = MoldHost(name: "Workstation", baseURL: URL(string: "http://remote")!)
        let first = FakeBackend(host: firstHost), second = FakeBackend(host: secondHost)
        let slug = "visibility-\(UUID().uuidString)"
        first.collectionRows = [Collection(id: "local", name: "Hidden", slug: slug, hidden: true)]
        second.collectionRows = [Collection(id: "remote", name: "Hidden", slug: slug, hidden: false)]
        let hosts = HostStore(hosts: [firstHost, secondHost]) { $0.id == firstHost.id ? first : second }
        let library = LibraryStore(hosts: hosts)
        await library.refreshOrganization()
        #expect(library.hiddenCollectionIDs[firstHost.id] == ["local"])
        #expect(library.hiddenCollectionIDs[secondHost.id] == ["remote"])
        #expect(second.collectionRows.first?.hidden == true)
        #expect(second.callCount("updateCollection") == 1)
        await library.refreshOrganization()
        #expect(library.collectionVisibility.intents[slug] == nil)
    }
}
