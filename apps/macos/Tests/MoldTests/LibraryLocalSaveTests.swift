import Foundation
import MoldClient
import Testing

@testable import Mold

@MainActor
struct LibraryLocalSaveTests {
    private func host(_ name: String) -> MoldHost {
        MoldHost(name: name, baseURL: URL(string: "http://\(name)")!)
    }

    @Test func savingAFilteredRemoteSelectionImportsOnlyPicturesIntoThisMac() async {
        let local = MoldEngine.localHost(port: 7680, apiKey: "test")!
        let remote = host("remote")
        let localBackend = FakeBackend(host: local)
        let remoteBackend = FakeBackend(host: remote)
        remoteBackend.mediaAnswer = Data([1, 2, 3])
        let hosts = HostStore(hosts: [local, remote]) { host in
            host.id == local.id ? localBackend : remoteBackend
        }
        hosts.reachability[local.id] = .up(FakeFixtures.serverStatus())
        let library = LibraryStore(hosts: hosts)
        let picture = LibraryEntry(host: remote, print: FakeFixtures.print("remote.png", prompt: "a fox"))
        remoteBackend.prints = [picture.print]
        let localRow = LibraryEntry(host: local, print: FakeFixtures.print("mine.png"))
        let mesh = LibraryEntry(host: remote, print: FakeFixtures.print("mesh.glb"))

        await library.saveLocally([picture, localRow, mesh])

        #expect(remoteBackend.callCount("media") == 1)
        #expect(remoteBackend.callCount("gallery") == 1)
        #expect(localBackend.importedNames == ["remote.png"])
        #expect(localBackend.importedItems.first?.file == Data([1, 2, 3]))
        #expect(localBackend.importedItems.first?.originalMetadata?.prompt == "a fox")
        #expect(localBackend.callCount("gallery") == 1)
        #expect(library.localSaveReport.contains("Skipped 2"))
    }

    @Test func anUnavailableLocalEngineExplainsWhyNothingWasSaved() async {
        let remote = host("remote")
        let remoteBackend = FakeBackend(host: remote)
        let hosts = HostStore(hosts: [remote]) { _ in remoteBackend }
        let library = LibraryStore(hosts: hosts)
        let picture = LibraryEntry(host: remote, print: FakeFixtures.print("remote.png"))

        await library.saveLocally([picture])

        #expect(remoteBackend.callCount("media") == 0)
        #expect(library.localSaveAlertPresented)
        #expect(library.localSaveReport.contains("Start This Mac’s engine"))
    }

    @Test func aFilenameCollisionReportsOneFailureAndContinuesTheBatch() async {
        let local = MoldEngine.localHost(port: 7680, apiKey: "test")!
        let remote = host("remote")
        let localBackend = FakeBackend(host: local)
        let remoteBackend = FakeBackend(host: remote)
        remoteBackend.mediaAnswer = Data([1, 2, 3])
        let hosts = HostStore(hosts: [local, remote]) { host in
            host.id == local.id ? localBackend : remoteBackend
        }
        hosts.reachability[local.id] = .up(FakeFixtures.serverStatus())
        let library = LibraryStore(hosts: hosts)
        let collision = LibraryEntry(host: remote, print: FakeFixtures.print("taken.png"))
        let available = LibraryEntry(host: remote, print: FakeFixtures.print("free.png"))
        remoteBackend.prints = [collision.print, available.print]
        localBackend.importFailures = ["taken.png"]

        await library.saveLocally([collision, available])

        #expect(localBackend.importedNames == ["free.png"])
        #expect(library.localSaveReport.contains("Saved 1 of 2"))
        #expect(library.localSaveReport.contains("taken.png"))
        #expect(library.localSaveAlertPresented)
        #expect(localBackend.callCount("gallery") == 1)
    }

    @Test func remoteTrashDoesNotSendALocalCopyToTrash() async {
        let local = MoldEngine.localHost(port: 7680, apiKey: "test")!
        let remote = host("remote")
        let localBackend = FakeBackend(host: local)
        let remoteBackend = FakeBackend(host: remote)
        let hosts = HostStore(hosts: [local, remote]) { host in
            host.id == local.id ? localBackend : remoteBackend
        }
        let library = LibraryStore(hosts: hosts)
        let remoteRow = LibraryEntry(host: remote, print: FakeFixtures.print("same.png"))
        let localRow = LibraryEntry(host: local, print: FakeFixtures.print("same.png"))
        library.perHost[remote.id] = [remoteRow]
        library.perHost[local.id] = [localRow]
        library.rebuild()

        await library.moveToTrash([remoteRow])

        #expect(remoteBackend.callCount("trash") == 1)
        #expect(localBackend.callCount("trash") == 0)
        #expect(library.items.map(\.id) == [localRow.id])
    }
}
