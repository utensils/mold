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
        library.localSaveTask = Task {}
        let picture = LibraryEntry(host: remote, print: FakeFixtures.print("remote.png"))

        await library.saveLocally([picture])

        #expect(remoteBackend.callCount("media") == 0)
        #expect(library.localSaveAlertPresented)
        #expect(library.localSaveTask == nil)
        #expect(library.localSaveReport.contains("Start This Mac’s engine"))
    }

    @Test func remoteCollectionIsCreatedAndFiledOnlyOnThisMac() async {
        let local = MoldEngine.localHost(port: 7680, apiKey: "test")!
        let remote = host("remote")
        let localBackend = FakeBackend(host: local)
        let remoteBackend = FakeBackend(host: remote)
        remoteBackend.mediaAnswer = Data([1, 2, 3])
        remoteBackend.collectionRows = [Collection(id: "remote-id", name: "Night Sky",
                                                    slug: "night-sky")]
        let hosts = HostStore(hosts: [local, remote]) { machine in
            machine.id == local.id ? localBackend : remoteBackend
        }
        hosts.reachability[local.id] = .up(FakeFixtures.serverStatus())
        let library = LibraryStore(hosts: hosts)
        var mutable = GalleryPrint.Mutable(FakeFixtures.print("star.png"))
        mutable.collections = ["remote-id"]
        let print = mutable.build()
        remoteBackend.prints = [print]

        await library.saveLocally([LibraryEntry(host: remote, print: print)])

        #expect(localBackend.importedNames == ["star.png"])
        #expect(localBackend.mutationRequests.count == 1)
        #expect(localBackend.mutationRequests.first?.filenames == ["star.png"])
        #expect(localBackend.mutationRequests.first?.addToCollection?.name == "Night Sky")
        #expect(remoteBackend.callCount("mutate") == 0)
        #expect(localBackend.callCount("createCollection") == 0)
    }

    @Test func unavailableSourceCollectionsDoNotBlockPictureCopies() async {
        let local = MoldEngine.localHost(port: 7680, apiKey: "test")!
        let remote = host("remote")
        let localBackend = FakeBackend(host: local)
        let remoteBackend = FakeBackend(host: remote)
        remoteBackend.mediaAnswer = Data([1, 2, 3])
        remoteBackend.plantedErrors["collections"] = MoldClientError.malformedResponse
        let hosts = HostStore(hosts: [local, remote]) { machine in
            machine.id == local.id ? localBackend : remoteBackend
        }
        hosts.reachability[local.id] = .up(FakeFixtures.serverStatus())
        let library = LibraryStore(hosts: hosts)
        var mutable = GalleryPrint.Mutable(FakeFixtures.print("star.png"))
        mutable.collections = ["remote-id"]
        let print = mutable.build()
        let plain = FakeFixtures.print("plain.png")
        remoteBackend.prints = [print, plain]

        await library.saveLocally([LibraryEntry(host: remote, print: print),
                                   LibraryEntry(host: remote, print: plain)])

        #expect(Set(localBackend.importedNames) == ["star.png", "plain.png"])
        #expect(localBackend.mutationRequests.isEmpty)
        #expect(library.localSaveFailures.contains { $0.contains("Collections on") })
    }

    @Test func stoppedSaveDoesNotStartAnotherTransfer() async {
        let local = MoldEngine.localHost(port: 7680, apiKey: "test")!
        let remote = host("remote")
        let localBackend = FakeBackend(host: local)
        let remoteBackend = FakeBackend(host: remote)
        let print = FakeFixtures.print("star.png")
        remoteBackend.prints = [print]
        remoteBackend.mediaAnswer = Data([1, 2, 3])
        let hosts = HostStore(hosts: [local, remote]) { machine in
            machine.id == local.id ? localBackend : remoteBackend
        }
        hosts.reachability[local.id] = .up(FakeFixtures.serverStatus())
        let library = LibraryStore(hosts: hosts)
        library.localSaveStopRequested = true
        await library.saveLocally([LibraryEntry(host: remote, print: print)])
        #expect(localBackend.importedNames.isEmpty)
        #expect(library.localSaveFailures.contains { $0.contains("stopped") })
    }

    @Test func aFailedImportReportsOneFailureAndContinuesTheBatch() async {
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
        #expect(library.localSaveReport.contains("Copied 1 pictures"))
        #expect(library.localSaveReport.contains("1 were not copied"))
        #expect(library.localSaveFailures.contains { $0.contains("taken.png") })
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
