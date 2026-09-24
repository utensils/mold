import Foundation
import MoldClient
import Testing

@testable import Mold

@MainActor
struct LibraryLocalSaveTests {
    private func host(_ name: String) -> MoldHost {
        MoldHost(name: name, baseURL: URL(string: "http://\(name)")!)
    }

    private func versionedPrint(_ name: String, prompt: String? = nil) -> GalleryPrint {
        let json = try! JSONSerialization.data(withJSONObject: [
            "filename": name, "timestamp": 1000, "size_bytes": 3,
            "media_version": "1000:3", "metadata": ["prompt": prompt.map { $0 as Any } ?? NSNull()],
        ])
        return try! MoldJSON.decoder.decode(GalleryPrint.self, from: json)
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

        #expect(remoteBackend.callCount("mediaFile") == 1)
        #expect(remoteBackend.callCount("gallery") == 1)
        #expect(localBackend.importedNames == ["remote.png"])
        #expect(localBackend.importedMedia.first == Data([1, 2, 3]))
        #expect(localBackend.importedItems.first?.originalMetadata?.prompt == "a fox")
        #expect(localBackend.callCount("gallery") == 2)
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

        #expect(remoteBackend.callCount("mediaFile") == 0)
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
        #expect(library.localSaveReport.contains("Copied 1 prints"))
        #expect(library.localSaveReport.contains("1 were not copied"))
        #expect(library.localSaveFailures.contains { $0.contains("taken.png") })
        #expect(library.localSaveAlertPresented)
        #expect(localBackend.callCount("gallery") == 2)
    }

    @Test func syncAllIgnoresSelectionAndCopiesEveryMediaKindAndEmptyCollections() async {
        let local = MoldEngine.localHost(port: 7680, apiKey: "test")!
        let remote = host("remote")
        let localBackend = FakeBackend(host: local)
        let remoteBackend = FakeBackend(host: remote)
        remoteBackend.mediaAnswer = Data([1, 2, 3])
        let print = try! MoldJSON.decoder.decode(GalleryPrint.self, from: Data(#"""
            {
            "filename":"star.png","timestamp":1000,
            "metadata":{"prompt":"star","model":"flux","seed":1},
            "collections":["remote-night"],"title":"A Star","favorite":true,"tags":["night"]
            }
            """#.utf8))
        remoteBackend.prints = [print, FakeFixtures.print("movie.mp4"),
                                FakeFixtures.print("shape.glb"), FakeFixtures.print("sound.wav")]
        remoteBackend.collectionRows = [
            Collection(id: "remote-night", name: "Night Sky", slug: "night-sky"),
            Collection(id: "remote-empty", name: "Empty Shelf", slug: "empty-shelf"),
        ]
        localBackend.collectionCreateResponses = [
            "Night Sky": Collection(id: "local-night", name: "Night Sky", slug: "night-sky"),
            "Empty Shelf": Collection(id: "local-empty", name: "Empty Shelf", slug: "empty-shelf"),
        ]
        let hosts = HostStore(hosts: [local, remote]) { machine in
            machine.id == local.id ? localBackend : remoteBackend
        }
        hosts.reachability[local.id] = .up(FakeFixtures.serverStatus())
        let library = LibraryStore(hosts: hosts)

        await library.syncAllLocally()

        #expect(Set(localBackend.importedNames) == ["star.png", "movie.mp4", "shape.glb", "sound.wav"])
        #expect(localBackend.callCount("createCollection") == 2)
        #expect(localBackend.mutationRequests.contains { $0.addToCollection?.name == "Night Sky"
            && $0.filenames == ["star.png"] })
        #expect(localBackend.mutationRequests.contains { $0.titles.contains {
            $0.filename == "star.png" && $0.title == "A Star" } })
        #expect(localBackend.mutationRequests.contains { $0.favorite == true
            && $0.filenames == ["star.png"] })
        #expect(localBackend.mutationRequests.contains { $0.addTags == ["night"]
            && $0.filenames == ["star.png"] })
        #expect(library.localSaveReport.contains("Created 2 collections"))
    }

    @Test func syncAllKeepsSameNamedPrintsFromDifferentHostsAndRerunsWithoutDuplicates() async {
        let local = MoldEngine.localHost(port: 7680, apiKey: "test")!
        let first = MoldHost(id: UUID(uuidString: "00000000-0000-0000-0000-000000000001")!,
                             name: "first", baseURL: URL(string: "http://first")!)
        let second = MoldHost(id: UUID(uuidString: "00000000-0000-0000-0000-000000000002")!,
                              name: "second", baseURL: URL(string: "http://second")!)
        let localBackend = FakeBackend(host: local)
        let firstBackend = FakeBackend(host: first)
        let secondBackend = FakeBackend(host: second)
        firstBackend.prints = [versionedPrint("same.png")]
        secondBackend.prints = [versionedPrint("same.png")]
        firstBackend.mediaAnswer = Data([1, 2, 3])
        secondBackend.mediaAnswer = Data([4, 5, 6])
        let hosts = HostStore(hosts: [local, first, second]) { machine in
            if machine.id == local.id { return localBackend }
            return machine.id == first.id ? firstBackend : secondBackend
        }
        hosts.reachability[local.id] = .up(FakeFixtures.serverStatus())
        let library = LibraryStore(hosts: hosts)

        await library.syncAllLocally()
        let copied = localBackend.importedNames
        #expect(copied.count == 2)
        #expect(Set(copied).count == 2)
        #expect(copied.contains("same.png"))
        for (name, bytes) in zip(localBackend.importedNames, localBackend.importedMedia) {
            localBackend.mediaAnswers[name] = bytes
            localBackend.prints.append(versionedPrint(name))
        }

        await library.syncAllLocally()

        #expect(localBackend.importedNames == copied)
        #expect(library.localSaveReport.contains("2 were already here"))
        let remoteReads = firstBackend.callCount("mediaFile") + secondBackend.callCount("mediaFile")
        await library.syncAllLocally()
        #expect(firstBackend.callCount("mediaFile") + secondBackend.callCount("mediaFile") == remoteReads)
    }

    @Test func syncAllKeepsDifferentRecipesEvenWhenMediaBytesMatch() async {
        let local = MoldEngine.localHost(port: 7680, apiKey: "test")!
        let remote = host("different-recipe")
        let localBackend = FakeBackend(host: local)
        let remoteBackend = FakeBackend(host: remote)
        let name = "same.png"
        localBackend.prints = [versionedPrint(name, prompt: "local recipe")]
        localBackend.mediaAnswers[name] = Data([1, 2, 3])
        remoteBackend.prints = [versionedPrint(name, prompt: "remote recipe")]
        remoteBackend.mediaAnswer = Data([1, 2, 3])
        let hosts = HostStore(hosts: [local, remote]) { machine in
            machine.id == local.id ? localBackend : remoteBackend
        }
        hosts.reachability[local.id] = .up(FakeFixtures.serverStatus())

        await LibraryStore(hosts: hosts).syncAllLocally()

        #expect(localBackend.importedNames.count == 1)
        #expect(localBackend.importedNames.first != name)
    }

    @Test func syncAllBoundsCollisionNamesToFileSystemLimit() async {
        let local = MoldEngine.localHost(port: 7680, apiKey: "test")!
        let remote = host("long-names")
        let localBackend = FakeBackend(host: local)
        let remoteBackend = FakeBackend(host: remote)
        let name = String(repeating: "🌟", count: 59) + ".png"
        localBackend.prints = [versionedPrint(name, prompt: "local")]
        localBackend.mediaAnswers[name] = Data([1, 2, 3])
        remoteBackend.prints = [versionedPrint(name, prompt: "remote")]
        remoteBackend.mediaAnswer = Data([4, 5, 6])
        let hosts = HostStore(hosts: [local, remote]) { machine in
            machine.id == local.id ? localBackend : remoteBackend
        }
        hosts.reachability[local.id] = .up(FakeFixtures.serverStatus())

        await LibraryStore(hosts: hosts).syncAllLocally()

        #expect(localBackend.importedNames.count == 1)
        #expect(localBackend.importedNames[0].utf8.count <= 255)
        #expect(localBackend.importedNames[0].hasSuffix(".png"))
    }

    @Test func syncAllReportsAnUnavailableRemoteHost() async {
        let local = MoldEngine.localHost(port: 7680, apiKey: "test")!
        let remote = host("unavailable")
        let localBackend = FakeBackend(host: local)
        let remoteBackend = FakeBackend(host: remote)
        remoteBackend.plantedErrors["gallery"] = MoldClientError.malformedResponse
        let hosts = HostStore(hosts: [local, remote]) { machine in
            machine.id == local.id ? localBackend : remoteBackend
        }
        hosts.reachability[local.id] = .up(FakeFixtures.serverStatus())
        let library = LibraryStore(hosts: hosts)

        await library.syncAllLocally()

        #expect(library.localSaveFailures.contains { $0.contains("unavailable") })
        #expect(library.localSaveReport.contains("issues need attention"))
    }

    @Test func failedOrganizationIsRetriedAfterMediaWasAlreadyPresent() async {
        let local = MoldEngine.localHost(port: 7680, apiKey: "test")!
        let remote = host("retry-organization")
        let localBackend = FakeBackend(host: local)
        let remoteBackend = FakeBackend(host: remote)
        var mutable = GalleryPrint.Mutable(versionedPrint("same.png", prompt: "recipe"))
        mutable.title = "Remote title"
        remoteBackend.prints = [mutable.build()]
        remoteBackend.mediaAnswer = Data([1, 2, 3])
        localBackend.plantedErrors["mutate"] = MoldClientError.malformedResponse
        let hosts = HostStore(hosts: [local, remote]) { machine in
            machine.id == local.id ? localBackend : remoteBackend
        }
        hosts.reachability[local.id] = .up(FakeFixtures.serverStatus())
        let library = LibraryStore(hosts: hosts)

        await library.syncAllLocally()
        #expect(localBackend.importedNames == ["same.png"])
        #expect(library.localSaveFailures.contains { $0.contains("Print titles") })

        localBackend.prints = [versionedPrint("same.png", prompt: "recipe")]
        localBackend.mediaAnswers["same.png"] = Data([1, 2, 3])
        localBackend.plantedErrors.removeValue(forKey: "mutate")
        await library.syncAllLocally()
        #expect(localBackend.mutationRequests.contains { $0.titles.contains {
            $0.filename == "same.png" && $0.title == "Remote title" } })
    }

    @Test func anAlreadyLocalPrintKeepsItsOwnTitleOnSyncAll() async {
        let local = MoldEngine.localHost(port: 7680, apiKey: "test")!
        let remote = host("existing-local-title")
        let localBackend = FakeBackend(host: local)
        let remoteBackend = FakeBackend(host: remote)
        var localMutable = GalleryPrint.Mutable(versionedPrint("same.png", prompt: "recipe"))
        localMutable.title = "My title"
        localBackend.prints = [localMutable.build()]
        localBackend.mediaAnswers["same.png"] = Data([1, 2, 3])
        var remoteMutable = GalleryPrint.Mutable(versionedPrint("same.png", prompt: "recipe"))
        remoteMutable.title = "Remote title"
        remoteBackend.prints = [remoteMutable.build()]
        remoteBackend.mediaAnswer = Data([1, 2, 3])
        let hosts = HostStore(hosts: [local, remote]) { machine in
            machine.id == local.id ? localBackend : remoteBackend
        }
        hosts.reachability[local.id] = .up(FakeFixtures.serverStatus())

        await LibraryStore(hosts: hosts).syncAllLocally()

        #expect(localBackend.importedNames.isEmpty)
        #expect(localBackend.mutationRequests.allSatisfy { $0.titles.isEmpty })
    }

    @Test func uncertainImportResponseKeepsPendingOrganizationForRetry() async {
        let local = MoldEngine.localHost(port: 7680, apiKey: "test")!
        let remote = host("uncertain-import")
        let localBackend = FakeBackend(host: local)
        let remoteBackend = FakeBackend(host: remote)
        var mutable = GalleryPrint.Mutable(versionedPrint("clip.mp4", prompt: "recipe"))
        mutable.title = "Remote clip"
        remoteBackend.prints = [mutable.build()]
        remoteBackend.mediaAnswer = Data([1, 2, 3])
        localBackend.importFailures = ["clip.mp4"]
        let hosts = HostStore(hosts: [local, remote]) { machine in
            machine.id == local.id ? localBackend : remoteBackend
        }
        hosts.reachability[local.id] = .up(FakeFixtures.serverStatus())
        let library = LibraryStore(hosts: hosts)

        await library.syncAllLocally()
        #expect(library.localSaveFailures.contains { $0.contains("clip.mp4") })

        // Simulate a server that committed before its response was lost.
        localBackend.prints = [versionedPrint("clip.mp4", prompt: "recipe")]
        localBackend.mediaAnswers["clip.mp4"] = Data([1, 2, 3])
        localBackend.importFailures = []
        await library.syncAllLocally()
        #expect(localBackend.importedNames.isEmpty)
        #expect(localBackend.mutationRequests.contains { $0.titles.contains {
            $0.filename == "clip.mp4" && $0.title == "Remote clip" } })
    }

    @Test func impossibleStagingSizeReportsFailureWithoutDownloading() async {
        let local = MoldEngine.localHost(port: 7680, apiKey: "test")!
        let remote = host("huge-gallery-row")
        let localBackend = FakeBackend(host: local)
        let remoteBackend = FakeBackend(host: remote)
        let row = try! JSONSerialization.data(withJSONObject: [
            "filename": "huge.mp4", "timestamp": 1000,
            "size_bytes": Int.max, "metadata": ["prompt": "clip"],
        ])
        remoteBackend.prints = [try! MoldJSON.decoder.decode(GalleryPrint.self, from: row)]
        let hosts = HostStore(hosts: [local, remote]) { machine in
            machine.id == local.id ? localBackend : remoteBackend
        }
        hosts.reachability[local.id] = .up(FakeFixtures.serverStatus())
        let library = LibraryStore(hosts: hosts)

        await library.syncAllLocally()

        #expect(remoteBackend.callCount("mediaFile") == 0)
        #expect(library.localSaveFailures.contains { $0.contains("enough free space") })
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
