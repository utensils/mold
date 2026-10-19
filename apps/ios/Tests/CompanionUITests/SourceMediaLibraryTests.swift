import XCTest

/// Native source browsing against synthetic loopback originals; never generates.
final class SourceMediaLibraryTests: XCTestCase {
    override func setUp() { super.setUp(); acceptCompanionPermissions() }

    @MainActor func testSourceLibraryIncludesHiddenOwnersAndReusesLibraryControls() async throws {
        continueAfterFailure = false
        let machine = try FixtureMachine(galleryPrints: 2, collectionFixture: true,
                                         hiddenCollectionFixture: true, retainedMediaFixture: true, sourceMediaLibraryFixture: true)
        let port = try await machine.start(), app = XCUIApplication()
        cleanUpFixture(machine, port: port, app: app)
        app.launch(); try addMachine(app, port: port)
        XCTAssertTrue(app.navigateToDestination("Library", shortcut: "2"))
        let ordinary = app.buttons.matching(NSPredicate(format: "label CONTAINS 'Fixture 1,'")).firstMatch
        XCTAssertTrue(ordinary.waitForExistence(timeout: 10), app.debugDescription)
        XCTAssertFalse(app.buttons.matching(NSPredicate(format: "label CONTAINS 'Fixture 0,'")).firstMatch.exists)
        let sources = app.buttons["Source Media"].firstMatch
        XCTAssertTrue(sources.waitForExistence(timeout: 5), app.debugDescription); sources.tap()
        let tile = app.descendants(matching: .any)["source-media-tile"].firstMatch
        XCTAssertTrue(tile.waitForExistence(timeout: 15), app.debugDescription)
        capture(app, "Source library includes hidden collection original")

        app.buttons["View Options"].firstMatch.tap()
        for label in ["All Collections", "UAT Drafts (Hidden)", "All media", "Images", "Videos"] {
            XCTAssertTrue(app.buttons[label].exists || app.staticTexts[label].exists, "Missing Library control \(label): \(app.debugDescription)")
        }
        let machineFilter = app.buttons["Machine"].firstMatch
        XCTAssertTrue(machineFilter.exists || app.buttons["All Machines"].exists, app.debugDescription)
        capture(app, "Source library consistent filter menu")
        app.buttons["UAT Drafts (Hidden)"].firstMatch.tap()
        XCTAssertTrue(tile.waitForExistence(timeout: 5), "Explicit hidden collection keeps its source visible")
        app.buttons["View Options"].firstMatch.tap()
        let chosenMachine = app.buttons["Sources UAT"].firstMatch
        if !chosenMachine.exists {
            let machineMenu = app.buttons.matching(NSPredicate(format: "label == 'Machine' OR label BEGINSWITH 'Machine,'")).firstMatch
            if machineMenu.exists { machineMenu.tap() }
        }
        XCTAssertTrue(chosenMachine.waitForExistence(timeout: 5), app.debugDescription); chosenMachine.tap()
        XCTAssertTrue(tile.waitForExistence(timeout: 5))
        app.buttons["View Options"].firstMatch.tap()
        app.buttons["Videos"].firstMatch.tap()
        XCTAssertTrue(tile.waitForNonExistence(timeout: 5), app.debugDescription)
        app.buttons["View Options"].firstMatch.tap(); app.buttons["Images"].firstMatch.tap()
        XCTAssertTrue(tile.waitForExistence(timeout: 5))
        app.buttons["View Options"].firstMatch.tap()
        let sizing = app.buttons.matching(NSPredicate(format: "label BEGINSWITH 'Tile Size'")).firstMatch
        if sizing.exists { sizing.tap() }
        app.buttons["Tiny"].firstMatch.tap()
        let before = tile.frame.width
        app.buttons["View Options"].firstMatch.tap()
        if sizing.exists { sizing.tap() }
        app.buttons["Largest"].firstMatch.tap()
        XCTAssertTrue(tile.waitForExistence(timeout: 5))
        XCTAssertGreaterThan(tile.frame.width, before + 40, "Source tile size follows the regular Library ladder")
        capture(app, "Source library resized tiles")
        tile.tap()
        XCTAssertTrue(app.navigationBars["Source details"].waitForExistence(timeout: 10), app.debugDescription)
        XCTAssertTrue(app.staticTexts["Original.png"].firstMatch.waitForExistence(timeout: 10), app.debugDescription)
        for _ in 0..<100 where !machine.requestLog().contains("GET /api/gallery/source-media/fixture-0.png/fixture-source/content") {
            try await Task.sleep(for: .milliseconds(100))
        }
        XCTAssertTrue(machine.requestLog().contains("GET /api/gallery/source-media/fixture-0.png/fixture-source/content"))
        capture(app, "Source details opens exact original")
        XCTAssertEqual(machine.requestLog().filter { $0 == "GET /api/gallery/source-media/fixture-0.png" }.count, 1,
                       "Filtering, resizing and details must reuse the cached inventory")
        XCTAssertTrue(machine.generationRequests.isEmpty)
    }

    @MainActor func testKnownSourceFreePrintHasNoSourceSectionOrRefreshSpinner() async throws {
        continueAfterFailure = false
        let machine = try FixtureMachine(galleryPrints: 1, sourceMediaLibraryFixture: true)
        let port = try await machine.start(), app = XCUIApplication()
        cleanUpFixture(machine, port: port, app: app)
        app.launch(); try addMachine(app, port: port)
        XCTAssertTrue(app.navigateToDestination("Library", shortcut: "2"))
        let picture = app.buttons.matching(NSPredicate(format: "label CONTAINS 'Fixture 0,'")).firstMatch
        XCTAssertTrue(picture.waitForExistence(timeout: 10)); picture.tap()
        let info = app.buttons["Info"].firstMatch
        XCTAssertTrue(info.waitForExistence(timeout: 5)); info.tap()
        let endpoint = "GET /api/gallery/source-media/fixture-0.png"
        for _ in 0..<100 where !machine.requestLog().contains(endpoint) { try await Task.sleep(for: .milliseconds(100)) }
        XCTAssertTrue(machine.requestLog().contains(endpoint))
        let infoBar = app.navigationBars["Info"]
        XCTAssertTrue(infoBar.waitForExistence(timeout: 5))
        let start = infoBar.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.1))
        start.press(forDuration: 0.1, thenDragTo: app.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.12)),
                    withVelocity: .slow, thenHoldForDuration: 0.2)
        XCTAssertFalse(app.staticTexts["Source Media"].exists, app.debugDescription)
        XCTAssertFalse(app.staticTexts["Loading sources…"].exists, app.debugDescription)
        let blankCells = app.collectionViews.cells.allElementsBoundByIndex.filter {
            $0.isHittable && $0.frame.height > 20 && $0.staticTexts.count == 0 && $0.textFields.count == 0 && $0.buttons.count == 0
        }
        XCTAssertTrue(blankCells.isEmpty, "A source-free print must not retain an empty Source Media List row: \(app.debugDescription)")
        let requests = machine.requestLog().filter { $0 == endpoint }.count
        try await Task.sleep(for: .seconds(2))
        XCTAssertEqual(machine.requestLog().filter { $0 == endpoint }.count, requests, "Negative inventory must not poll continuously")
        capture(app, "Known source-free print omits source section")
        XCTAssertTrue(machine.generationRequests.isEmpty)
    }

    @MainActor func testSourceTilesMatchDecodedAspectAndStayWarmThroughResizeAndSync() async throws {
        continueAfterFailure = false
        let machine = try FixtureMachine(galleryPrints: 1, retainedMediaFixture: true,
                                         sourceMediaLibraryFixture: true, sourceAspectFixture: true)
        let port = try await machine.start(), app = XCUIApplication()
        cleanUpFixture(machine, port: port, app: app)
        app.launch(); try addMachine(app, port: port)
        try openSources(app)
        try await assertTightSource(app, member: "fixture-portrait", ratio: 180.0 / 320)
        try await assertTightSource(app, member: "fixture-landscape", ratio: 320.0 / 180)
        let firstRow = [sourceTile(app, member: "fixture-portrait").frame,
                        sourceTile(app, member: "fixture-landscape").frame].sorted { $0.minX < $1.minX }
        XCTAssertEqual(firstRow[0].minY, firstRow[1].minY, accuracy: 1, "Mixed aspect sources share a justified row")
        XCTAssertEqual(firstRow[1].minX - firstRow[0].maxX, 2, accuracy: 1, "Use the Library's tight two-point seam")
        capture(app, "Portrait and landscape sources fit tight Library tiles")
        let initialRequests = sourceThumbnailRequests(machine)
        XCTAssertEqual(initialRequests.count, 2, "Each synthetic input should download only once")
        let inventoryEndpoint = "GET /api/gallery/source-media/fixture-0.png"
        let inventoryRequests = machine.requestLog().filter { $0 == inventoryEndpoint }.count
        XCTAssertEqual(inventoryRequests, 1)

        try chooseSourceSize(app, "Tiny")
        try await assertTightSource(app, member: "fixture-portrait", ratio: 180.0 / 320)
        let tinyWidth = sourceTile(app, member: "fixture-portrait").frame.width
        try chooseSourceSize(app, "Largest")
        try await assertTightSource(app, member: "fixture-portrait", ratio: 180.0 / 320)
        XCTAssertGreaterThan(sourceTile(app, member: "fixture-portrait").frame.width, tinyWidth + 10)
        try chooseSourceSize(app, "Tiny")
        try await assertTightSource(app, member: "fixture-landscape", ratio: 320.0 / 180)
        XCTAssertEqual(sourceThumbnailRequests(machine), initialRequests, "Changing geometry must retain decoded previews")

        for _ in 0..<100 where machine.eventConnectionCount == 0 { try await Task.sleep(for: .milliseconds(100)) }
        XCTAssertGreaterThan(machine.eventConnectionCount, 0, "The fixture must deliver real sync events")
        let galleryRequests = machine.requestLog().filter { $0 == "GET /api/gallery" }.count
        await machine.emitSourceDisplayOnlyUpdates(count: 20)
        for _ in 0..<50 where machine.requestLog().filter({ $0 == "GET /api/gallery" }).count <= galleryRequests {
            try await Task.sleep(for: .milliseconds(100))
        }
        XCTAssertGreaterThan(machine.requestLog().filter { $0 == "GET /api/gallery" }.count, galleryRequests,
                             "The app must consume and reconcile the emitted sync burst")
        for _ in 0..<12 {
            XCTAssertTrue(app.images["source-media-preview-fixture-portrait"].exists, "A display-only sync must not replace decoded pixels with a placeholder")
            XCTAssertTrue(app.images["source-media-preview-fixture-landscape"].exists)
            try await Task.sleep(for: .milliseconds(100))
        }
        XCTAssertEqual(sourceThumbnailRequests(machine), initialRequests, "Sync metadata must not redownload unchanged source bytes")
        XCTAssertEqual(machine.requestLog().filter { $0 == inventoryEndpoint }.count, inventoryRequests,
                       "Display-only changes must not restart source inventory scans")
        capture(app, "Sources remain decoded after repeated display-only sync events")
        XCTAssertTrue(machine.generationRequests.isEmpty)
    }

    @MainActor func testTransientThumbnailFailureRecoversWithoutScrolling() async throws {
        continueAfterFailure = false
        let machine = try FixtureMachine(galleryPrints: 1, retainedMediaFixture: true,
            sourceMediaLibraryFixture: true, sourceAspectFixture: true, failFirstSourceThumbnail: true)
        let port = try await machine.start(), app = XCUIApplication()
        cleanUpFixture(machine, port: port, app: app)
        app.launch(); try addMachine(app, port: port)
        try openSources(app)
        // No scroll, resize, refresh or reopening is allowed to rescue this load.
        try await assertTightSource(app, member: "fixture-portrait", ratio: 180.0 / 320)
        let endpoint = "GET /api/gallery/source-media/fixture-0.png/fixture-portrait/thumbnail"
        XCTAssertEqual(machine.requestLog().filter { $0 == endpoint }.count, 2,
                       "One transient failure must retry automatically and then stop")
        try await Task.sleep(for: .seconds(1))
        XCTAssertEqual(machine.requestLog().filter { $0 == endpoint }.count, 2)
        capture(app, "First thumbnail failure recovers in place without scrolling")
        XCTAssertTrue(machine.generationRequests.isEmpty)
    }

    @MainActor private func openSources(_ app: XCUIApplication) throws {
        XCTAssertTrue(app.navigateToDestination("Library", shortcut: "2"))
        let sources = app.buttons["Source Media"].firstMatch
        XCTAssertTrue(sources.waitForExistence(timeout: 10), app.debugDescription)
        sources.tap()
    }
    @MainActor private func sourceTile(_ app: XCUIApplication, member: String) -> XCUIElement {
        app.descendants(matching: .any).matching(identifier: "source-media-tile")
            .containing(.image, identifier: "source-media-preview-" + member).firstMatch
    }
    @MainActor private func assertTightSource(_ app: XCUIApplication, member: String, ratio: CGFloat) async throws {
        let image = app.images["source-media-preview-" + member]
        XCTAssertTrue(image.waitForExistence(timeout: 15), "The source must decode without scrolling: \(app.debugDescription)")
        let tile = sourceTile(app, member: member)
        XCTAssertTrue(tile.waitForExistence(timeout: 5), app.debugDescription)
        for _ in 0..<50 where abs(tile.frame.width / max(1, tile.frame.height) - ratio) > 0.03 {
            try await Task.sleep(for: .milliseconds(100))
        }
        let frame = tile.frame
        XCTAssertGreaterThan(frame.height, 20)
        XCTAssertEqual(frame.width / frame.height, ratio, accuracy: 0.03,
                       "A rectangular original must not sit inside a forced-square tile")
        XCTAssertEqual(image.frame.width, frame.width, accuracy: 2, "Image and tile must share their rendered bounds")
        XCTAssertEqual(image.frame.height, frame.height, accuracy: 2)
        let badge = app.descendants(matching: .any)["source-media-badge-" + member].firstMatch
        XCTAssertTrue(badge.exists, app.debugDescription)
        XCTAssertTrue(frame.insetBy(dx: -1, dy: -1).contains(badge.frame), "Media badge must remain inside the source image")
    }
    @MainActor private func chooseSourceSize(_ app: XCUIApplication, _ title: String) throws {
        app.buttons["View Options"].firstMatch.tap()
        let menu = app.buttons.matching(NSPredicate(format: "label BEGINSWITH 'Tile Size'")).firstMatch
        if menu.exists { menu.tap() }
        let size = app.buttons[title].firstMatch
        XCTAssertTrue(size.waitForExistence(timeout: 5), app.debugDescription); size.tap()
    }
    private func sourceThumbnailRequests(_ machine: FixtureMachine) -> [String] {
        machine.requestLog().filter { $0.hasPrefix("GET /api/gallery/source-media/") && $0.hasSuffix("/thumbnail") }.sorted()
    }

    @MainActor private func addMachine(_ app: XCUIApplication, port: UInt16) throws {
        XCTAssertTrue(app.navigateToDestination("Machines", shortcut: "5"))
        app.buttons["Add a Machine"].firstMatch.tap()
        app.buttons.matching(NSPredicate(format: "label BEGINSWITH 'Enter an Address'")).firstMatch.tap()
        app.buttons.matching(NSPredicate(format: "label BEGINSWITH 'Connection,'")).firstMatch.tap()
        app.buttons.matching(NSPredicate(format: "label BEGINSWITH 'Regular'")).firstMatch.tap()
        let name = app.textFields["machine-name"]
        XCTAssertTrue(name.waitForExistence(timeout: 5)); name.tap(); name.typeText("Sources UAT")
        let address = app.textFields["machine-address"]; address.tap(); address.typeText("127.0.0.1:\(port)")
        app.buttons["Add"].firstMatch.tap()
    }
    @MainActor private func capture(_ app: XCUIApplication, _ title: String) {
        let attachment = XCTAttachment(screenshot: app.screenshot())
        attachment.name = title; attachment.lifetime = .keepAlways; add(attachment)
    }
}
