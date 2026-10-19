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
        let tile = app.buttons["source-media-tile"].firstMatch
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
