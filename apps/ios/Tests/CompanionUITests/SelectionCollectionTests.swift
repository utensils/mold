import XCTest

/// Unknown capability inventory must not erase native organization actions.
final class SelectionCollectionTests: XCTestCase {
    override func setUp() { super.setUp(); acceptCompanionPermissions() }

    @MainActor func testSelectedPrintsCanFileIntoExistingAndNewCollections() async throws {
        try await exerciseCollections(size: "UICTContentSizeCategoryL")
    }

    @MainActor func testSelectionCollectionsAtAX5() async throws {
        try await exerciseCollections(size: "UICTContentSizeCategoryAccessibilityXXXL")
    }

    @MainActor private func exerciseCollections(size: String) async throws {
        continueAfterFailure = false
        let machine = try FixtureMachine(galleryPrints: 3, collectionFixture: true,
                                         omitOrganizationCapabilities: true, libraryMutations: true)
        let port = try await machine.start()
        let app = XCUIApplication()
        cleanUpFixture(machine, port: port, app: app)
        app.launchArguments = ["-UIPreferredContentSizeCategoryName", size]
        app.launch()
        XCTAssertTrue(app.navigateToDestination("Machines", shortcut: "5"))
        app.buttons["Add a Machine"].firstMatch.tap()
        app.buttons.matching(NSPredicate(format: "label BEGINSWITH 'Enter an Address'")).firstMatch.tap()
        app.buttons.matching(NSPredicate(format: "label BEGINSWITH 'Connection'")).firstMatch.tap()
        app.buttons["Regular"].firstMatch.tap()
        let address = app.textFields["machine-address"]
        XCTAssertTrue(address.waitForExistence(timeout: 5))
        address.tap(); address.typeText("127.0.0.1:\(port)")
        app.buttons["Add"].firstMatch.tap()
        XCTAssertTrue(app.navigateToDestination("Library", shortcut: "2"))
        let tiles = (1...2).map { index in
            app.buttons.matching(NSPredicate(format: "label CONTAINS %@", "Fixture \(index),")).firstMatch
        }
        XCTAssertTrue(tiles[0].waitForExistence(timeout: 10))
        if !app.buttons["Select"].exists { app.buttons["OverflowBarButtonItem"].tap() }
        app.buttons["Select"].firstMatch.tap()
        for tile in tiles { tile.tap() }
        let collection = app.buttons["Add to Collection"].firstMatch
        guard collection.waitForExistence(timeout: 5) else {
            XCTFail("Selected prints must offer Add to Collection even with unknown capabilities")
            return
        }
        collection.tap()
        capture(app, "Selected prints collection menu at \(size)")
        app.buttons["UAT Drafts"].firstMatch.tap()
        guard waitForMutations(machine, count: 1) else { XCTFail("Existing collection filing was not delivered"); return }
        try checkMutation(machine.libraryMutationRequests[0], name: "UAT Drafts")
        XCTAssertTrue(tiles.allSatisfy(\.isSelected), "Filing preserves the selection")
        collection.tap()
        app.buttons["New Collection…"].firstMatch.tap()
        let sheet = app.navigationBars["New Collection"]
        XCTAssertTrue(sheet.waitForExistence(timeout: 5))
        XCTAssertFalse(sheet.buttons["Create"].isEnabled)
        capture(app, "New collection sheet at \(size)")
        let name = app.textFields["Name"]
        name.tap(); name.typeText("Selected Prints")
        sheet.buttons["Create"].tap()
        XCTAssertTrue(sheet.waitForNonExistence(timeout: 5))
        guard waitForMutations(machine, count: 2) else { XCTFail("New collection filing was not delivered"); return }
        try checkMutation(machine.libraryMutationRequests[1], name: "Selected Prints")
        collection.tap()
        XCTAssertTrue(app.buttons["Selected Prints"].firstMatch.waitForExistence(timeout: 10), "The outbox refresh publishes the new shelf")
        app.buttons["New Collection…"].firstMatch.tap()
        XCTAssertTrue(sheet.waitForExistence(timeout: 5))
        sheet.buttons["Cancel"].tap()
        XCTAssertTrue(sheet.waitForNonExistence(timeout: 5))
        XCTAssertEqual(machine.libraryMutationRequests.count, 2, "Cancel makes no edit")
        // The created shelf contains exactly the selected prints after the server reread.
        if !app.buttons["Done"].exists { app.buttons["OverflowBarButtonItem"].tap() }
        app.buttons["Done"].firstMatch.tap()
        app.chooseLibraryShelf("Selected Prints")
        XCTAssertTrue(tiles.allSatisfy { $0.waitForExistence(timeout: 5) })
        XCTAssertFalse(app.buttons.matching(NSPredicate(format: "label CONTAINS 'Fixture 0,'")).firstMatch.exists)
        if !app.buttons["Select"].exists { app.buttons["OverflowBarButtonItem"].tap() }
        app.buttons["Select"].firstMatch.tap()
        for tile in tiles { tile.tap() }
        collection.tap()
        app.buttons["Remove from Collection"].firstMatch.tap()
        app.buttons["Selected Prints"].firstMatch.tap()
        guard waitForMutations(machine, count: 3) else { XCTFail("Collection removal was not delivered"); return }
        let removal = try XCTUnwrap(JSONSerialization.jsonObject(with: machine.libraryMutationRequests[2]) as? [String: Any])
        XCTAssertEqual(removal["remove_from_collection_slug"] as? String, "selected-prints")
        XCTAssertEqual(Set(try XCTUnwrap(removal["filenames"] as? [String])), ["fixture-1.png", "fixture-2.png"])
        XCTAssertTrue(tiles[0].waitForNonExistence(timeout: 10))
        capture(app, "Collection removal at \(size)")
    }

    @MainActor private func capture(_ app: XCUIApplication, _ name: String) {
        let shot = XCTAttachment(screenshot: app.screenshot())
        shot.name = name
        shot.lifetime = .keepAlways
        add(shot)
    }

    private func waitForMutations(_ machine: FixtureMachine, count: Int) -> Bool {
        let expectation = XCTNSPredicateExpectation(predicate: NSPredicate { _, _ in
            machine.libraryMutationRequests.count >= count
        }, object: nil)
        return XCTWaiter.wait(for: [expectation], timeout: 10) == .completed
    }

    private func checkMutation(_ data: Data, name: String) throws {
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(Set(try XCTUnwrap(object["filenames"] as? [String])), ["fixture-1.png", "fixture-2.png"])
        XCTAssertEqual((object["add_to_collection"] as? [String: String])?["name"], name)
        XCTAssertNotNil(object["operation_id"])
    }
}
