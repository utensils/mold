import XCTest

/// Real native interaction against a loopback machine whose collection
/// mutations stay in the fixture process. Run on iPhone and iPad.
final class HiddenCollectionTests: XCTestCase {
    override func setUp() {
        super.setUp()
        acceptCompanionPermissions()
    }

    @MainActor func testHideShowAndOpenHiddenCollection() async throws {
        continueAfterFailure = false
        let machine = try FixtureMachine(galleryPrints: 2, collectionFixture: true)
        let port = try await machine.start()
        defer { machine.stop() }
        let app = XCUIApplication()
        app.launchArguments = ["-UIPreferredContentSizeCategoryName", "UICTContentSizeCategoryL"]
        app.launch()
        XCTAssertTrue(app.navigateToDestination("Machines", shortcut: "5"))
        app.buttons["Add a Machine"].firstMatch.tap()
        app.buttons.matching(NSPredicate(format: "label BEGINSWITH 'Enter an Address'")).firstMatch.tap()
        let address = app.textFields["machine-address"]
        XCTAssertTrue(address.waitForExistence(timeout: 5))
        address.tap()
        address.typeText("127.0.0.1:\(port)")
        app.buttons["Add"].firstMatch.tap()
        XCTAssertTrue(app.navigateToDestination("Library", shortcut: "2"))
        let hiddenPrint = app.buttons.matching(NSPredicate(format: "label BEGINSWITH 'Fixture 0,'")).firstMatch
        let normalPrint = app.buttons.matching(NSPredicate(format: "label BEGINSWITH 'Fixture 1,'")).firstMatch
        XCTAssertTrue(hiddenPrint.waitForExistence(timeout: 10))
        XCTAssertTrue(normalPrint.exists)
        manage(app)
        let toggle = app.switches["Hide from All Prints"].firstMatch
        XCTAssertTrue(toggle.waitForExistence(timeout: 5))
        toggle.coordinate(withNormalizedOffset: CGVector(dx: 0.95, dy: 0.5)).tap()
        XCTAssertTrue(waitForValue(toggle, "1"))
        capture(app, "Hidden collection management")
        app.buttons["Done"].firstMatch.tap()
        XCTAssertTrue(hiddenPrint.waitForNonExistence(timeout: 10))
        XCTAssertTrue(normalPrint.exists)
        capture(app, "All Prints excludes hidden member")

        manage(app)
        app.buttons["UAT Drafts"].firstMatch.tap()
        XCTAssertTrue(app.navigationBars["UAT Drafts"].waitForExistence(timeout: 5))
        XCTAssertTrue(hiddenPrint.waitForExistence(timeout: 5))
        XCTAssertFalse(normalPrint.exists)
        capture(app, "Explicit hidden shelf remains accessible")
        manage(app)
        toggle.coordinate(withNormalizedOffset: CGVector(dx: 0.95, dy: 0.5)).tap()
        XCTAssertTrue(waitForValue(toggle, "0"))
        app.buttons["Done"].firstMatch.tap()
        // Selecting the main Library tab returns the general grid on iPad;
        // phone keeps the chosen shelf, so use its shelf picker if present.
        if app.buttons["library-collections"].exists {
            app.buttons["library-collections"].tap()
            app.buttons["All Prints"].firstMatch.tap()
        } else {
            XCTAssertTrue(app.navigateToDestination("Library", shortcut: "2"))
        }
        XCTAssertTrue(hiddenPrint.waitForExistence(timeout: 5))
        XCTAssertTrue(normalPrint.waitForExistence(timeout: 5))
        capture(app, "Showing collection restores All Prints")

        XCTAssertTrue(app.navigateToDestination("Machines", shortcut: "5"))
        let card = app.descendants(matching: .any).matching(NSPredicate(
            format: "identifier BEGINSWITH 'machine-card-' AND label CONTAINS %@", "127.0.0.1:\(port)")).firstMatch
        XCTAssertTrue(card.waitForExistence(timeout: 5))
        for _ in 0..<8 where !card.isHittable { app.swipeUp() }
        card.press(forDuration: 1)
        app.buttons["Remove…"].firstMatch.tap()
        app.buttons["Remove"].firstMatch.tap()
        app.terminate()
    }

    @MainActor private func manage(_ app: XCUIApplication) {
        app.buttons["View Options"].firstMatch.tap()
        app.buttons["Manage Collections…"].firstMatch.tap()
        XCTAssertTrue(app.navigationBars["Collections"].waitForExistence(timeout: 5))
    }

    @MainActor private func waitForValue(_ element: XCUIElement, _ value: String) -> Bool {
        XCTWaiter.wait(for: [XCTNSPredicateExpectation(predicate: NSPredicate(format: "value == %@", value),
                                                      object: element)], timeout: 10) == .completed
    }

    @MainActor private func capture(_ app: XCUIApplication, _ name: String) {
        let attachment = XCTAttachment(screenshot: app.screenshot())
        attachment.name = name
        attachment.lifetime = .keepAlways
        add(attachment)
    }
}
