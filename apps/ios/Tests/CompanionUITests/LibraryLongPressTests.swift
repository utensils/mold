import XCTest

/// Context menus and drag previews are separately hosted by UIKit. Exercise
/// their real long-press boundary with a populated library, not just a tap.
final class LibraryLongPressTests: XCTestCase {
    override func setUp() { super.setUp(); acceptCompanionPermissions() }

    @MainActor func testLibraryLongPressOpensMenuAndRemainsUsable() async throws {
        continueAfterFailure = false
        let app = try await populatedLibrary()
        let print = fixturePrint(in: app)
        XCTAssertTrue(print.waitForExistence(timeout: 10))
        print.press(forDuration: 1)
        XCTAssertTrue(app.buttons["Copy"].firstMatch.waitForExistence(timeout: 5))
        XCTAssertEqual(app.state, .runningForeground)
        attach(app, name: "Library long-press context menu")
        app.buttons["Use These Settings"].firstMatch.tap()
        XCTAssertTrue(app.navigateToDestination("Generate", shortcut: "1"))
        XCTAssertTrue(app.navigateToDestination("Library", shortcut: "2"))
        print.tap()
        XCTAssertTrue(app.buttons["Info"].firstMatch.waitForExistence(timeout: 5))
        app.navigationBars.buttons["BackButton"].tap()
        XCTAssertTrue(print.isHittable)
    }

    @MainActor func testSourceLibraryLongPressAndSelectionRemainUsable() async throws {
        continueAfterFailure = false
        let app = try await populatedLibrary()
        XCTAssertTrue(app.navigateToDestination("Generate", shortcut: "1"))
        app.buttons["choose-model"].tap()
        let model = app.buttons["model-flux-dev:q4"]
        XCTAssertTrue(model.waitForExistence(timeout: 5))
        model.tap()
        let source = app.buttons["Start from, empty"].firstMatch
        let form = app.scrollViews["phone-generate-form"]
        for _ in 0..<6 where !source.isHittable { form.swipeUp() }
        XCTAssertTrue(source.isHittable)
        source.tap()
        let chooseLibrary = app.buttons["Choose from Library…"].firstMatch
        XCTAssertTrue(chooseLibrary.waitForExistence(timeout: 5))
        chooseLibrary.tap()
        XCTAssertTrue(app.navigationBars["Choose from Library"].waitForExistence(timeout: 5))
        let picked = fixturePrint(in: app)
        XCTAssertTrue(picked.waitForExistence(timeout: 5))
        picked.press(forDuration: 1)
        XCTAssertEqual(app.state, .runningForeground)
        // A source tile is a picker button, not an organizing context menu.
        if app.navigationBars["Choose from Library"].exists { picked.tap() }
        XCTAssertTrue(app.navigationBars["Choose from Library"].waitForNonExistence(timeout: 10))
        XCTAssertTrue(app.buttons["Start from"].firstMatch.waitForExistence(timeout: 5))
        attach(app, name: "Source selected after long press")
    }

    @MainActor private func populatedLibrary() async throws -> XCUIApplication {
        let machine = try FixtureMachine(galleryPrints: 6, collectionFixture: true)
        let port = try await machine.start()
        let app = XCUIApplication()
        cleanUpFixture(machine, port: port, app: app)
        app.launch()
        XCTAssertTrue(app.navigateToDestination("Machines", shortcut: "5"))
        app.buttons["Add a Machine"].firstMatch.tap()
        app.buttons.matching(NSPredicate(format: "label BEGINSWITH 'Enter an Address'")).firstMatch.tap()
        let address = app.textFields["machine-address"]
        XCTAssertTrue(address.waitForExistence(timeout: 5))
        address.tap(); address.typeText("127.0.0.1:\(port)")
        app.buttons["Add"].firstMatch.tap()
        XCTAssertTrue(app.navigateToDestination("Library", shortcut: "2"))
        XCTAssertTrue(fixturePrint(in: app).waitForExistence(timeout: 10))
        return app
    }

    @MainActor private func fixturePrint(in app: XCUIApplication) -> XCUIElement {
        app.buttons.matching(NSPredicate(format: "label BEGINSWITH 'Fixture 0,'")).firstMatch
    }

    @MainActor private func attach(_ app: XCUIApplication, name: String) {
        let screenshot = XCTAttachment(screenshot: app.screenshot())
        screenshot.name = name
        screenshot.lifetime = .keepAlways
        add(screenshot)
    }
}
