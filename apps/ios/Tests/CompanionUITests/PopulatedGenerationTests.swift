import XCTest

/// Populated screens need populated regression tests: first-run empty states
/// cannot reveal compressed option controls or a misleading model search.
final class PopulatedGenerationTests: XCTestCase {
    @MainActor func testOptionsAndModelSearchWithAnInstalledModel() async throws {
        continueAfterFailure = false
        let machine = try FixtureMachine()
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
        let name = app.textFields["machine-name"]
        name.tap()
        name.typeText("Fixture Machine")
        app.buttons["Add"].firstMatch.tap()
        XCTAssertTrue(app.navigateToDestination("Generate", shortcut: "1"))
        let chooser = app.buttons["choose-model"]
        reveal(chooser, in: app)
        chooser.tap()
        let model = app.buttons["model-flux-dev:q4"]
        XCTAssertTrue(model.waitForExistence(timeout: 10))
        model.tap()

        app.terminate()
        app.launchArguments = ["-UIPreferredContentSizeCategoryName", "UICTContentSizeCategoryAccessibilityXXXL"]
        app.launch()
        XCTAssertTrue(app.navigateToDestination("Generate", shortcut: "1"))
        let options = app.buttons["Options"].firstMatch
        reveal(options, in: app)
        options.tap()
        XCTAssertTrue(app.navigationBars["More Options"].waitForExistence(timeout: 5))
        for identifier in ["options-shape", "options-steps", "options-batch"] {
            let control = app.buttons[identifier]
            reveal(control, in: app)
            XCTAssertLessThan(control.frame.height, app.frame.height * 0.45,
                              "An option must remain readable words, not a screen-high column of letters")
            XCTAssertGreaterThanOrEqual(control.frame.minX, 0)
            XCTAssertLessThanOrEqual(control.frame.maxX, app.frame.maxX)
        }
        XCTAssertFalse(app.buttons["More Options"].exists, "The sheet must not offer an inert button to reopen itself")
        attach(app)
        app.buttons["Done"].firstMatch.tap()
        reveal(chooser, in: app)
        chooser.tap()
        let search = app.searchFields.firstMatch
        XCTAssertTrue(search.waitForExistence(timeout: 5))
        search.tap()
        search.typeText("zzzznomodel")
        XCTAssertTrue(app.staticTexts["No matching models"].waitForExistence(timeout: 5))
        attach(app)

        // Remove only this test's local pairing; no remote mutation is possible.
        app.terminate()
        app.launchArguments = ["-UIPreferredContentSizeCategoryName", "UICTContentSizeCategoryL"]
        app.launch()
        XCTAssertTrue(app.navigateToDestination("Machines", shortcut: "5"))
        let card = app.buttons.matching(NSPredicate(format: "label CONTAINS %@", "127.0.0.1:\(port)")).firstMatch
        reveal(card, in: app)
        card.press(forDuration: 1)
        app.buttons["Remove…"].firstMatch.tap()
        app.buttons["Remove"].firstMatch.tap()
    }

    @MainActor private func reveal(_ element: XCUIElement, in app: XCUIApplication) {
        for _ in 0..<8 where !element.isHittable {
            if element.exists, element.frame.minY < app.frame.height * 0.25 { app.swipeDown() }
            else { app.swipeUp() }
        }
        XCTAssertTrue(element.isHittable)
    }

    @MainActor private func attach(_ app: XCUIApplication) {
        let screenshot = XCTAttachment(screenshot: app.screenshot())
        screenshot.lifetime = .keepAlways
        add(screenshot)
    }
}
