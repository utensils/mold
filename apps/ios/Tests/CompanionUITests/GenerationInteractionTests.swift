import XCTest

/// Exercise a saved-machine screen, not just first-run empty states.
final class GenerationInteractionTests: XCTestCase {
    @MainActor private func launch(size: String = "UICTContentSizeCategoryL") -> XCUIApplication {
        continueAfterFailure = false
        let app = XCUIApplication()
        app.launchArguments = ["-UIPreferredContentSizeCategoryName", "UICTContentSizeCategoryL"]
        app.launch()
        app.buttons["Generate"].firstMatch.tap()
        if app.buttons["Add a Machine…"].firstMatch.exists {
            app.buttons["Add a Machine…"].firstMatch.tap()
            app.buttons.matching(NSPredicate(format: "label BEGINSWITH 'Enter an Address'")).firstMatch.tap()
            let address = app.textFields["Address"]
            XCTAssertTrue(address.waitForExistence(timeout: 5))
            address.tap()
            address.typeText("127.0.0.1:9")
            app.buttons["Add"].firstMatch.tap()
            app.buttons["Generate"].firstMatch.tap()
        }
        if size != "UICTContentSizeCategoryL" {
            app.terminate()
            app.launchArguments = ["-UIPreferredContentSizeCategoryName", size]
            app.launch()
            app.buttons["Generate"].firstMatch.tap()
        }
        return app
    }

    @MainActor func testAccessibilityComposerAndMachineCardsFitTheScreen() throws {
        let app = launch(size: "UICTContentSizeCategoryAccessibilityXXXL")
        let composer = app.descendants(matching: .any)["bottom-chrome"].firstMatch
        XCTAssertTrue(composer.waitForExistence(timeout: 5))
        XCTAssertLessThanOrEqual(composer.frame.height, app.frame.height * 0.55)
        capture(app)
        app.buttons["Machines"].firstMatch.tap()
        let card = app.descendants(matching: .any).matching(NSPredicate(format: "identifier BEGINSWITH 'machine-card-'")).firstMatch
        XCTAssertTrue(card.waitForExistence(timeout: 5))
        XCTAssertGreaterThanOrEqual(card.frame.minX, app.frame.minX)
        XCTAssertLessThanOrEqual(card.frame.maxX, app.frame.maxX)
        capture(app)
    }

    @MainActor func testPromptAcceptsTypingAndKeepsKeyboard() throws {
        let app = launch()
        let prompt = app.descendants(matching: .any)["generation-prompt"].firstMatch
        XCTAssertTrue(prompt.waitForExistence(timeout: 5))
        prompt.tap()
        XCTAssertTrue(app.keyboards.firstMatch.waitForExistence(timeout: 5))
        prompt.typeText("A lighthouse at dusk")
        XCTAssertTrue((prompt.value as? String)?.contains("A lighthouse at dusk") == true)
        XCTAssertTrue(app.keyboards.firstMatch.exists, "Typing must not replace the focused composer")
        capture(app)
    }

    @MainActor func testModelChooserAndDownloadRoute() throws {
        let app = launch()
        let chooser = app.buttons["choose-model"]
        for _ in 0..<4 where !chooser.isHittable { app.descendants(matching: .any)["bottom-chrome"].firstMatch.swipeUp() }
        XCTAssertTrue(chooser.isHittable)
        chooser.tap()
        XCTAssertTrue(app.navigationBars["Choose a Model"].waitForExistence(timeout: 5))
        capture(app)
        let more = app.buttons["Get More Models…"].firstMatch
        for _ in 0..<8 where !more.isHittable { app.swipeUp() }
        XCTAssertTrue(more.isHittable)
        more.tap()
        XCTAssertTrue(app.navigationBars["Models"].waitForExistence(timeout: 5))
        capture(app)
    }

    @MainActor private func capture(_ app: XCUIApplication) {
        let shot = XCTAttachment(screenshot: app.screenshot())
        shot.lifetime = .keepAlways
        add(shot)
    }
}
