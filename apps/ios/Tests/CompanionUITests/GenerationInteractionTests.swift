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
            let address = app.textFields["machine-address"]
            XCTAssertTrue(address.waitForExistence(timeout: 5))
            address.tap()
            address.typeText("127.0.0.1:9")
            let name = app.textFields["machine-name"]
            name.tap()
            name.typeText("UAT Machine")
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
        XCTAssertLessThanOrEqual(composer.frame.height, app.frame.height * 0.9)
        capture(app)
        XCTAssertTrue(app.navigateToDestination("Machines", shortcut: "5"))
        let card = app.descendants(matching: .any).matching(NSPredicate(format: "identifier BEGINSWITH 'machine-card-'")).firstMatch
        XCTAssertTrue(card.waitForExistence(timeout: 5))
        XCTAssertGreaterThanOrEqual(card.frame.minX, app.frame.minX)
        XCTAssertLessThanOrEqual(card.frame.maxX, app.frame.maxX)
        capture(app)
    }

    @MainActor func testOfflineQueueExplanationScrollsAtLargestText() throws {
        let app = launch(size: "UICTContentSizeCategoryAccessibilityXXXL")
        XCTAssertTrue(app.navigateToDestination("Queue", shortcut: "3"))
        let message = app.staticTexts["Some machines could not provide their queues. Check Machines to reconnect, then pull to refresh."]
        XCTAssertTrue(message.waitForExistence(timeout: 5))
        let action = app.buttons["Check Machines"]
        for _ in 0..<8 where message.frame.maxY > action.frame.minY {
            // A full-screen swipe starts on the pinned button on small phones.
            // Drag the visible explanation instead, as a person would.
            let origin = app.coordinate(withNormalizedOffset: .zero)
            let start = origin.withOffset(CGVector(dx: app.frame.midX, dy: action.frame.minY - 24))
            let end = origin.withOffset(CGVector(dx: app.frame.midX, dy: app.navigationBars.firstMatch.frame.maxY + 24))
            start.press(forDuration: 0.05, thenDragTo: end)
        }
        capture(app)
        XCTAssertLessThanOrEqual(message.frame.maxY, action.frame.minY,
                                 "The final line must scroll above the pinned action")
        XCTAssertTrue(action.isHittable)
    }

    @MainActor func testFloatingBarKeepsModelsInSidebar() throws {
        let app = launch(size: "UICTContentSizeCategoryAccessibilityXXXL")
        guard app.buttons["ToggleSideBar"].exists else { return }
        let favourites = app.descendants(matching: .any)["Favourites"].firstMatch
        if favourites.exists, favourites.isHittable { app.buttons["ToggleSideBar"].tap() }
        XCTAssertFalse(app.buttons["Models"].firstMatch.exists)
        XCTAssertFalse(app.buttons["Next Page"].firstMatch.exists)
        XCTAssertTrue(app.navigateToDestination("Machines", shortcut: "5"))
        XCTAssertTrue(app.navigateToDestination("Models", shortcut: "4"))
    }

    @MainActor func testPromptAcceptsTypingAndKeepsKeyboard() throws {
        try checkPrompt(size: "UICTContentSizeCategoryL")
    }

    @MainActor func testLargestTextPromptKeepsKeyboard() throws {
        try checkPrompt(size: "UICTContentSizeCategoryAccessibilityXXXL")
    }

    @MainActor private func checkPrompt(size: String) throws {
        let app = launch(size: size)
        let prompt = app.descendants(matching: .any)["generation-prompt"].firstMatch
        XCTAssertTrue(prompt.waitForExistence(timeout: 5))
        prompt.tap()
        XCTAssertTrue(app.keyboards.firstMatch.waitForExistence(timeout: 5))
        prompt.typeText("A lighthouse at dusk")
        XCTAssertTrue((prompt.value as? String)?.contains("A lighthouse at dusk") == true)
        XCTAssertTrue(app.keyboards.firstMatch.exists, "Typing must not replace the focused composer")
        capture(app)
    }

    @MainActor func testSettingsPresentationAndAddMachineRoute() throws {
        let app = launch()
        XCTAssertTrue(app.navigateToDestination("Machines", shortcut: "5"))
        app.buttons["Settings"].firstMatch.tap()
        XCTAssertTrue(app.buttons["Done"].firstMatch.waitForExistence(timeout: 5))
        app.buttons["Done"].firstMatch.tap()
        XCTAssertTrue(app.navigationBars["Machines"].waitForExistence(timeout: 5))
        app.buttons["Settings"].firstMatch.tap()
        app.buttons["Add a Machine…"].firstMatch.tap()
        XCTAssertTrue(app.buttons["Cancel"].firstMatch.waitForExistence(timeout: 5))
        app.buttons["Cancel"].firstMatch.tap()
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

@MainActor extension XCUIApplication {
    /// iPadOS can page the floating bar on the first tap at large text sizes.
    /// Require the destination, then retry the now-visible tab if necessary.
    func navigateToDestination(_ title: String, shortcut: String) -> Bool {
        let bar = navigationBars[title == "Library" ? "All Prints" : title]
        for _ in 0..<3 {
            let tab = buttons[title].firstMatch
            if tab.waitForExistence(timeout: 2), tab.isHittable {
                tab.tap()
            } else if buttons["Next Page"].firstMatch.exists, buttons["Next Page"].firstMatch.isHittable {
                buttons["Next Page"].firstMatch.tap()
                continue
            } else {
                typeKey(shortcut, modifierFlags: .command)
            }
            if bar.waitForExistence(timeout: 2) { return true }
        }
        return bar.exists
    }
}
