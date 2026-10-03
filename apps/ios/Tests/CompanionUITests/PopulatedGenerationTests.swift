import XCTest

/// Populated screens need populated regression tests: first-run empty states
/// cannot reveal compressed option controls or a misleading model search.
final class PopulatedGenerationTests: XCTestCase {
    override func setUp() {
        super.setUp()
        acceptCompanionPermissions()
    }

    @MainActor func testOptionsAndModelSearchWithAnInstalledModel() async throws {
        continueAfterFailure = false
        let machine = try FixtureMachine()
        let port = try await machine.start()
        let app = XCUIApplication()
        defer { app.terminate(); machine.stop() }
        cleanUpFixture(machine, port: port, app: app)
        app.launchArguments = ["-UIPreferredContentSizeCategoryName", "UICTContentSizeCategoryL"]
        app.launch()
        XCTAssertTrue(app.navigateToDestination("Machines", shortcut: "5"))
        app.buttons["Add a Machine"].firstMatch.tap()
        app.buttons.matching(NSPredicate(format: "label BEGINSWITH 'Enter an Address'")).firstMatch.tap()
        let name = app.textFields["machine-name"]
        XCTAssertTrue(name.waitForExistence(timeout: 5))
        name.tap()
        name.typeText("Fixture Machine")
        let address = app.textFields["machine-address"]
        address.tap()
        address.typeText("127.0.0.1:\(port)")
        app.buttons["Add"].firstMatch.tap()
        XCTAssertTrue(app.navigateToDestination("Generate", shortcut: "1"))
        let chooser = app.buttons["choose-model"]
        reveal(chooser, in: app)
        chooser.tap()
        let model = app.buttons["model-flux-dev:q4"]
        XCTAssertTrue(model.waitForExistence(timeout: 10))
        model.tap()
        try checkLandscapeComposer(app)

        app.terminate()
        app.launchArguments = ["-UIPreferredContentSizeCategoryName", "UICTContentSizeCategoryAccessibilityXXXL"]
        app.launch()
        XCTAssertTrue(app.navigateToDestination("Generate", shortcut: "1"))
        reveal(chooser, in: app)
        chooser.tap()
        let search = app.searchFields.firstMatch
        XCTAssertTrue(search.waitForExistence(timeout: 5))
        search.tap()
        search.typeText("zzzznomodel")
        XCTAssertTrue(app.staticTexts["No matching models"].waitForExistence(timeout: 5))
        let closeSearch = app.buttons["model-chooser-close"]
        XCTAssertTrue(closeSearch.isHittable, "Model search must have a visible exit with the keyboard open")
        attach(app)
        closeSearch.tap()
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
    }

    @MainActor private func checkLandscapeComposer(_ app: XCUIApplication) throws {
        XCUIDevice.shared.orientation = .landscapeLeft
        defer { XCUIDevice.shared.orientation = .portrait }
        let rotated = NSPredicate { _, _ in app.frame.width > app.frame.height }
        XCTAssertEqual(XCTWaiter.wait(for: [XCTNSPredicateExpectation(predicate: rotated, object: app)], timeout: 5), .completed)
        // Rotation delivers its new geometry before the composer settles.
        try awaitRotationLayout()
        let composer = app.scrollViews["phone-generate-form"]
        let prompt = app.descendants(matching: .any)["generation-prompt"].firstMatch
        for _ in 0..<6 where !prompt.isHittable { composer.swipeDown() }
        XCTAssertTrue(prompt.isHittable)
        prompt.tap()
        XCTAssertTrue(app.keyboards.firstMatch.waitForExistence(timeout: 5))
        prompt.typeText("Landscape lighthouse")
        XCTAssertTrue((prompt.value as? String)?.contains("Landscape lighthouse") == true)
        app.buttons["Done"].firstMatch.tap()
        XCTAssertTrue(app.keyboards.firstMatch.waitForNonExistence(timeout: 5))
        let submit = app.buttons["submit-generation"]
        for _ in 0..<12 where !submit.isHittable { composer.swipeUp() }
        XCTAssertTrue(submit.isHittable, "Generate must be reachable in landscape; never tap it in UAT")
        let options = app.buttons.matching(NSPredicate(format: "label == 'Options' OR label == 'More Options'")).firstMatch
        reveal(options, in: app)
        XCTAssertTrue(options.isHittable)
        options.tap()
        XCTAssertTrue(app.navigationBars["More Options"].waitForExistence(timeout: 5))
        attach(app)
        app.buttons["Done"].firstMatch.tap()
    }

    @MainActor func testCuratedDiscoveryAndQueuedSourceImage() async throws {
        continueAfterFailure = false
        let machine = try FixtureMachine(queueFixture: true)
        let port = try await machine.start()
        let app = XCUIApplication()
        defer { app.terminate(); machine.stop() }
        cleanUpFixture(machine, port: port, app: app)
        app.launchArguments = ["-UIPreferredContentSizeCategoryName", "UICTContentSizeCategoryL"]
        app.launch()
        XCTAssertTrue(app.navigateToDestination("Machines", shortcut: "5"))
        app.buttons["Add a Machine"].firstMatch.tap()
        app.buttons.matching(NSPredicate(format: "label BEGINSWITH 'Enter an Address'")).firstMatch.tap()
        let name = app.textFields["machine-name"]
        XCTAssertTrue(name.waitForExistence(timeout: 5))
        name.tap(); name.typeText("Media Fixture")
        let address = app.textFields["machine-address"]
        address.tap(); address.typeText("127.0.0.1:\(port)")
        app.buttons["Add"].firstMatch.tap()
        XCTAssertTrue(app.navigateToDestination("Queue", shortcut: "3"))
        let row = app.descendants(matching: .any)["queue-entry-fixture-video"].firstMatch
        XCTAssertTrue(row.waitForExistence(timeout: 10))
        XCTAssertTrue(app.staticTexts["A coastal path at sunrise"].waitForExistence(timeout: 5))
        XCTAssertTrue(app.images["queue-source-fixture-video"].waitForExistence(timeout: 10))
        XCTAssertTrue(app.staticTexts["LTX-2.5 Distilled BF16"].exists)
        XCTAssertLessThanOrEqual(row.frame.width, 860)
        attach(app)
        for category in ["UICTContentSizeCategoryXS", "UICTContentSizeCategoryAccessibilityXXXL"] {
            app.terminate()
            app.launchArguments = ["-UIPreferredContentSizeCategoryName", category]
            app.launch()
            XCTAssertTrue(app.navigateToDestination("Queue", shortcut: "3"))
            XCTAssertTrue(app.images["queue-source-fixture-video"].waitForExistence(timeout: 10))
            XCTAssertTrue(app.staticTexts["A coastal path at sunrise"].exists)
            attach(app)
        }
        app.terminate()
        app.launchArguments = ["-UIPreferredContentSizeCategoryName", "UICTContentSizeCategoryL"]
        app.launch()
        if app.buttons["ToggleSideBar"].exists || app.buttons["Models"].exists {
            XCTAssertTrue(app.navigateToDestination("Models", shortcut: "4"))
            app.buttons["models-pane"].tap()
            app.buttons["Discover"].firstMatch.tap()
        } else {
            XCTAssertTrue(app.navigateToDestination("Machines", shortcut: "5"))
            let card = app.descendants(matching: .any).matching(NSPredicate(format:
                "identifier BEGINSWITH 'machine-card-' AND label CONTAINS %@", "127.0.0.1:\(port)")).firstMatch
            XCTAssertTrue(card.waitForExistence(timeout: 5))
            card.tap()
            app.buttons["Models"].firstMatch.tap()
            app.buttons["models-pane"].tap()
            app.buttons["Discover"].firstMatch.tap()
        }
        let curated = app.descendants(matching: .any)["curated-model-flux-dev:q4"].firstMatch
        XCTAssertTrue(curated.waitForExistence(timeout: 10))
        XCTAssertTrue(app.staticTexts["Hugging Face"].firstMatch.exists)
        let search = app.searchFields.firstMatch
        search.tap(); search.typeText("black-forest")
        XCTAssertTrue(curated.waitForExistence(timeout: 5))
        XCTAssertFalse(app.descendants(matching: .any)["curated-model-ltx-2.5-22b-distilled:bf16"].exists)
        if app.buttons["Cancel"].firstMatch.isHittable { app.buttons["Cancel"].firstMatch.tap() }
        let get = app.buttons["Get FLUX.1 Dev Q4"]
        XCTAssertTrue(get.waitForExistence(timeout: 5))
        for _ in 0..<5 where !get.isHittable { app.swipeUp() }
        get.tap()
        for _ in 0..<50 where machine.installedRequests.isEmpty { try await Task.sleep(for: .milliseconds(100)) }
        XCTAssertEqual(machine.installedRequests, ["flux-dev:q4"], "Curated Get must target one exact manifest checkpoint")
        attach(app)
    }

    @MainActor private func awaitRotationLayout() throws {
        // UIKit's orientation animation is not included in XCTest's app-idle wait.
        RunLoop.current.run(until: Date().addingTimeInterval(1))
    }

    @MainActor private func reveal(_ element: XCUIElement, in app: XCUIApplication) {
        let form = app.scrollViews["phone-generate-form"]
        for _ in 0..<12 {
            if !form.exists {
                if element.isHittable { break }
                app.swipeUp()
                continue
            }
            if app.navigationBars["More Options"].exists {
                if element.isHittable { break }
                app.swipeUp()
                continue
            }
            let top = app.navigationBars.firstMatch.frame.maxY + 12
            let bottom = app.buttons["submit-generation"].frame.minY - 12
            if element.isHittable, element.frame.midY >= top + 20, element.frame.midY <= bottom - 20 { break }
            // Swipe the clear right edge. A centered swipe lands in the
            // horizontal picture wells and leaves the form where it was.
            let down = element.exists && element.frame.midY < top + 20
            // top/bottom are screen coordinates. Adding them to the form's
            // origin put the gesture below its viewport, over fixed chrome.
            let origin = app.coordinate(withNormalizedOffset: .zero)
            let x = form.frame.minX + form.frame.width * 0.9
            let upper = max(top, form.frame.minY) + 20
            let lower = min(bottom, form.frame.maxY) - 20
            XCTAssertGreaterThan(lower, upper, "The form needs a visible scrolling viewport")
            let high = upper + (lower - upper) * 0.2
            let low = upper + (lower - upper) * 0.7
            let start = origin.withOffset(CGVector(dx: x, dy: down ? high : low))
            let end = origin.withOffset(CGVector(dx: x, dy: down ? low : high))
            start.press(forDuration: 0.1, thenDragTo: end,
                        withVelocity: .slow, thenHoldForDuration: 0.1)
        }
        XCTAssertTrue(element.isHittable, "The control must be reachable within the Generate form: \(app.debugDescription)")
    }

    @MainActor private func attach(_ app: XCUIApplication) {
        let screenshot = XCTAttachment(screenshot: app.screenshot())
        screenshot.lifetime = .keepAlways
        add(screenshot)
    }
}
