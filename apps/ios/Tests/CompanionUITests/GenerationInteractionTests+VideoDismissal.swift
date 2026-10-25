import XCTest

extension GenerationInteractionTests {
    @MainActor func testCompletedVideoDismissIsIndependentAndSurvivesRelaunch() async throws {
        let (machine, app) = try await durationQueueApp()
        let dismiss = app.buttons.matching(NSPredicate(format: "label BEGINSWITH 'Dismiss'")).firstMatch
        XCTAssertTrue(dismiss.waitForExistence(timeout: 15), app.debugDescription)
        let checks = machine.requestLog().filter { $0 == "GET /api/video-jobs/fixture-completed" }.count
        // Delete job… lives in the row's contextual menu, like Cancel Job.
        app.buttons["video-queue-open-fixture-completed"].press(forDuration: 1.2)
        XCTAssertTrue(app.buttons["Delete job…"].waitForExistence(timeout: 5), app.debugDescription)
        app.buttons["Delete job…"].tap()
        XCTAssertTrue(app.buttons["Delete recovery data"].waitForExistence(timeout: 5), app.debugDescription + "\nRequests: " + machine.requestLog().joined(separator: "\n"))
        XCTAssertTrue(app.navigationBars["Queue"].exists)
        XCTAssertFalse(machine.requestLog().contains { $0.hasPrefix("DELETE ") })
        let confirmation = XCTAttachment(screenshot: app.screenshot())
        confirmation.name = "Delete job confirmation does not dismiss the video"
        confirmation.lifetime = .keepAlways; add(confirmation)
        if app.buttons["Cancel"].exists {
            app.buttons["Cancel"].tap()
        } else {
            // Compact iOS 27 popovers cancel with an outside tap.
            app.navigationBars["Queue"].staticTexts["Queue"].tap()
        }
        XCTAssertTrue(app.buttons["Delete recovery data"].waitForNonExistence(timeout: 3))
        XCTAssertEqual(machine.requestLog().filter { $0 == "GET /api/video-jobs/fixture-completed" }.count, checks,
                       "opening Delete confirmation must not trigger another owner re-check")
        dismiss.tap()
        XCTAssertTrue(dismiss.waitForNonExistence(timeout: 5), "one real tap removes the completed Queue row")
        XCTAssertFalse(app.buttons["Delete recovery data"].exists, "Dismiss must not activate Delete job")
        XCTAssertTrue(app.navigationBars["Queue"].exists, "Dismiss must not activate Open in Library")
        XCTAssertFalse(machine.requestLog().contains { $0.hasPrefix("DELETE ") || $0.hasPrefix("POST /api/video-jobs") })
        app.terminate(); app.launch()
        XCTAssertTrue(app.navigateToDestination("Queue", shortcut: "3"))
        XCTAssertFalse(app.buttons.matching(NSPredicate(format: "label BEGINSWITH 'Dismiss'")).firstMatch.waitForExistence(timeout: 3))
        let attachment = XCTAttachment(screenshot: app.screenshot())
        attachment.name = "Completed duration parent dismissed from Queue"
        attachment.lifetime = .keepAlways; add(attachment)
    }
    @MainActor func testCompletedParentDoesNotLinkToDeletedLibraryMovie() async throws {
        let (_, app) = try await durationQueueApp()
        let missing = app.staticTexts["Saved video is no longer in Library."]
        XCTAssertTrue(missing.waitForExistence(timeout: 10), app.debugDescription)
        XCTAssertFalse(app.buttons["Open in Library"].exists)
        XCTAssertTrue(app.buttons.matching(NSPredicate(format: "label BEGINSWITH 'Dismiss'")).firstMatch.exists)
    }

    @MainActor private func durationQueueApp() async throws -> (FixtureMachine, XCUIApplication) {
        continueAfterFailure = false
        let machine = try FixtureMachine(durationQueueFixture: true)
        let port = try await machine.start(), app = XCUIApplication()
        cleanUpFixture(machine, port: port, app: app)
        app.launch()
        XCTAssertTrue(app.navigateToDestination("Machines", shortcut: "5"))
        app.buttons["Add a Machine"].firstMatch.tap()
        app.buttons.matching(NSPredicate(format: "label BEGINSWITH 'Enter an Address'")).firstMatch.tap()
        app.chooseRegularConnection()
        app.buttons.matching(NSPredicate(format: "label BEGINSWITH 'Connection,'")).firstMatch.tap()
        app.buttons.matching(NSPredicate(format: "label BEGINSWITH 'Regular'")).firstMatch.tap()
        let name = app.textFields["machine-name"]
        XCTAssertTrue(name.waitForExistence(timeout: 5)); name.tap(); name.typeText("Duration Queue UAT")
        let address = app.textFields["machine-address"]; address.tap(); address.typeText("127.0.0.1:\(port)")
        app.buttons["Add"].firstMatch.tap()
        XCTAssertTrue(app.navigateToDestination("Queue", shortcut: "3"))
        return (machine, app)
    }

}
