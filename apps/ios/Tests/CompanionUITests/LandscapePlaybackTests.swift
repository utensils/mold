import XCTest

final class LandscapePlaybackTests: XCTestCase {
    override func setUp() {
        super.setUp()
        acceptCompanionPermissions()
        continueAfterFailure = false
    }

    @MainActor func testClipFillsBothLandscapeOrientationsAndRestoresPortrait() async throws {
        XCUIDevice.shared.orientation = .portrait
        defer { XCUIDevice.shared.orientation = .portrait }
        let identity = UUID().uuidString
        let machine = try FixtureMachine(landscapePlaybackFixture: true, galleryPrints: 3, galleryID: identity, mixedMedia: true)
        let port = try await machine.start()
        let app = XCUIApplication()
        defer { app.terminate(); machine.stop() }
        cleanUpFixture(machine, port: port, app: app)
        app.launchArguments += ["-videoPlaybackAutoplay", "YES", "-videoPlaybackRepeat", "YES"]
        app.launch()
        XCTAssertTrue(app.navigateToDestination("Machines", shortcut: "5"))
        app.buttons["Add a Machine"].firstMatch.tap()
        app.buttons.matching(NSPredicate(format: "label BEGINSWITH 'Enter an Address'")).firstMatch.tap()
        let address = app.textFields["machine-address"]
        XCTAssertTrue(address.waitForExistence(timeout: 5))
        address.tap(); address.typeText("127.0.0.1:\(port)")
        app.buttons["Add"].firstMatch.tap()
        XCTAssertTrue(app.navigateToDestination("Library", shortcut: "2"))
        if !app.navigationBars["All Prints"].exists { app.chooseLibraryShelf("All Prints") }
        let tile = app.buttons.matching(NSPredicate(format: "label BEGINSWITH %@", "Photos-\(identity) 1,")).firstMatch
        XCTAssertTrue(tile.waitForExistence(timeout: 10))
        tile.tap()
        let more = app.buttons["More"].firstMatch
        XCTAssertTrue(more.waitForExistence(timeout: 5))
        let player = app.otherElements["viewer-print-fixture-\(identity)-1.mp4"].firstMatch
        guard player.waitForExistence(timeout: 10) else { XCTFail(app.debugDescription); return }
        if !app.buttons["Pause"].firstMatch.exists { player.coordinate(withNormalizedOffset: CGVector(dx: 0.8, dy: 0.5)).tap() }
        guard app.buttons["Pause"].firstMatch.waitForExistence(timeout: 5) else { XCTFail("A decoded video must be playing: \(app.debugDescription)"); return }
        let close = app.buttons["landscape-playback-close"]
        XCTAssertFalse(close.exists)
        evidence(app, "Portrait playback before rotation")
        for orientation in [UIDeviceOrientation.landscapeLeft, .landscapeRight] {
            XCUIDevice.shared.orientation = orientation
            XCTAssertTrue(close.waitForExistence(timeout: 5))
            let fillsWindow = NSPredicate { _, _ in
                app.frame.width > app.frame.height && abs(player.frame.height - app.frame.height) < 2
                    && abs(player.frame.width - app.frame.width) < 2
            }
            let settled = expectation(for: fillsWindow, evaluatedWith: app)
            await fulfillment(of: [settled], timeout: 5)
            XCTAssertTrue(close.isHittable)
            XCTAssertGreaterThanOrEqual(close.frame.width, 44)
            XCTAssertGreaterThanOrEqual(close.frame.height, 44)
            XCTAssertTrue(app.frame.contains(close.frame), "Close stays within the display")
            XCTAssertFalse(more.exists, "Gallery bars must not take space from landscape playback")
            if !app.buttons["Pause"].firstMatch.exists { player.coordinate(withNormalizedOffset: CGVector(dx: 0.8, dy: 0.5)).tap() }
            XCTAssertTrue(app.buttons["Pause"].firstMatch.waitForExistence(timeout: 5), app.debugDescription)
            let closeFrame = close.frame
            for control in app.buttons.allElementsBoundByIndex where control.identifier != "landscape-playback-close" && control.isHittable {
                XCTAssertFalse(closeFrame.intersects(control.frame), "Close must not overlap \(control.label)")
            }
            evidence(app, "Landscape playback \(orientation.rawValue)")
        }
        XCUIDevice.shared.orientation = .portrait
        XCTAssertTrue(close.waitForNonExistence(timeout: 5))
        XCTAssertTrue(more.waitForExistence(timeout: 5))
        evidence(app, "Portrait gallery actions restored")
        XCUIDevice.shared.orientation = .landscapeLeft
        XCTAssertTrue(close.waitForExistence(timeout: 5))
        close.tap()
        XCTAssertTrue(app.navigationBars["All Prints"].waitForExistence(timeout: 5))
        XCTAssertTrue(machine.generationRequests.isEmpty)
    }

    @MainActor private func evidence(_ app: XCUIApplication, _ name: String) {
        let attachment = XCTAttachment(screenshot: XCUIScreen.main.screenshot())
        attachment.name = name
        attachment.lifetime = .keepAlways
        add(attachment)
    }
}
