import CoreGraphics
import XCTest

/// Actual AVKit controls and decoded screen pixels, exercised headlessly on iPhone and iPad.
final class PlaybackEndpointTests: XCTestCase {
    override func setUp() { super.setUp(); acceptCompanionPermissions(); continueAfterFailure = false }

    @MainActor func testNaturalEndScrubAndReplayKeepTheLastDecodedFrame() async throws {
        let (app, machine, player) = try await fixture(repeating: false)
        try await Task.sleep(for: .seconds(3))
        try await expectColor(.blue, in: player, name: "Natural end holds final blue sample")
        revealControls(app, player)
        let play = app.buttons["Play"].firstMatch
        XCTAssertTrue(play.waitForExistence(timeout: 5), app.debugDescription)
        play.tap()
        try await expectColor(.red, in: player, name: "Play restarts at first red sample")
        revealControls(app, player)
        let pause = app.buttons["Pause"].firstMatch
        if pause.waitForExistence(timeout: 3) { pause.tap() }
        let scrubber = app.sliders.firstMatch
        XCTAssertTrue(scrubber.waitForExistence(timeout: 5), app.debugDescription)
        scrubber.adjust(toNormalizedSliderPosition: 1)
        try await expectColor(.blue, in: player, name: "Full-end scrub holds final blue sample")
        XCTAssertTrue(machine.generationRequests.isEmpty)
    }

    @MainActor func testRepeatContinuesAfterTheVisualEnd() async throws {
        let (app, machine, player) = try await fixture(repeating: true, autoplay: false)
        revealControls(app, player)
        let scrubber = app.sliders.firstMatch
        XCTAssertTrue(scrubber.waitForExistence(timeout: 5), app.debugDescription)
        scrubber.adjust(toNormalizedSliderPosition: 1)
        try await expectColor(.blue, in: player, name: "Repeat fixture paused at final sample")
        revealControls(app, player)
        let play = app.buttons["Play"].firstMatch
        XCTAssertTrue(play.waitForExistence(timeout: 5))
        play.tap()
        // 3 seconds exceeds this fixture's complete 2.25-second first pass.
        try await Task.sleep(for: .seconds(3))
        try await expectColor(.red, in: player, name: "Repeat is playing the next pass")
        revealControls(app, player)
        XCTAssertTrue(app.buttons["Pause"].firstMatch.waitForExistence(timeout: 3), app.debugDescription)
        XCTAssertTrue(machine.generationRequests.isEmpty)
    }

    @MainActor func testSilentAudioTailBecomesReadyAndHoldsItsFinalVideoFrame() async throws {
        let (app, machine, player) = try await fixture(repeating: false, fileName: "endpoint-silent-tail.mp4")
        try await Task.sleep(for: .seconds(4))
        try await expectColor(.blue, in: player, name: "Silent audio tail completes before final frame hold")
        revealControls(app, player)
        let play = app.buttons["Play"].firstMatch
        XCTAssertTrue(play.waitForExistence(timeout: 5))
        play.tap()
        try await expectColor(.red, in: player, name: "Audio-tail clip replays from its first sample")
        XCTAssertTrue(machine.generationRequests.isEmpty)
    }

    @MainActor private func fixture(repeating: Bool, autoplay: Bool = true, fileName: String = "endpoint-bframes.mp4") async throws -> (XCUIApplication, FixtureMachine, XCUIElement) {
        var root = URL(fileURLWithPath: #filePath)
        while root.pathComponents.count > 1 && !FileManager.default.fileExists(atPath: root.appending(path: "docs/generated").path) { root.deleteLastPathComponent() }
        let bytes = try Data(contentsOf: root.appending(path: "apps/shared/Packages/MoldClient/Tests/MoldClientTests/Fixtures/" + fileName))
        let identity = UUID().uuidString
        let machine = try FixtureMachine(landscapePlaybackFixture: true, galleryPrints: 3, galleryID: identity, mixedMedia: true, playbackData: bytes)
        let port = try await machine.start()
        let app = XCUIApplication()
        cleanUpFixture(machine, port: port, app: app)
        app.launchArguments += ["-videoPlaybackAutoplay", autoplay ? "YES" : "NO", "-videoPlaybackRepeat", repeating ? "YES" : "NO"]
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
        if !app.navigationBars["All Prints"].exists { app.chooseLibraryShelf("All Prints") }
        let tile = app.buttons.matching(NSPredicate(format: "label BEGINSWITH %@", "Photos-\(identity) 1,")).firstMatch
        XCTAssertTrue(tile.waitForExistence(timeout: 10)); tile.tap()
        let player = app.otherElements["viewer-print-fixture-\(identity)-1.mp4"].firstMatch
        XCTAssertTrue(player.waitForExistence(timeout: 10))
        let ready = expectation(for: NSPredicate(format: "value == 'Ready'"), evaluatedWith: player)
        await fulfillment(of: [ready], timeout: 10)
        return (app, machine, player)
    }

    @MainActor private func revealControls(_ app: XCUIApplication, _ player: XCUIElement) {
        if !app.buttons["Play"].firstMatch.exists && !app.buttons["Pause"].firstMatch.exists {
            player.coordinate(withNormalizedOffset: CGVector(dx: 0.85, dy: 0.5)).tap()
        }
    }
    private enum Color: Equatable { case red, blue }
    @MainActor private func expectColor(_ color: Color, in player: XCUIElement, name: String) async throws {
        for _ in 0..<30 {
            let screenshot = XCUIScreen.main.screenshot()
            if matches(color, screenshot: screenshot, frame: player.frame) {
                let attachment = XCTAttachment(screenshot: screenshot); attachment.name = name
                attachment.lifetime = .keepAlways; add(attachment); return
            }
            try await Task.sleep(for: .milliseconds(100))
        }
        let attachment = XCTAttachment(screenshot: XCUIScreen.main.screenshot()); attachment.name = name + " failure"
        attachment.lifetime = .keepAlways; add(attachment)
        XCTFail("Decoded player did not show the expected \(color) sample")
    }
    @MainActor private func matches(_ color: Color, screenshot: XCUIScreenshot, frame: CGRect) -> Bool {
        guard let image = screenshot.image.cgImage else { return false }
        let scale = CGFloat(image.width) / screenshot.image.size.width
        let side = min(frame.width, frame.height)
        let area = CGRect(x: (frame.midX - side * 0.22) * scale, y: (frame.midY - side * 0.22) * scale, width: 1, height: 1)
        guard let sample = image.cropping(to: area) else { return false }
        var pixel: [UInt8] = [0, 0, 0, 0]
        let drawn = pixel.withUnsafeMutableBytes { buffer -> Bool in
            guard let context = CGContext(data: buffer.baseAddress, width: 1, height: 1, bitsPerComponent: 8, bytesPerRow: 4,
                space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue) else { return false }
            context.draw(sample, in: CGRect(x: 0, y: 0, width: 1, height: 1)); return true
        }
        // AVKit dims the entire picture while its native controls are visible.
        // Test the decoded hue, not the transient overlay's brightness.
        return drawn && (color == .blue
            ? pixel[2] > 70 && Int(pixel[2]) - Int(pixel[0]) > 50
            : pixel[0] > 70 && Int(pixel[0]) - Int(pixel[2]) > 50)
    }
}
