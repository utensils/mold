import XCTest

/// Exercise UIKit's real notification activation completion, which a deep-link
/// launch and direct Swift delegate calls do not cover.
final class NotificationTapTests: XCTestCase {
    override func setUp() {
        super.setUp()
        acceptCompanionPermissions()
    }

    @MainActor func testWarmNotificationTapOpensPrintWithoutCrashing() throws {
        try openCompletedPrint(coldLaunch: false)
    }

    @MainActor func testColdNotificationTapOpensPrintWithoutCrashing() throws {
        try openCompletedPrint(coldLaunch: true)
    }

    @MainActor private func openCompletedPrint(coldLaunch: Bool) throws {
        continueAfterFailure = false
        let app = XCUIApplication()
        // An unknown machine is deliberate: the viewer must handle missing
        // media gracefully, and this test cannot contact an inference server.
        let link = "moldstudio://print/\(UUID().uuidString)/notification-fixture.png"
        app.launchArguments = ["--notification-fixture-link", link]
        app.launch()
        let springboard = XCUIApplication(bundleIdentifier: "com.apple.springboard")
        let allow = springboard.buttons["Allow"]
        if allow.waitForExistence(timeout: 2) { allow.tap() }
        XCUIDevice.shared.press(.home)
        // Terminate before waiting: termination can outlast the short-lived
        // banner on a hosted runner, making a previously found element stale.
        if coldLaunch { app.terminate() }
        // Tap the real system banner while the app is away.
        springboard.swipeDown()
        let notification = springboard.buttons.matching(NSPredicate(format:
            "label CONTAINS 'Notification tap regression fixture'")).firstMatch
        XCTAssertTrue(notification.waitForExistence(timeout: 35))
        let banner = XCTAttachment(screenshot: springboard.screenshot())
        banner.name = "Notification Center banner before activation"
        banner.lifetime = .keepAlways
        add(banner)
        notification.tap()
        XCTAssertTrue(app.staticTexts["Not in the Library"].waitForExistence(timeout: 10),
                      "A system notification tap must keep the app alive and route to its print viewer")
        XCTAssertEqual(app.state, .runningForeground)
        let screenshot = XCTAttachment(screenshot: app.screenshot())
        screenshot.lifetime = .keepAlways
        add(screenshot)
        app.buttons["Done"].firstMatch.tap()
        XCTAssertTrue(app.navigateToDestination("Generate", shortcut: "1"))
    }
}
