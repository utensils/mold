import XCTest

final class PairingPasteUITests: XCTestCase {
    @MainActor func testLongPairingLinkKeepsConfirmationReachable() {
        continueAfterFailure = false
        let app = XCUIApplication()
        defer { app.terminate() }
        app.launch()
        XCTAssertTrue(app.navigateToDestination("Machines", shortcut: "5"))
        app.buttons["Add a Machine"].firstMatch.tap()
        app.buttons.matching(NSPredicate(format: "label BEGINSWITH 'Scan a Pairing Code'")).firstMatch.tap()
        let input = app.textFields["pairing-link-input"]
        XCTAssertTrue(input.waitForExistence(timeout: 5))
        input.tap()
        input.typeText("https://utensils.io/mold/pair#" + String(repeating: "a", count: 1600))
        XCTAssertLessThan(input.frame.height, 200, "A real Secure code must scroll inside a bounded input instead of moving Pair many screens away")
        app.swipeUp()
        let pair = app.buttons["pairing-link-submit"]
        XCTAssertTrue(pair.waitForExistence(timeout: 5))
        XCTAssertTrue(pair.isHittable, "Pair must remain reachable after a long paste")
    }
}
