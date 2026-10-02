import XCTest
import UIKit

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
        let app = try await populatedLibrary(offlineCopy: true)
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
        // Releasing the long press can select the picker button. Wait for its
        // asynchronous fetch and dismissal before attempting a fallback tap.
        let picker = app.navigationBars["Choose from Library"]
        if !picker.waitForNonExistence(timeout: 10) {
            XCTAssertTrue(picked.waitForExistence(timeout: 5))
            picked.tap()
        }
        XCTAssertTrue(picker.waitForNonExistence(timeout: 10))
        XCTAssertTrue(app.buttons["Start from"].firstMatch.waitForExistence(timeout: 5))
        attach(app, name: "Source selected after long press")
    }

    @MainActor func testIPadDragPreviewRemainsUsable() async throws {
        try XCTSkipUnless(UIDevice.current.userInterfaceIdiom == .pad, "iPad drag interaction")
        continueAfterFailure = false
        let app = try await populatedLibrary()
        let tile = fixturePrint(in: app)
        let start = tile.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.5))
        let outside = app.coordinate(withNormalizedOffset: CGVector(dx: 0.95, dy: 0.03))
        start.press(forDuration: 1, thenDragTo: outside)
        XCTAssertEqual(app.state, .runningForeground)
        XCTAssertTrue(tile.waitForExistence(timeout: 5))
        tile.tap()
        XCTAssertTrue(app.buttons["Info"].firstMatch.waitForExistence(timeout: 5))
        attach(app, name: "iPad library after drag preview")
    }

    @MainActor private func populatedLibrary(offlineCopy: Bool = false) async throws -> XCUIApplication {
        if offlineCopy {
            let earlier = try FixtureMachine(galleryPrints: 6, collectionFixture: true)
            let port = try await earlier.start()
            let app = XCUIApplication()
            cleanUpFixture(earlier, port: port, app: app)
            app.launch()
            pair(port, in: app, name: "Offline Source Fixture")
            XCTAssertTrue(app.navigateToDestination("Library", shortcut: "2"))
            XCTAssertTrue(fixturePrint(in: app).waitForExistence(timeout: 10))
            earlier.stop()
            app.terminate()
        }
        let machine = try FixtureMachine(galleryPrints: 6, collectionFixture: true)
        let port = try await machine.start()
        let app = XCUIApplication()
        cleanUpFixture(machine, port: port, app: app)
        app.launch()
        pair(port, in: app, name: offlineCopy ? "Live Source Fixture" : nil)
        XCTAssertTrue(app.navigateToDestination("Library", shortcut: "2"))
        XCTAssertTrue(fixturePrint(in: app).waitForExistence(timeout: 10))
        if offlineCopy {
            let merged = app.buttons.matching(NSPredicate(format:
                "label BEGINSWITH 'Fixture 0,' AND label CONTAINS 'Offline Source Fixture' AND label CONTAINS 'Live Source Fixture'")).firstMatch
            XCTAssertTrue(merged.waitForExistence(timeout: 10), "Both machine copies must join the source tile")
        }
        return app
    }

    @MainActor private func pair(_ port: UInt16, in app: XCUIApplication, name: String? = nil) {
        XCTAssertTrue(app.navigateToDestination("Machines", shortcut: "5"))
        app.buttons["Add a Machine"].firstMatch.tap()
        app.buttons.matching(NSPredicate(format: "label BEGINSWITH 'Enter an Address'")).firstMatch.tap()
        if let name {
            let field = app.textFields["machine-name"]
            XCTAssertTrue(field.waitForExistence(timeout: 5))
            field.tap(); field.typeText(name)
        }
        let address = app.textFields["machine-address"]
        XCTAssertTrue(address.waitForExistence(timeout: 5))
        address.tap(); address.typeText("127.0.0.1:\(port)")
        app.buttons["Add"].firstMatch.tap()
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
