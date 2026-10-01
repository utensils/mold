import XCTest

extension XCTestCase {
    /// Fresh simulators can ask about Local Network or Notifications during
    /// an otherwise unrelated tap. Accept the app's own permissions so the
    /// interaction under test remains the one XCTest is exercising.
    func acceptCompanionPermissions() {
        addUIInterruptionMonitor(withDescription: "Mold Studio permissions") { alert in
            let allow = alert.buttons["Allow"]
            guard allow.exists else { return false }
            allow.tap()
            return true
        }
    }
}

extension XCTestCase {
    /// Always remove this test's exact loopback pairing, including on failure.
    /// Relaunch dismisses whichever viewer, sheet, or menu was left open.
    @MainActor func cleanUpFixture(_ machine: FixtureMachine, port: UInt16, app: XCUIApplication) {
        addTeardownBlock {
            await MainActor.run {
                defer { app.terminate(); machine.stop() }
                app.terminate()
                app.launchArguments = ["-UIPreferredContentSizeCategoryName", "UICTContentSizeCategoryL"]
                app.launch()
                XCTAssertTrue(app.navigateToDestination("Library", shortcut: "2"))
                app.chooseLibraryShelf("All Prints")
                XCTAssertTrue(app.navigateToDestination("Machines", shortcut: "5"))
                let card = app.descendants(matching: .any).matching(NSPredicate(format:
                    "identifier BEGINSWITH 'machine-card-' AND label MATCHES %@",
                    ".*127\\.0\\.0\\.1:\(port)([^0-9].*|$)")).firstMatch
                if card.waitForExistence(timeout: 5) {
                    for _ in 0..<8 where !card.isHittable { app.swipeUp() }
                    XCTAssertTrue(card.isHittable)
                    card.tap()
                    let remove = app.buttons["Remove…"].firstMatch
                    for _ in 0..<8 where !remove.isHittable { app.swipeUp() }
                    XCTAssertTrue(remove.isHittable)
                    remove.tap()
                    let confirm = app.buttons["Remove"].firstMatch
                    XCTAssertTrue(confirm.waitForExistence(timeout: 5))
                    confirm.tap()
                    XCTAssertTrue(card.waitForNonExistence(timeout: 5), "Fixture pairing must not survive teardown")
                }
            }
        }
    }
}

extension XCUIApplication {
    /// The system title menu has no custom label to identify; derive its title.
    @MainActor func chooseLibraryShelf(_ title: String) {
        let picker = descendants(matching: .any)["library-collections"].firstMatch
        if picker.exists {
            picker.tap()
        } else {
            let currentTitle = navigationBars.firstMatch.identifier
            let titleButton = buttons.matching(NSPredicate(format: "label == %@", currentTitle)).firstMatch
            XCTAssertTrue(titleButton.waitForExistence(timeout: 5))
            titleButton.tap()
        }
        func entry() -> XCUIElement? {
            let predicate = NSPredicate(format: "label == %@ OR identifier == %@", title, title)
            return [menuItems, buttons, popUpButtons].flatMap {
                $0.matching(predicate).allElementsBoundByIndex
            }.first { $0.isHittable }
        }
        // SwiftUI Picker can be nested inside a Menu on some OS versions.
        if entry() == nil {
            let shelf = popUpButtons.matching(NSPredicate(format: "label BEGINSWITH 'Shelf'")).firstMatch
            if shelf.waitForExistence(timeout: 2) { shelf.tap() }
        }
        let ready = NSPredicate { _, _ in entry() != nil }
        XCTAssertEqual(XCTWaiter.wait(for: [XCTNSPredicateExpectation(predicate: ready, object: nil)], timeout: 5), .completed)
        entry()?.tap()
    }
}
