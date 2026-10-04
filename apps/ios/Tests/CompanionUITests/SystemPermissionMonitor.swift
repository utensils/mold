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
                if !app.navigationBars["All Prints"].exists { app.chooseLibraryShelf("All Prints") }
                XCTAssertTrue(app.navigateToDestination("Machines", shortcut: "5"))
                let card = app.descendants(matching: .any).matching(NSPredicate(format:
                    "identifier BEGINSWITH 'machine-card-' AND label MATCHES %@",
                    ".*127\\.0\\.0\\.1:\(port)([^0-9].*|$)")).firstMatch
                if app.staticTexts["No machines yet"].exists { return }
                let fleet = app.scrollViews["machines-fleet"]
                XCTAssertTrue(fleet.waitForExistence(timeout: 5), "Fixture cleanup must own the Machines viewport")
                if XCTestCase.revealFixtureControl(card, in: fleet, app: app) {
                    card.tap()
                    let detail = app.collectionViews["machine-details"]
                    XCTAssertTrue(detail.waitForExistence(timeout: 5))
                    let remove = detail.buttons["Remove…"].firstMatch
                    XCTAssertTrue(XCTestCase.revealFixtureControl(remove, in: detail, app: app), "The exact fixture's Remove action must be reachable")
                    remove.tap()
                    let confirm = app.buttons["Remove"].firstMatch
                    XCTAssertTrue(confirm.waitForExistence(timeout: 5))
                    confirm.tap()
                    XCTAssertTrue(card.waitForNonExistence(timeout: 5), "Fixture pairing must not survive teardown")
                } else {
                    // The test can fail before Add succeeds. Absence is only
                    // accepted after searching the actual fleet in both directions.
                    XCTAssertFalse(card.exists)
                }
            }
        }
    }

    @MainActor private static func revealFixtureControl(_ target: XCUIElement, in owner: XCUIElement,
                                                 app: XCUIApplication) -> Bool {
        guard owner.exists else { XCTFail("Fixture cleanup scroll owner is missing"); return false }
        func visibleSignature() -> String {
            owner.descendants(matching: .any).allElementsBoundByIndex
                .filter { !$0.identifier.isEmpty && owner.frame.intersects($0.frame) }
                .map { "\($0.identifier):\($0.frame)" }.sorted().joined(separator: "|")
        }
        func visibleTarget() -> Bool {
            target.exists && target.isHittable && owner.frame.intersects(target.frame)
        }
        func pan(down: Bool) {
            let box = owner.frame.intersection(app.frame)
            let top = max(box.minY, app.navigationBars.firstMatch.frame.maxY)
            let bottom = app.tabBars.firstMatch.exists ? min(box.maxY, app.tabBars.firstMatch.frame.minY) : box.maxY
            let height = max(0, bottom - top)
            let origin = app.coordinate(withNormalizedOffset: .zero)
            let x = box.minX + box.width * 0.03
            let center = (top + bottom) / 2
            let direction: CGFloat = down ? 1 : -1
            origin.withOffset(CGVector(dx: x, dy: center - direction * height * 0.2))
                .press(forDuration: 0.1, thenDragTo: origin.withOffset(CGVector(dx: x, dy: center + direction * height * 0.2)),
                       withVelocity: .slow, thenHoldForDuration: 0.3)
        }
        if visibleTarget() { return true }
        // Restore the top before searching lazy content down to its boundary.
        for _ in 0..<12 {
            let before = visibleSignature(); pan(down: true)
            if visibleTarget() { return true }
            if visibleSignature() == before { break }
        }
        for _ in 0..<12 {
            let before = visibleSignature(); pan(down: false)
            if visibleTarget() { return true }
            if visibleSignature() == before { return false }
        }
        let hierarchy = XCTAttachment(string: app.debugDescription)
        hierarchy.name = "Fixture cleanup search exhausted"; hierarchy.lifetime = .keepAlways
        XCTContext.runActivity(named: "Fixture cleanup search exhausted") { $0.add(hierarchy) }
        XCTFail("Fixture cleanup could not reach a scroll boundary while searching for \(target.identifier)")
        return false
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
