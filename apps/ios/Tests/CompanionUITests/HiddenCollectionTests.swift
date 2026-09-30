import XCTest

/// Real native interaction against a loopback machine whose collection
/// mutations stay in the fixture process. Run on iPhone and iPad.
final class HiddenCollectionTests: XCTestCase {
    override func setUp() {
        super.setUp()
        acceptCompanionPermissions()
    }

    @MainActor func testHideShowAndOpenHiddenCollection() async throws {
        continueAfterFailure = false
        let machine = try FixtureMachine(galleryPrints: 2, collectionFixture: true)
        let port = try await machine.start()
        defer { machine.stop() }
        let app = XCUIApplication()
        app.launchArguments = ["-UIPreferredContentSizeCategoryName", "UICTContentSizeCategoryL"]
        app.launch()
        XCTAssertTrue(app.navigateToDestination("Machines", shortcut: "5"))
        app.buttons["Add a Machine"].firstMatch.tap()
        app.buttons.matching(NSPredicate(format: "label BEGINSWITH 'Enter an Address'")).firstMatch.tap()
        let address = app.textFields["machine-address"]
        XCTAssertTrue(address.waitForExistence(timeout: 5))
        address.tap()
        address.typeText("127.0.0.1:\(port)")
        app.buttons["Add"].firstMatch.tap()
        XCTAssertTrue(app.navigateToDestination("Library", shortcut: "2"))
        let hiddenPrint = app.buttons.matching(NSPredicate(format: "label BEGINSWITH 'Fixture 0,'")).firstMatch
        let normalPrint = app.buttons.matching(NSPredicate(format: "label BEGINSWITH 'Fixture 1,'")).firstMatch
        XCTAssertTrue(hiddenPrint.waitForExistence(timeout: 10))
        XCTAssertTrue(normalPrint.exists)
        try manage(app)
        let toggle = app.switches["Hide from All Prints"].firstMatch
        XCTAssertTrue(toggle.waitForExistence(timeout: 5))
        toggle.coordinate(withNormalizedOffset: CGVector(dx: 0.95, dy: 0.5)).tap()
        XCTAssertTrue(waitForValue(toggle, "1"))
        capture(app, "Hidden collection management")
        app.navigationBars["Collections"].buttons["Done"].tap()
        XCTAssertTrue(hiddenPrint.waitForNonExistence(timeout: 10))
        XCTAssertTrue(normalPrint.exists)
        capture(app, "All Prints excludes hidden member")

        try manage(app)
        app.buttons["UAT Drafts"].firstMatch.tap()
        XCTAssertTrue(app.navigationBars["UAT Drafts"].waitForExistence(timeout: 5))
        XCTAssertTrue(hiddenPrint.waitForExistence(timeout: 5))
        XCTAssertFalse(normalPrint.exists)
        capture(app, "Explicit hidden shelf remains accessible")
        try manage(app)
        toggle.coordinate(withNormalizedOffset: CGVector(dx: 0.95, dy: 0.5)).tap()
        XCTAssertTrue(waitForValue(toggle, "0"))
        app.navigationBars["Collections"].buttons["Done"].tap()
        // Selecting the main Library tab returns the general grid on iPad;
        // phone keeps the chosen shelf, so use its shelf picker if present.
        if app.buttons["library-collections"].exists {
            app.buttons["library-collections"].tap()
        } else {
            app.navigationBars["UAT Drafts"].buttons["UAT Drafts"].tap()
        }
        // Native menu entries may expose Button or PopUpButton across size classes.
        let allPrints = app.descendants(matching: .any)
            .matching(NSPredicate(format: "label == 'All Prints'")).firstMatch
        XCTAssertTrue(allPrints.waitForExistence(timeout: 5))
        allPrints.tap()
        XCTAssertTrue(hiddenPrint.waitForExistence(timeout: 5))
        XCTAssertTrue(normalPrint.waitForExistence(timeout: 5))
        capture(app, "Showing collection restores All Prints")

        app.terminate()
    }

    @MainActor func testCollectionsExtraSmallAccessibility() async throws {
        try await auditCollections(size: "UICTContentSizeCategoryXS")
    }

    @MainActor func testCollectionsLargeAccessibility() async throws {
        try await auditCollections(size: "UICTContentSizeCategoryL")
    }

    @MainActor func testCollectionsAX5Accessibility() async throws {
        try await auditCollections(size: "UICTContentSizeCategoryAccessibilityXXXL")
    }

    @MainActor private func auditCollections(size: String) async throws {
        let machine = try FixtureMachine(galleryPrints: 2, collectionFixture: true)
        let port = try await machine.start()
        defer { machine.stop() }
        let app = XCUIApplication()
        app.launchArguments = ["-UIPreferredContentSizeCategoryName", size]
        app.launch()
        XCTAssertTrue(app.navigateToDestination("Machines", shortcut: "5"))
        app.buttons["Add a Machine"].firstMatch.tap()
        app.buttons.matching(NSPredicate(format: "label BEGINSWITH 'Enter an Address'")).firstMatch.tap()
        let address = app.textFields["machine-address"]
        XCTAssertTrue(address.waitForExistence(timeout: 5))
        address.tap()
        address.typeText("127.0.0.1:\(port)")
        app.buttons["Add"].firstMatch.tap()
        XCTAssertTrue(app.navigateToDestination("Library", shortcut: "2"))
        try manage(app)
        let sheet = app.descendants(matching: .any)["collections-sheet"].firstMatch
        XCTAssertTrue(sheet.waitForExistence(timeout: 5))
        // Inspect actual sheet descendants; dimmed background is not its UI.
        for types: XCUIAccessibilityAuditType in [
            .contrast, [.dynamicType, .textClipped, .hitRegion, .sufficientElementDescription],
        ] {
            for attempt in 0..<2 {
                do {
                    try app.performAccessibilityAudit(for: types) { issue in
                        if issue.auditType == .dynamicType, let element = issue.element,
                           app.navigationBars["Collections"].descendants(matching: element.elementType)
                            .matching(NSPredicate(format: "label == %@", element.label))
                            .allElementsBoundByIndex.contains(where: { $0.frame == element.frame }) {
                            // Native navigation bars cap text and provide Large Content Viewer.
                            return true
                        }
                        guard issue.auditType == .contrast else { return false }
                        guard let element = issue.element else { return false }
                        return ![sheet, app.navigationBars["Collections"]].contains { container in
                            container.descendants(matching: element.elementType)
                                .matching(NSPredicate(format: "label == %@", element.label))
                                .allElementsBoundByIndex.contains { $0.frame == element.frame }
                        }
                    }
                    break
                } catch {
                    let failure = error as NSError
                    guard attempt == 0,
                          (failure.domain == "com.apple.xcode.xctest.accessibilityAudit" && failure.code == -56)
                            || failure.domain == "com.apple.dt.XCTest.XCTFuture" else { throw error }
                    // Match the shell audit's one bounded harness-timeout retry.
                }
            }
        }
        capture(app, "Collections accessibility at \(size)")
        app.navigationBars["Collections"].buttons["Done"].tap()
        app.terminate()
    }

    @MainActor private func manage(_ app: XCUIApplication) throws {
        let sheet = app.navigationBars["Collections"]
        let item = app.buttons["Manage Collections…"].firstMatch
        for _ in 0..<3 {
            if !item.exists { app.buttons["View Options"].firstMatch.tap() }
            guard item.waitForExistence(timeout: 3) else { continue }
            Thread.sleep(forTimeInterval: 0.75)
            item.tap()
            if sheet.waitForExistence(timeout: 3) { return }
        }
        XCTFail("Collections sheet could not be opened by the native menu")
        throw NSError(domain: "HiddenCollectionTests", code: 1)
    }

    @MainActor private func waitForValue(_ element: XCUIElement, _ value: String) -> Bool {
        XCTWaiter.wait(for: [XCTNSPredicateExpectation(predicate: NSPredicate(format: "value == %@", value),
                                                      object: element)], timeout: 10) == .completed
    }

    @MainActor private func capture(_ app: XCUIApplication, _ name: String) {
        let attachment = XCTAttachment(screenshot: app.screenshot())
        attachment.name = name
        attachment.lifetime = .keepAlways
        add(attachment)
    }
}
