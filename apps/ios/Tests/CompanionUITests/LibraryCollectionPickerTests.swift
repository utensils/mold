import XCTest

/// The inline shelf picker and date heading must remain readable on a populated
/// Library. The runner audits each size in both system appearances.
final class LibraryCollectionPickerTests: XCTestCase {
    override func setUp() {
        super.setUp()
        acceptCompanionPermissions()
    }

    @MainActor func testExtraSmallPickerContrast() async throws {
        try await auditPicker(size: "UICTContentSizeCategoryXS")
    }

    @MainActor func testLargePickerContrast() async throws {
        try await auditPicker(size: "UICTContentSizeCategoryL")
    }

    @MainActor func testAX5PickerContrast() async throws {
        try await auditPicker(size: "UICTContentSizeCategoryAccessibilityXXXL")
    }

    @MainActor private func auditPicker(size: String) async throws {
        continueAfterFailure = false
        let machine = try FixtureMachine(galleryPrints: 2, collectionFixture: true)
        let port = try await machine.start()
        let app = XCUIApplication()
        cleanUpFixture(machine, port: port, app: app)
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
        let print = app.buttons.matching(NSPredicate(format: "label BEGINSWITH 'Fixture 0,'")).firstMatch
        XCTAssertTrue(print.waitForExistence(timeout: 10))
        let picker = app.navigationBars.buttons["All Prints"].firstMatch
        XCTAssertTrue(picker.waitForExistence(timeout: 5))
        XCTAssertTrue(picker.isHittable, "The navigation title shelf picker must be visible at \(size)")
        XCTAssertTrue(picker.label.contains("All Prints"))
        let dayHeader = app.staticTexts["day-header"].firstMatch
        XCTAssertTrue(dayHeader.waitForExistence(timeout: 5))

        try app.performAccessibilityAudit(for: .contrast) { issue in
            // Unknown audit targets must fail. Scope only known, unrelated
            // elements out of this focused control regression; the full shell
            // audit owns the rest of the populated Library.
            guard let element = issue.element else { return false }
            if [picker, dayHeader].contains(where: { element.identifier == $0.identifier
                && element.elementType == $0.elementType && element.frame == $0.frame }) {
                return false
            }
            let descendants = picker.descendants(matching: element.elementType)
                .matching(NSPredicate(format: "label == %@ AND identifier == %@",
                                      element.label, element.identifier))
            return !descendants.allElementsBoundByIndex.contains { $0.frame == element.frame }
        }
        let screenshot = XCTAttachment(screenshot: app.screenshot())
        screenshot.name = "Library title and date heading contrast at \(size)"
        screenshot.lifetime = .keepAlways
        add(screenshot)
    }
}
