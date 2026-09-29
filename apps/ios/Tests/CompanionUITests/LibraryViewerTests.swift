import XCTest

final class LibraryViewerTests: XCTestCase {
    override func setUp() {
        super.setUp()
        acceptCompanionPermissions()
    }

    @MainActor func testClosingViewerKeepsLibraryScrollPlace() async throws {
        continueAfterFailure = false
        let machine = try FixtureMachine(galleryPrints: 60, galleryFavorites: 30)
        let port = try await machine.start()
        defer { machine.stop() }

        let app = XCUIApplication()
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

        let grid = app.scrollViews.firstMatch
        let print = app.buttons.matching(NSPredicate(format: "label BEGINSWITH 'Fixture 50,'")).firstMatch
        for _ in 0..<12 where !print.isHittable { grid.swipeUp(velocity: .slow) }
        XCTAssertTrue(print.isHittable, "the test print must be reached below the first screen")
        print.tap()
        XCTAssertTrue(app.buttons["Info"].firstMatch.waitForExistence(timeout: 5))
        app.navigationBars.buttons["BackButton"].tap()
        XCTAssertTrue(app.navigationBars["All Prints"].waitForExistence(timeout: 5))
        XCTAssertTrue(print.waitForExistence(timeout: 5))
        XCTAssertTrue(print.isHittable, "closing the viewer should return to the same grid position")

        // A different nonempty shelf starts at its first tile, not at the
        // deep offset inherited from All Prints.
        app.buttons["library-collections"].tap()
        app.buttons["Favourites"].tap()
        let firstFavorite = app.buttons.matching(NSPredicate(format: "label BEGINSWITH 'Fixture 0,'")).firstMatch
        XCTAssertTrue(firstFavorite.waitForExistence(timeout: 5))
        XCTAssertTrue(firstFavorite.isHittable)

        // Leave the simulator's paired-machine list as this test found it.
        XCTAssertTrue(app.navigateToDestination("Machines", shortcut: "5"))
        let card = app.buttons.matching(NSPredicate(format: "label CONTAINS %@", "127.0.0.1:\(port)")).firstMatch
        XCTAssertTrue(card.waitForExistence(timeout: 5))
        for _ in 0..<8 where !card.isHittable { app.swipeUp() }
        card.press(forDuration: 1)
        app.buttons["Remove…"].firstMatch.tap()
        app.buttons["Remove"].firstMatch.tap()
    }
}
