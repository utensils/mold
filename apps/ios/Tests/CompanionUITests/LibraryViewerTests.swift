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

        let app = XCUIApplication()
        cleanUpFixture(machine, port: port, app: app)
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
        XCTAssertFalse(app.buttons["Library"].firstMatch.isHittable, "Viewer keeps main tab chrome hidden")
        app.navigationBars.buttons["BackButton"].tap()
        XCTAssertTrue(app.navigationBars["All Prints"].waitForExistence(timeout: 5))
        XCTAssertTrue(print.waitForExistence(timeout: 5))
        XCTAssertTrue(print.isHittable, "closing the viewer should return to the same grid position")

        // A different nonempty shelf starts at its first tile, not at the
        // deep offset inherited from All Prints.
        app.chooseLibraryShelf("Favourites")
        let firstFavorite = app.buttons.matching(NSPredicate(format: "label BEGINSWITH 'Fixture 0,'")).firstMatch
        XCTAssertTrue(firstFavorite.waitForExistence(timeout: 5))
        XCTAssertTrue(firstFavorite.isHittable)

    }
}
