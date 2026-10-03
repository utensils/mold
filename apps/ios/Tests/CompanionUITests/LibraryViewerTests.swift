import XCTest

final class LibraryViewerTests: XCTestCase {
    override func setUp() {
        super.setUp()
        acceptCompanionPermissions()
    }

    @MainActor func testClosingViewerKeepsLibraryScrollPlace() async throws {
        continueAfterFailure = false
        let machine = try FixtureMachine(galleryPrints: 300, galleryFavorites: 30, libraryMutations: true)
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
        for _ in 0..<20 where !print.isHittable { grid.swipeUp(velocity: .slow) }
        XCTAssertTrue(print.isHittable, "the test print must be reached below the first screen")
        let originalFrame = print.frame
        let before = XCTAttachment(screenshot: app.screenshot())
        before.name = "Library viewport before viewer"
        before.lifetime = .keepAlways
        add(before)
        print.tap()
        XCTAssertTrue(app.buttons["Info"].firstMatch.waitForExistence(timeout: 5))
        XCTAssertFalse(app.buttons["Library"].firstMatch.isHittable, "Viewer keeps main tab chrome hidden")
        // Move beyond the five-page window, then back to the opening print.
        // The selected domain ID must survive every window recentering.
        for _ in 0..<4 { app.swipeLeft(velocity: .slow) }
        XCTAssertTrue(app.descendants(matching: .any)["viewer-print-fixture-54.png"].firstMatch.isHittable,
                      "forward swipes must cross the initial page window")
        for _ in 0..<4 { app.swipeRight(velocity: .slow) }
        XCTAssertTrue(app.descendants(matching: .any)["viewer-print-fixture-50.png"].firstMatch.isHittable,
                      "backward swipes must return to the original selected print")
        app.navigationBars.buttons["BackButton"].tap()
        XCTAssertTrue(app.navigationBars["All Prints"].waitForExistence(timeout: 5))
        XCTAssertTrue(print.waitForExistence(timeout: 5))
        XCTAssertTrue(print.isHittable, "closing the viewer should return to the same grid position")

        XCTAssertEqual(print.frame.minY, originalFrame.minY, accuracy: 3,
                       "return must preserve the exact viewport, not just reveal the opened tile")

        let after = XCTAttachment(screenshot: app.screenshot())
        after.name = "Library viewport after viewer"
        after.lifetime = .keepAlways
        add(after)

        // A different nonempty shelf starts at its first tile, not at the
        // deep offset inherited from All Prints.
        app.chooseLibraryShelf("Favourites")
        let firstFavorite = app.buttons.matching(NSPredicate(format: "label BEGINSWITH 'Fixture 0,'")).firstMatch
        XCTAssertTrue(firstFavorite.waitForExistence(timeout: 5))
        XCTAssertTrue(firstFavorite.isHittable)

        // Only this loopback fixture accepts in-memory gallery edits. Verify
        // that the pushed viewer receives new immutable parent projections.
        app.chooseLibraryShelf("All Prints")
        for _ in 0..<20 where !print.isHittable { grid.swipeUp(velocity: .slow) }
        XCTAssertTrue(print.isHittable)
        print.tap()
        let favorite = app.buttons["Favourite"].firstMatch
        XCTAssertTrue(favorite.waitForExistence(timeout: 5))
        favorite.tap()
        XCTAssertTrue(app.buttons["Unfavourite"].firstMatch.waitForExistence(timeout: 5))
        app.buttons["Delete"].firstMatch.tap()
        XCTAssertTrue(app.navigationBars["All Prints"].waitForExistence(timeout: 5),
                      "removing the current print must dismiss its viewer")
        XCTAssertFalse(print.exists)
        XCTAssertTrue(machine.requestLog().contains("POST /api/gallery/mutations"))
        XCTAssertTrue(machine.requestLog().contains("POST /api/gallery/trash"))

    }
}
