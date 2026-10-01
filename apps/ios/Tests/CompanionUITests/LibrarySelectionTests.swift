import XCTest

final class LibrarySelectionTests: XCTestCase {
    override func setUp() { super.setUp(); acceptCompanionPermissions() }

    @MainActor func testSelectionKeepsViewportAndDragSelectsRange() async throws {
        continueAfterFailure = false
        let machine = try FixtureMachine(galleryPrints: 90, mixedMedia: true)
        let port = try await machine.start()
        let app = XCUIApplication()
        cleanUpFixture(machine, port: port, app: app)
        app.launch()
        XCTAssertTrue(app.navigateToDestination("Machines", shortcut: "5"))
        app.buttons["Add a Machine"].firstMatch.tap()
        app.buttons.matching(NSPredicate(format: "label BEGINSWITH 'Enter an Address'")).firstMatch.tap()
        let address = app.textFields["machine-address"]
        XCTAssertTrue(address.waitForExistence(timeout: 5))
        address.tap(); address.typeText("127.0.0.1:\(port)")
        app.buttons["Add"].firstMatch.tap()
        XCTAssertTrue(app.navigateToDestination("Library", shortcut: "2"))
        let grid = app.scrollViews.firstMatch
        grid.swipeUp(velocity: .slow)
        grid.swipeUp(velocity: .slow)
        app.buttons["Select"].tap()
        let tiles = app.buttons.matching(NSPredicate(format: "label BEGINSWITH 'Fixture '"))
        let visible = tiles.allElementsBoundByIndex.filter { $0.isHittable && $0.frame.minY > 180 && $0.frame.maxY < 650 }
        XCTAssertGreaterThanOrEqual(visible.count, 3)
        let first = visible[0]
        let before = first.frame
        first.tap()
        XCTAssertEqual(first.frame.minY, before.minY, accuracy: 2, "a selection must not re-anchor the scroll view")
        visible[1].tap()
        XCTAssertEqual(first.frame.minY, before.minY, accuracy: 2)
        first.tap() // Begin the drag on an unselected tile.
        first.press(forDuration: 0.05, thenDragTo: visible[2])
        XCTAssertTrue(first.isSelected)
        XCTAssertTrue(visible[1].isSelected)
        XCTAssertTrue(visible[2].isSelected)
        XCTAssertEqual(first.frame.minY, before.minY, accuracy: 2)
        first.press(forDuration: 0.05, thenDragTo: visible[2])
        XCTAssertFalse(first.isSelected)
        XCTAssertFalse(visible[1].isSelected)
        XCTAssertFalse(visible[2].isSelected)
        let shot = XCTAttachment(screenshot: app.screenshot())
        shot.name = "Selection stays at its scrolled position"
        shot.lifetime = .keepAlways
        add(shot)
        let sweptTiles = tiles.allElementsBoundByIndex.filter {
            $0.isHittable && $0.frame.minY > 180 && $0.frame.maxY < 650
        }
        let bottomY = try XCTUnwrap(sweptTiles.map(\.frame.minY).max())
        let edgeStart = try XCTUnwrap(sweptTiles.first { abs($0.frame.minY - bottomY) < 2 })
        let edgeBefore = edgeStart.frame.minY
        let endY = app.buttons["Delete"].firstMatch.frame.minY - 10
        let startPoint = edgeStart.coordinate(withNormalizedOffset: CGVector(dx: 0.2, dy: 0.5))
        let endPoint = app.coordinate(withNormalizedOffset: .zero)
            .withOffset(CGVector(dx: app.frame.maxX - 12, dy: endY))
        startPoint.press(forDuration: 0.05, thenDragTo: endPoint,
                         withVelocity: .slow, thenHoldForDuration: 1.2)
        XCTAssertLessThan(edgeStart.frame.minY, edgeBefore - 10, "holding a sweep at the edge should scroll")
        let oldY = first.frame.minY
        grid.swipeUp(velocity: .slow)
        XCTAssertTrue(!first.isHittable || first.frame.minY < oldY - 20,
                      "vertical scrolling remains available in Select mode")
        app.buttons["Done"].tap()
        app.buttons["View Options"].tap()
        let media = app.buttons.matching(NSPredicate(format: "label BEGINSWITH 'Media Type'")).firstMatch
        if media.waitForExistence(timeout: 2) { media.tap() }
        app.buttons["Videos"].firstMatch.tap()
        XCTAssertTrue(app.navigationBars["All Prints · Videos"].waitForExistence(timeout: 5))
        let video = app.buttons.matching(NSPredicate(format: "label BEGINSWITH 'Fixture ' AND label CONTAINS 'clip'"))
        XCTAssertTrue(video.firstMatch.waitForExistence(timeout: 5))
        XCTAssertFalse(app.buttons.matching(NSPredicate(format: "label BEGINSWITH 'Fixture ' AND label CONTAINS 'picture'")).firstMatch.exists)
        app.buttons["View Options"].tap()
        let mediaAgain = app.buttons.matching(NSPredicate(format: "label BEGINSWITH 'Media Type'")).firstMatch
        if mediaAgain.waitForExistence(timeout: 2) { mediaAgain.tap() }
        app.buttons["All Media"].firstMatch.tap()
        XCTAssertTrue(app.navigationBars["All Prints"].waitForExistence(timeout: 5))
    }
}
