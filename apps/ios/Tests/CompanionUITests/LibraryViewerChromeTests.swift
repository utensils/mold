import XCTest

/// Leaving the viewer must give the Library back its own chrome. The viewer
/// hides the tab bar and puts Share / Favourite / Info / Delete in a bottom
/// bar; a zoom-transition dismissal (swipe down, or a back swipe) used to
/// leave both behind on the grid: no tab bar, and a viewer's buttons floating
/// over prints they no longer act on.
final class LibraryViewerChromeTests: XCTestCase {
    @MainActor func testEveryWayOutOfTheViewerRestoresTheTabBar() async throws {
        continueAfterFailure = false
        let machine = try FixtureMachine(galleryPrints: 40)
        let port = try await machine.start()
        let app = XCUIApplication()
        cleanUpFixture(machine, port: port, app: app)
        app.launch()
        XCTAssertTrue(app.navigateToDestination("Machines", shortcut: "5"))
        app.buttons["Add a Machine"].firstMatch.tap()
        app.buttons.matching(NSPredicate(format: "label BEGINSWITH 'Enter an Address'")).firstMatch.tap()
        app.chooseRegularConnection()
        let address = app.textFields["machine-address"]
        XCTAssertTrue(address.waitForExistence(timeout: 5))
        address.tap()
        address.typeText("127.0.0.1:\(port)")
        app.buttons["Add"].firstMatch.tap()
        XCTAssertTrue(app.navigateToDestination("Library", shortcut: "2"))

        let print = app.buttons.matching(NSPredicate(format: "label CONTAINS 'Fixture 3,'")).firstMatch
        XCTAssertTrue(print.waitForExistence(timeout: 10))
        let ways: [(String, (XCUIApplication) -> Void)] = [
            ("Back button", { $0.navigationBars.buttons["BackButton"].tap() }),
            ("swipe down", { app in
                let top = app.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.4))
                top.press(forDuration: 0.05, thenDragTo: app.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.95)),
                          withVelocity: .default, thenHoldForDuration: 0)
            }),
            ("edge swipe back", { app in
                let edge = app.coordinate(withNormalizedOffset: CGVector(dx: 0.01, dy: 0.5))
                edge.press(forDuration: 0.05, thenDragTo: app.coordinate(withNormalizedOffset: CGVector(dx: 0.9, dy: 0.5)),
                           withVelocity: .default, thenHoldForDuration: 0)
            }),
        ]
        for (name, leave) in ways {
            print.tap()
            XCTAssertTrue(app.buttons["Info"].firstMatch.waitForExistence(timeout: 5), name)
            XCTAssertFalse(app.buttons["Library"].firstMatch.isHittable, "\(name): the viewer hides the tab bar")
            leave(app)
            XCTAssertTrue(app.navigationBars["All Prints"].waitForExistence(timeout: 5), name)
            XCTAssertTrue(print.waitForExistence(timeout: 5), name)
            let shot = XCTAttachment(screenshot: app.screenshot())
            shot.name = "Library after leaving the viewer by \(name)"
            shot.lifetime = .keepAlways
            add(shot)
            XCTAssertTrue(app.buttons["Library"].firstMatch.waitForHittable(timeout: 5),
                          "\(name): the tab bar comes back")
            for button in ["Share", "Info", "Delete"] {
                XCTAssertFalse(app.buttons[button].firstMatch.isHittable,
                               "\(name): the viewer's \(button) must not stay on the grid")
            }
        }
    }
}

private extension XCUIElement {
    @MainActor func waitForHittable(timeout: TimeInterval) -> Bool {
        let hittable = XCTNSPredicateExpectation(predicate: NSPredicate(format: "isHittable == true"), object: self)
        return XCTWaiter.wait(for: [hittable], timeout: timeout) == .completed
    }
}
