import XCTest
import UIKit

/// Context menus and drag previews are separately hosted by UIKit. Exercise
/// their real long-press boundary with a populated library, not just a tap.
final class LibraryLongPressTests: XCTestCase {
    override func setUp() { super.setUp(); acceptCompanionPermissions() }

    @MainActor func testPopulatedLibrarySelectContrastAtLargestText() async throws {
        continueAfterFailure = false
        let machine = try FixtureMachine(galleryPrints: 6)
        let port = try await machine.start()
        let app = XCUIApplication()
        defer { app.terminate(); machine.stop() }
        cleanUpFixture(machine, port: port, app: app)
        app.launchArguments = ["-UIPreferredContentSizeCategoryName", "UICTContentSizeCategoryAccessibilityXXXL"]
        app.launch(); try assertEmptyMachines(app)
        pair(port, in: app, name: "Toolbar Contrast Fixture")
        let identity = try machineIdentity(port: port, app: app)
        XCTAssertTrue(app.navigateToDestination("Library", shortcut: "2"))
        XCTAssertTrue(fixturePrint(in: app).waitForExistence(timeout: 10))
        try auditSelect(app, stage: "Live populated Library")
        XCUIDevice.shared.press(.home)
        XCTAssertTrue(app.wait(for: .runningBackground, timeout: 5))
        machine.stop(); app.terminate(); app.launch()
        XCTAssertTrue(app.navigateToDestination("Library", shortcut: "2"))
        XCTAssertTrue(fixturePrint(in: app).waitForExistence(timeout: 10), "Offline regression requires retained visible prints")
        try verifyOfflineLibrary(app, names: ["Toolbar Contrast Fixture"], identities: [identity], stage: "Single offline Library AX5")
    }

    @MainActor func testFourOfflineLibrariesKeepPrintsReachableAtEveryTextSize() async throws {
        continueAfterFailure = false
        let machines = try (0..<4).map { _ in try FixtureMachine(galleryPrints: 6) }
        let app = XCUIApplication()
        defer { app.terminate(); machines.forEach { $0.stop() } }
        let names = ["Reference media workstation East", "Repeated machine", "Repeated machine", "Other media workstation"]
        var identities: [String] = []
        app.launchArguments = ["-UIPreferredContentSizeCategoryName", "UICTContentSizeCategoryL"]
        app.launch(); try assertEmptyMachines(app)
        for (index, machine) in machines.enumerated() {
            let port = try await machine.start()
            cleanUpFixture(machine, port: port, app: app)
            pair(port, in: app, name: names[index])
            identities.append(try machineIdentity(port: port, app: app))
            XCTAssertTrue(app.navigateToDestination("Library", shortcut: "2"))
            let listing = XCTNSPredicateExpectation(predicate: NSPredicate { _, _ in
                machine.requestLog().contains("GET /api/gallery")
            }, object: nil)
            XCTAssertEqual(XCTWaiter.wait(for: [listing], timeout: 10), .completed,
                           "Each host must supply its gallery listing")
            XCTAssertTrue(fixturePrint(in: app).waitForExistence(timeout: 10))
        }
        XCUIDevice.shared.press(.home)
        XCTAssertTrue(app.wait(for: .runningBackground, timeout: 5))
        machines.forEach { $0.stop() }; app.terminate()
        for size in ["UICTContentSizeCategoryXS", "UICTContentSizeCategoryL", "UICTContentSizeCategoryAccessibilityXXXL"] {
            app.launchArguments = ["-UIPreferredContentSizeCategoryName", size]
            app.launch()
            XCTAssertTrue(app.navigateToDestination("Library", shortcut: "2"))
            try verifyOfflineLibrary(app, names: names, identities: identities, stage: "Four offline Libraries \(size)")
            app.terminate()
        }
    }

    @MainActor private func verifyOfflineLibrary(_ app: XCUIApplication, names: [String], identities: [String], stage: String) throws {
        let status = app.buttons["offline-library-status"]
        XCTAssertTrue(status.waitForExistence(timeout: 10))
        let completeCount = XCTNSPredicateExpectation(predicate: NSPredicate(format: "label CONTAINS %@", "\(names.count) machine"), object: status)
        XCTAssertEqual(XCTWaiter.wait(for: [completeCount], timeout: 10), .completed, "All retained hosts must finish their reachability checks")
        settle(status)
        XCTAssertTrue(status.label.localizedCaseInsensitiveContains("saved prints"))
        XCTAssertGreaterThanOrEqual(status.frame.height, 44)
        let caption = status.staticTexts[status.label].firstMatch
        XCTAssertTrue(caption.exists)
        XCTAssertTrue(status.frame.contains(caption.frame))
        XCTAssertLessThan(caption.frame.width, status.frame.width, "Text must retain its intrinsic bounds inside the padded status button")
        XCTAssertLessThan(caption.frame.height, status.frame.height)
        XCTAssertTrue(status.isHittable)
        let top = app.navigationBars.firstMatch.frame.maxY
        let bottom = app.tabBars.firstMatch.exists ? app.tabBars.firstMatch.frame.minY : app.frame.maxY
        let statusViewport = CGRect(x: app.frame.minX, y: top, width: app.frame.width, height: max(0, bottom - top))
        XCTAssertTrue(statusViewport.contains(status.frame))
        try auditContrast(status, app: app, stage: stage + " status")
        let tile = fixturePrint(in: app)
        XCTAssertTrue(tile.waitForExistence(timeout: 10), "The pinned status must leave retained prints available")
        try revealPrint(tile, app: app)
        attach(app, name: stage + " Retained print below bounded status")
        try selectRetainedPrint(tile, app: app)
        status.tap()
        attach(app, name: stage + " Initial offline details")
        let candidate = app.collectionViews["offline-library-details"].firstMatch
        let details = try XCTUnwrap(candidate.waitForExistence(timeout: 5) ? candidate : nil, "The details sheet must own its native List: \(app.debugDescription)")
        let heading = app.navigationBars["Saved Prints"]
        let title = try XCTUnwrap(heading.waitForExistence(timeout: 5) ? heading : nil, "The details heading must be presented: \(app.debugDescription)")
        XCTAssertEqual(Set(identities).count, names.count, "Duplicate names must keep distinct host identities")
        for (index, name) in names.enumerated() {
            let text = details.staticTexts["offline-library-host-" + identities[index]].firstMatch
            for _ in 0..<12 {
                let viewport = details.frame.intersection(app.frame)
                let visible = CGRect(x: viewport.minX, y: max(viewport.minY, title.frame.maxY),
                                     width: viewport.width, height: max(0, viewport.maxY - max(viewport.minY, title.frame.maxY)))
                if text.exists, visible.contains(text.frame) { break }
                if !text.exists { details.swipeUp() }
                else {
                    XCTAssertLessThanOrEqual(text.frame.height, visible.height, "Full machine name must fit the details viewport")
                    let delta = text.frame.midY - visible.midY
                    let distance = min(abs(delta), visible.height * 0.4)
                    let origin = app.coordinate(withNormalizedOffset: .zero)
                    let direction: CGFloat = delta > 0 ? 1 : -1
                    let x = visible.minX + visible.width * 0.03
                    origin.withOffset(CGVector(dx: x, dy: visible.midY + direction * distance / 2))
                        .press(forDuration: 0.1, thenDragTo: origin.withOffset(CGVector(dx: x, dy: visible.midY - direction * distance / 2)),
                               withVelocity: .slow, thenHoldForDuration: 0.3)
                }
            }
            XCTAssertTrue(text.exists, "Every full unavailable machine name must be reachable: \(name)")
            XCTAssertEqual(text.label, name)
            settle(text)
            let frame = details.frame.intersection(app.frame)
            let top = max(frame.minY, title.frame.maxY)
            let viewport = CGRect(x: frame.minX, y: top, width: frame.width, height: max(0, frame.maxY - top))
            XCTAssertTrue(viewport.contains(text.frame), "Full host name \(text.frame) must fit \(viewport)")
            try auditContrast(text, app: app, stage: stage + " name " + identities[index])
            attach(app, name: stage + " Full offline name " + identities[index])
        }
        let done = title.buttons["Done"]
        XCTAssertTrue(done.exists)
        XCTAssertTrue(app.frame.contains(done.frame))
        XCTAssertTrue(title.frame.contains(done.frame))
        XCTAssertTrue(done.isHittable)
        try auditContrast(done, app: app, stage: stage + " details Done")
        done.tap()
        XCTAssertTrue(details.waitForNonExistence(timeout: 5))
        try revealPrint(tile, app: app)
        try auditSelect(app, stage: stage + " after details dismissal")
        try selectRetainedPrint(tile, app: app)
    }

    @MainActor private func libraryViewport(_ app: XCUIApplication) -> CGRect {
        let grid = app.scrollViews.firstMatch
        XCTAssertTrue(grid.exists)
        let status = app.buttons["offline-library-status"]
        let box = grid.frame.intersection(app.frame)
        let top = max(box.minY, max(app.navigationBars.firstMatch.frame.maxY, status.frame.maxY))
        let bottom = app.tabBars.firstMatch.exists ? min(box.maxY, app.tabBars.firstMatch.frame.minY) : box.maxY
        return CGRect(x: box.minX, y: top, width: box.width, height: max(0, bottom - top))
    }

    @discardableResult @MainActor private func revealPrint(_ tile: XCUIElement, app: XCUIApplication) throws -> CGRect {
        let viewport = libraryViewport(app)
        XCTAssertGreaterThanOrEqual(viewport.height, 44)
        func fullyExposed() -> Bool {
            if tile.frame.height <= viewport.height { return viewport.contains(tile.frame) }
            let visible = viewport.intersection(tile.frame)
            return visible.height >= viewport.height && visible.width >= tile.frame.width
        }
        for _ in 0..<8 where !fullyExposed() {
            let delta = tile.frame.midY - viewport.midY
            let distance = min(abs(delta), viewport.height * 0.4)
            let direction: CGFloat = delta > 0 ? 1 : -1
            let origin = app.coordinate(withNormalizedOffset: .zero)
            let x = viewport.minX + viewport.width * 0.03
            origin.withOffset(CGVector(dx: x, dy: viewport.midY + direction * distance / 2))
                .press(forDuration: 0.1, thenDragTo: origin.withOffset(CGVector(dx: x, dy: viewport.midY - direction * distance / 2)),
                               withVelocity: .slow, thenHoldForDuration: 0.3)
            settle(tile)
        }
        let visible = viewport.intersection(tile.frame)
        XCTAssertTrue(fullyExposed(), "Retained print \(tile.frame) must expose its full available image area in \(viewport)")
        XCTAssertGreaterThanOrEqual(visible.width, 44)
        XCTAssertGreaterThanOrEqual(visible.height, 44)
        XCTAssertTrue(tile.isHittable)
        return visible
    }

    @MainActor private func selectRetainedPrint(_ tile: XCUIElement, app: XCUIApplication) throws {
        let select = app.buttons["Select"].firstMatch
        XCTAssertTrue(select.isHittable)
        select.tap()
        let done = app.buttons["Done"].firstMatch
        XCTAssertTrue(done.waitForExistence(timeout: 5))
        settle(tile)
        let viewport = libraryViewport(app)
        let share = app.buttons["Share"].firstMatch
        XCTAssertTrue(share.exists)
        let unobscured = CGRect(x: viewport.minX, y: viewport.minY, width: viewport.width,
                               height: max(0, min(viewport.maxY, share.frame.minY - 12) - viewport.minY))
        let hit = tile.frame.intersection(unobscured)
        XCTAssertGreaterThanOrEqual(hit.width, 44)
        XCTAssertGreaterThanOrEqual(hit.height, 44)
        app.coordinate(withNormalizedOffset: .zero).withOffset(CGVector(dx: hit.midX, dy: hit.midY)).tap()
        let selected = XCTNSPredicateExpectation(predicate: NSPredicate(format: "selected == true"), object: tile)
        XCTAssertEqual(XCTWaiter.wait(for: [selected], timeout: 5), .completed, "The exposed image must select this exact retained print")
        XCTAssertTrue(done.isHittable)
        done.tap()
        try revealPrint(tile, app: app)
    }

    @MainActor private func assertEmptyMachines(_ app: XCUIApplication) throws {
        XCTAssertTrue(app.navigateToDestination("Machines", shortcut: "5"))
        let empty = app.staticTexts["No machines yet"]
        _ = try XCTUnwrap(empty.waitForExistence(timeout: 5) ? empty : nil, "Controlled fixture requires empty Machines: \(app.debugDescription)")
        XCTAssertFalse(app.descendants(matching: .any).matching(NSPredicate(format: "identifier BEGINSWITH 'machine-card-'")).firstMatch.exists)
        attach(app, name: "Empty Machines before controlled offline fixtures")
    }

    @MainActor private func machineIdentity(port: UInt16, app: XCUIApplication) throws -> String {
        let card = app.descendants(matching: .any).matching(NSPredicate(format:
            "identifier BEGINSWITH 'machine-card-' AND label MATCHES %@", ".*127\\.0\\.0\\.1:\(port)([^0-9].*|$)")).firstMatch
        for _ in 0..<8 where !card.exists { app.scrollViews.firstMatch.swipeUp() }
        let found = try XCTUnwrap(card.exists ? card : nil, "The paired endpoint must expose its host identity: \(app.debugDescription)")
        let identity = String(found.identifier.dropFirst("machine-card-".count))
        XCTAssertNotNil(UUID(uuidString: identity))
        return identity
    }

    @MainActor private func settle(_ element: XCUIElement) {
        var previous = CGRect.null
        let stable = XCTNSPredicateExpectation(predicate: NSPredicate { _, _ in
            let current = element.frame
            defer { previous = current }
            return element.exists && current == previous
        }, object: nil)
        XCTAssertEqual(XCTWaiter.wait(for: [stable], timeout: 5), .completed)
    }

    @MainActor private func auditSelect(_ app: XCUIApplication, stage: String) throws {
        let select = app.navigationBars.buttons["Select"].firstMatch
        XCTAssertTrue(select.waitForExistence(timeout: 5))
        XCTAssertTrue(select.isEnabled); XCTAssertTrue(select.isHittable)
        attach(app, name: stage + " Select before contrast audit")
        try auditContrast(select, app: app, stage: stage + " Select")
    }

    @MainActor private func auditContrast(_ scope: XCUIElement, app: XCUIApplication, stage: String) throws {
        try app.performAccessibilityAudit(for: .contrast) { issue in
            guard let element = issue.element else { return false }
            let belongs = element.elementType == scope.elementType && element.label == scope.label && element.frame == scope.frame
                || scope.descendants(matching: element.elementType)
                    .matching(NSPredicate(format: "label == %@", element.label)).allElementsBoundByIndex
                    .contains { $0.frame == element.frame }
            guard belongs else { return true }
            let hierarchy = XCTAttachment(string: app.debugDescription)
            hierarchy.name = stage + " Select contrast failure hierarchy"; hierarchy.lifetime = .keepAlways; self.add(hierarchy)
            self.attach(app, name: stage + " Select contrast failure")
            XCTFail("\(stage): \(issue.compactDescription), \(element.label) at \(element.frame)")
            return true
        }
    }

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

    @MainActor func testRetainedSourceReuseAppearsInWellAndCanBeRemoved() async throws {
        continueAfterFailure = false
        let app = try await populatedLibrary(retainedMediaFixture: true)
        let print = fixturePrint(in: app)
        XCTAssertTrue(print.waitForExistence(timeout: 10))
        print.press(forDuration: 1)
        app.buttons["Use These Settings"].firstMatch.tap()
        XCTAssertTrue(app.navigateToDestination("Generate", shortcut: "1"))
        let source = app.buttons["Start from"].firstMatch
        XCTAssertTrue(source.waitForExistence(timeout: 10), "The private retained source must appear even without an output metadata source marker")
        let form = app.scrollViews["phone-generate-form"]
        for _ in 0..<6 where !source.isHittable && form.exists { form.swipeUp() }
        XCTAssertTrue(source.isHittable)
        XCTAssertFalse(app.buttons["Start from, empty"].firstMatch.exists)
        attach(app, name: "Retained source restored into ordinary well")
        source.tap()
        let remove = app.buttons["Remove"].firstMatch
        XCTAssertTrue(remove.waitForExistence(timeout: 5))
        remove.tap()
        XCTAssertTrue(app.buttons["Start from, empty"].firstMatch.waitForExistence(timeout: 5))
        attach(app, name: "Retained source explicitly removed")
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

    @MainActor private func populatedLibrary(offlineCopy: Bool = false, retainedMediaFixture: Bool = false) async throws -> XCUIApplication {
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
        let machine = try FixtureMachine(galleryPrints: 6, collectionFixture: true, retainedMediaFixture: retainedMediaFixture)
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
