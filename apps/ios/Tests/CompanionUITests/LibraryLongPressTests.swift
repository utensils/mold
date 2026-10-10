import XCTest
import UIKit

/// Context menus and drag previews are separately hosted by UIKit. Exercise
/// their real long-press boundary with a populated library, not just a tap.
final class LibraryLongPressTests: XCTestCase {
    override func setUp() { super.setUp(); acceptCompanionPermissions() }

    @MainActor func testNewClipBadgeAndMachinePlaybackPlacement() async throws {
        continueAfterFailure = false
        let machine = try FixtureMachine(galleryPrints: 3, mixedMedia: true)
        let other = try FixtureMachine(galleryPrints: 1, galleryID: "other")
        let port = try await machine.start()
        let otherPort = try await other.start()
        let app = XCUIApplication()
        defer { app.terminate(); machine.stop(); other.stop() }
        cleanUpFixture(machine, port: port, app: app)
        cleanUpFixture(other, port: otherPort, app: app)
        app.launch()
        pair(port, in: app, name: "Long media workstation East")
        pair(otherPort, in: app, name: "Other workstation")
        XCTAssertTrue(app.navigateToDestination("Library", shortcut: "2"))
        XCTAssertTrue(fixturePrint(in: app).waitForExistence(timeout: 10))
        XCTAssertFalse(app.buttons.matching(NSPredicate(format: "label BEGINSWITH 'New, '")).firstMatch.exists)
        XCTAssertTrue(app.navigateToDestination("Generate", shortcut: "1"))
        await machine.addNewClip()
        XCTAssertTrue(app.navigateToDestination("Library", shortcut: "2"))
        app.swipeDown()
        let fresh = app.buttons.matching(NSPredicate(format: "label BEGINSWITH 'New, New clip'")).firstMatch
        XCTAssertTrue(fresh.waitForExistence(timeout: 15))
        attach(app, name: "New clip and separated machine playback badges")
        fresh.tap()
        XCTAssertTrue(app.buttons["Info"].firstMatch.waitForExistence(timeout: 5))
        app.navigationBars.buttons.firstMatch.tap()
        XCTAssertTrue(fresh.waitForNonExistence(timeout: 5), "Viewing immediately removes this visit's New badge")
        XCTAssertTrue(app.navigateToDestination("Generate", shortcut: "1"))
        XCTAssertTrue(app.navigateToDestination("Library", shortcut: "2"))
        XCTAssertFalse(fresh.exists, "Next Library visit clears the badge")
        attach(app, name: "Viewed clip stays seen on next visit")
    }

    @MainActor func testAppIconCountsNewMediaAndViewingClearsCurrentVisitImmediately() async throws {
        continueAfterFailure = false
        let identity = UUID().uuidString
        let machine = try FixtureMachine(galleryPrints: 3, galleryID: identity, mixedMedia: true)
        let port = try await machine.start()
        let app = XCUIApplication()
        defer { app.terminate(); machine.stop() }
        cleanUpFixture(machine, port: port, app: app)
        app.launch()
        pair(port, in: app, name: "Badge workstation")
        XCTAssertTrue(app.navigateToDestination("Library", shortcut: "2"))
        if !app.navigationBars["All Prints"].exists { app.chooseLibraryShelf("All Prints") }
        XCTAssertTrue(fixturePrint(in: app).waitForExistence(timeout: 10))
        XCTAssertTrue(app.navigateToDestination("Generate", shortcut: "1"))
        await machine.addNewMedia(filename: "new-\(identity).png", title: "Unread picture")
        await machine.addNewMedia(filename: "new-\(identity).mp4", title: "Unread video", clip: true)
        // Foreground refresh from a route return without opening Library.
        XCUIDevice.shared.press(.home)
        app.activate()
        XCTAssertTrue(app.navigateToDestination("Generate", shortcut: "1"))
        // Wait for refresh/permission before checking the real SpringBoard icon.
        try await Task.sleep(for: .seconds(3))
        XCUIDevice.shared.press(.home)
        let springboard = XCUIApplication(bundleIdentifier: "com.apple.springboard")
        let icon = springboard.icons.matching(NSPredicate(format: "label BEGINSWITH 'Mold Studio'")).firstMatch
        XCTAssertTrue(icon.waitForExistence(timeout: 5))
        let two = NSPredicate { _, _ in
            (icon.value as? String)?.contains("2") == true || icon.label.contains("2")
        }
        await fulfillment(of: [expectation(for: two, evaluatedWith: icon)], timeout: 10)
        XCTAssertTrue(two.evaluate(with: icon), icon.debugDescription)
        attach(springboard, name: "Home Screen counts two new media")
        app.activate()
        XCTAssertTrue(app.navigateToDestination("Library", shortcut: "2"))
        let freshPicture = app.buttons.matching(NSPredicate(format: "label BEGINSWITH 'New, Unread picture'")).firstMatch
        let freshVideo = app.buttons.matching(NSPredicate(format: "label BEGINSWITH 'New, Unread video'")).firstMatch
        XCTAssertTrue(freshPicture.waitForExistence(timeout: 10))
        XCTAssertTrue(freshVideo.waitForExistence(timeout: 5))
        freshPicture.tap()
        XCTAssertTrue(app.buttons["Info"].firstMatch.waitForExistence(timeout: 5))
        app.navigationBars.buttons.firstMatch.tap()
        XCTAssertTrue(freshPicture.waitForNonExistence(timeout: 5))
        XCTAssertTrue(freshVideo.exists, "Preloaded neighboring pages are not viewed")
        attach(app, name: "Only viewed picture loses New in current visit")
        freshVideo.tap()
        XCTAssertTrue(app.buttons["Info"].firstMatch.waitForExistence(timeout: 5))
        app.navigationBars.buttons.firstMatch.tap()
        XCTAssertTrue(freshVideo.waitForNonExistence(timeout: 5))
        XCUIDevice.shared.press(.home)
        let noBadge = NSPredicate { _, _ in
            let value = icon.value as? String ?? ""
            return (value.isEmpty || value == "0") && icon.label == "Mold Studio"
        }
        await fulfillment(of: [expectation(for: noBadge, evaluatedWith: icon)], timeout: 5)
        XCTAssertTrue(noBadge.evaluate(with: icon), icon.debugDescription)
        attach(springboard, name: "Home Screen badge cleared after Library visit")
        app.activate()
        XCTAssertTrue(app.navigateToDestination("Generate", shortcut: "1"))
        XCTAssertTrue(app.navigateToDestination("Library", shortcut: "2"))
        XCTAssertFalse(freshPicture.exists)
        XCTAssertFalse(freshVideo.exists)
        XCTAssertTrue(app.navigateToDestination("Generate", shortcut: "1"))
        await machine.addNewMedia(filename: "relaunch-\(identity).png", title: "Relaunch picture")
        XCUIDevice.shared.press(.home)
        try await Task.sleep(for: .seconds(1))
        app.activate()
        try await Task.sleep(for: .seconds(3))
        XCUIDevice.shared.press(.home)
        let one = NSPredicate { _, _ in (icon.value as? String)?.contains("1") == true || icon.label.contains("1 notification") }
        await fulfillment(of: [expectation(for: one, evaluatedWith: icon)], timeout: 10)
        app.terminate(); app.launch()
        try await Task.sleep(for: .seconds(3))
        XCUIDevice.shared.press(.home)
        await fulfillment(of: [expectation(for: one, evaluatedWith: icon)], timeout: 10)
        attach(springboard, name: "Home Screen count survives app relaunch")
        app.activate()
        XCTAssertTrue(app.navigateToDestination("Library", shortcut: "2"))
        XCTAssertFalse(app.buttons.matching(NSPredicate(format: "label BEGINSWITH 'New, Relaunch picture'")).firstMatch.exists,
                       "Session-only first-visit baseline remains unchanged after relaunch")
        XCUIDevice.shared.press(.home)
        await fulfillment(of: [expectation(for: noBadge, evaluatedWith: icon)], timeout: 5)
        app.activate()
        XCTAssertTrue(machine.generationRequests.isEmpty)
    }

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
        let presentingTabs = try libraryNavigationChrome(app)
        attach(app, name: stage + " Presenting native tabs before details")
        status.tap()
        attach(app, name: stage + " Initial offline details")
        let candidate = app.collectionViews["offline-library-details"].firstMatch
        let details = try XCTUnwrap(candidate.waitForExistence(timeout: 5) ? candidate : nil, "The details sheet must own its native List: \(app.debugDescription)")
        let heading = app.navigationBars["Saved Prints"]
        let title = try XCTUnwrap(heading.waitForExistence(timeout: 5) ? heading : nil, "The details heading must be presented: \(app.debugDescription)")
        // The native List and navigation bar are siblings. Select their
        // innermost actual common owner, not the dimmed Library behind it.
        let owners = app.otherElements.containing(.collectionView, identifier: "offline-library-details")
            .containing(.navigationBar, identifier: "Saved Prints")
            .containing(.button, identifier: "offline-library-done").allElementsBoundByIndex
        let owner = try XCTUnwrap(owners.last, "The presented sheet must own its List, navigation bar and footer: \(app.debugDescription)")
        attach(app, name: stage + " Details owning navigation before tab visibility assertion")
        XCTAssertTrue(presentingTabs.waitForNonExistence(timeout: 5),
                      "The modal must remove presenting native tab chrome: \(app.debugDescription)")
        XCTAssertFalse(owner.frame.isEmpty)
        let explanation = details.staticTexts["This device's saved prints remain available."].firstMatch
        let section = details.staticTexts.matching(NSPredicate(format: "label ==[c] %@", "Unavailable Machines")).firstMatch
        // Pixel sampling always precedes the auditor's private size cycling,
        // including samples of rows revealed later in this scrollable List.
        attach(app, name: stage + " Full details sheet before contrast audit")
        let contrastLabels = Set(names + ["Saved Prints", "This device's saved prints remain available.", "Unavailable Machines", "Done"])
        try assertDetailsContrastInventory(owner, labels: contrastLabels)
        XCTAssertTrue(owner.frame.contains(title.frame))
        let titleText = title.staticTexts["Saved Prints"]
        XCTAssertTrue(titleText.exists)
        XCTAssertTrue(title.frame.contains(titleText.frame))
        // Audit every semantic label only once its complete bounds are exposed.
        // An initial List row can extend beneath the opaque pinned footer.
        try auditContrast(title, app: app, stage: stage + " details title")
        let information = [(explanation, "explanation"), (section, "section heading")]
        var requestedExplanationSize = CGSize.zero
        for (text, label) in information {
            try revealDetailsText(text, details: details, title: title, app: app)
            try auditContrast(text, app: app, stage: stage + " details " + label)
            if label == "explanation" { requestedExplanationSize = text.frame.size }
        }
        XCTAssertEqual(Set(identities).count, names.count, "Duplicate names must keep distinct host identities")
        for (index, name) in names.enumerated() {
            let text = details.staticTexts["offline-library-host-" + identities[index]].firstMatch
            for _ in 0..<12 {
                let visible = detailsViewport(details, title: title, app: app)
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
            let viewport = detailsViewport(details, title: title, app: app)
            XCTAssertTrue(viewport.contains(text.frame), "Full host name \(text.frame) must fit \(viewport)")
            try assertDetailsContrastInventory(owner, labels: contrastLabels)
            try auditContrast(text, app: app, stage: stage + " name " + identities[index])
            attach(app, name: stage + " Full offline name " + identities[index])
        }
        let done = owner.buttons["offline-library-done"]
        XCTAssertTrue(done.exists)
        XCTAssertTrue(app.frame.contains(done.frame))
        let footer = app.descendants(matching: .any)["offline-library-footer"].firstMatch
        XCTAssertTrue(footer.exists)
        XCTAssertTrue(owner.frame.contains(footer.frame))
        XCTAssertTrue(footer.frame.contains(done.frame))
        XCTAssertGreaterThanOrEqual(done.frame.height, 44)
        XCTAssertTrue(done.isHittable)
        try auditContrast(done, app: app, stage: stage + " details Done")
        let requestedDoneHeight = done.frame.height
        let requestedTitleHeight = title.frame.height
        let noncontrast: XCUIAccessibilityAuditType = [.dynamicType, .textClipped, .hitRegion, .sufficientElementDescription]
        // Each private size sweep gets a fresh native presentation at the
        // requested launch category. Preserve the original nil-element failure
        // evidence while avoiding state carried from an earlier private sweep.
        attach(app, name: stage + " Details before noncontrast gates")
        var fresh = try reopenDetails(app, stage: stage + " Done", doneHeight: requestedDoneHeight, titleHeight: requestedTitleHeight, explanationSize: requestedExplanationSize, contrastLabels: contrastLabels)
        try auditAccessibility(fresh.owner.buttons["offline-library-done"], app: app,
                               stage: stage + " details Done", types: noncontrast)
        fresh = try reopenDetails(app, stage: stage + " full sheet", doneHeight: requestedDoneHeight, titleHeight: requestedTitleHeight, explanationSize: requestedExplanationSize, contrastLabels: contrastLabels)
        try auditAccessibility(fresh.owner, app: app, stage: stage + " details sheet", types: noncontrast)
        for label in ["explanation", "section heading"] {
            fresh = try reopenDetails(app, stage: stage + " " + label, doneHeight: requestedDoneHeight, titleHeight: requestedTitleHeight, explanationSize: requestedExplanationSize, contrastLabels: contrastLabels)
            let text = label == "explanation"
                ? fresh.details.staticTexts["This device's saved prints remain available."].firstMatch
                : fresh.details.staticTexts.matching(NSPredicate(format: "label ==[c] %@", "Unavailable Machines")).firstMatch
            try revealDetailsText(text, details: fresh.details, title: fresh.title, app: app)
            try auditAccessibility(text, app: app, stage: stage + " details " + label, types: noncontrast)
        }
        let finalDone = fresh.owner.buttons["offline-library-done"]
        XCTAssertTrue(finalDone.isHittable)
        finalDone.tap()
        XCTAssertTrue(details.waitForNonExistence(timeout: 5))
        _ = try libraryNavigationChrome(app)
        try revealPrint(tile, app: app)
        try auditSelect(app, stage: stage + " after details dismissal")
        try selectRetainedPrint(tile, app: app)
    }

    @MainActor private func reopenDetails(_ app: XCUIApplication, stage: String, doneHeight: CGFloat, titleHeight: CGFloat, explanationSize: CGSize, contrastLabels: Set<String>) throws
        -> (details: XCUIElement, title: XCUIElement, owner: XCUIElement) {
        let previous = app.collectionViews["offline-library-details"].firstMatch
        let done = app.buttons["offline-library-done"].firstMatch
        XCTAssertTrue(done.exists && done.isHittable, "Each native sheet must dismiss through its visible Done")
        done.tap()
        XCTAssertTrue(previous.waitForNonExistence(timeout: 5))
        let presentingTabs = try libraryNavigationChrome(app)
        let status = app.buttons["offline-library-status"]
        XCTAssertTrue(status.exists && status.isHittable)
        status.tap()
        let details = app.collectionViews["offline-library-details"].firstMatch
        let title = app.navigationBars["Saved Prints"]
        XCTAssertTrue(details.waitForExistence(timeout: 5))
        XCTAssertTrue(title.waitForExistence(timeout: 5))
        XCTAssertTrue(presentingTabs.waitForNonExistence(timeout: 5), "Every details presentation must remove native tab chrome")
        let owners = app.otherElements.containing(.collectionView, identifier: "offline-library-details")
            .containing(.navigationBar, identifier: "Saved Prints")
            .containing(.button, identifier: "offline-library-done").allElementsBoundByIndex
        let owner = try XCTUnwrap(owners.last, "The fresh sheet must own its List, title and footer: \(app.debugDescription)")
        let footer = owner.descendants(matching: .any)["offline-library-footer"].firstMatch
        let freshDone = owner.buttons["offline-library-done"]
        settle(freshDone)
        XCTAssertFalse(owner.frame.isEmpty)
        XCTAssertTrue(app.frame.contains(owner.frame))
        XCTAssertTrue(owner.frame.contains(title.frame))
        XCTAssertTrue(footer.exists)
        XCTAssertTrue(owner.frame.contains(footer.frame))
        XCTAssertTrue(footer.frame.contains(freshDone.frame))
        XCTAssertTrue(app.frame.contains(freshDone.frame))
        XCTAssertGreaterThanOrEqual(freshDone.frame.height, 44)
        XCTAssertEqual(freshDone.frame.height, doneHeight, "Each presentation must restore the requested-size native action")
        XCTAssertEqual(title.frame.height, titleHeight, "Each presentation must restore the requested-size native title")
        XCTAssertTrue(freshDone.isHittable)
        let explanation = details.staticTexts["This device's saved prints remain available."].firstMatch
        try revealDetailsText(explanation, details: details, title: title, app: app)
        XCTAssertEqual(explanation.frame.size, explanationSize,
                       "The fresh text must restore the complete requested-size geometry before a private sweep")
        try assertDetailsContrastInventory(owner, labels: contrastLabels)
        attach(app, name: stage + " Fresh native details before private sweep")
        return (details, title, owner)
    }

    @MainActor private func libraryNavigationChrome(_ app: XCUIApplication) throws -> XCUIElement {
        let library = app.buttons.matching(NSPredicate(format: "label == %@", "Library")).firstMatch
        XCTAssertTrue(library.waitForExistence(timeout: 5), "Dismissal must restore native Library navigation")
        let tabBar = app.tabBars.firstMatch
        let chrome: XCUIElement
        if tabBar.exists {
            chrome = tabBar
        } else {
            // iPad exports its native floating tabs as an Other container.
            let owners = app.otherElements.containing(.button, identifier: "wand.and.sparkles")
                .containing(.button, identifier: "square.grid.2x2")
                .containing(.button, identifier: "list.bullet")
                .containing(.button, identifier: "desktopcomputer").allElementsBoundByIndex
            chrome = try XCTUnwrap(owners.last, "Native floating tab ownership must be observable: \(app.debugDescription)")
        }
        XCTAssertFalse(chrome.frame.isEmpty)
        XCTAssertTrue(app.frame.contains(chrome.frame))
        XCTAssertLessThan(chrome.frame.height, app.frame.height, "The native tab owner must be distinct from the app root")
        XCTAssertTrue(chrome.frame.contains(library.frame))
        XCTAssertTrue(library.isSelected)
        XCTAssertTrue(library.isHittable)
        return chrome
    }

    @MainActor private func assertDetailsContrastInventory(_ owner: XCUIElement, labels: Set<String>) throws {
        for text in owner.staticTexts.allElementsBoundByIndex {
            let covered = labels.contains(text.label) || text.label.caseInsensitiveCompare("Unavailable Machines") == .orderedSame
            XCTAssertTrue(covered, "Every semantic details label must have explicit full-visible contrast coverage: \(text.label)")
        }
        for button in owner.buttons.allElementsBoundByIndex {
            XCTAssertEqual(button.identifier, "offline-library-done", "Every details action must have explicit contrast coverage")
        }
        XCTAssertEqual(owner.textFields.count, 0, "New details inputs require explicit contrast coverage")
        XCTAssertEqual(owner.textViews.count, 0, "New details text requires explicit contrast coverage")
        XCTAssertEqual(owner.links.count, 0, "New details links require explicit contrast coverage")
        XCTAssertEqual(owner.switches.count, 0, "New details switches require explicit contrast coverage")
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
        let baseline = tile.frame
        // AX can export a fraction of an image beyond the fixed native grid.
        // Vertical scrolling can expose its entire grid intersection, but
        // cannot move that horizontal interval onto the screen.
        let reachable = baseline.intersection(CGRect(x: viewport.minX, y: baseline.minY,
                                                    width: viewport.width, height: baseline.height))
        func fullyExposed() -> Bool {
            if tile.frame.height <= viewport.height, tile.frame.width <= viewport.width,
               tile.frame.minX >= viewport.minX, tile.frame.maxX <= viewport.maxX {
                return viewport.contains(tile.frame)
            }
            let visible = viewport.intersection(tile.frame)
            return visible.height >= min(tile.frame.height, viewport.height) && visible.width >= reachable.width
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
            XCTAssertEqual(tile.frame.minX, baseline.minX, "A vertical reveal must preserve the measured horizontal image interval")
            XCTAssertEqual(tile.frame.maxX, baseline.maxX)
        }
        let visible = viewport.intersection(tile.frame)
        let geometry = XCTAttachment(string: "Baseline \(baseline); reachable horizontal interval \(reachable.minX)...\(reachable.maxX); final \(tile.frame); viewport \(viewport); visible \(visible)")
        geometry.name = "Retained photo maximum exposure geometry"; geometry.lifetime = .keepAlways; add(geometry)
        XCTAssertTrue(fullyExposed(), "Retained print \(tile.frame) must expose its full available image area in \(viewport), baseline \(baseline), reachable \(reachable)")
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
        // Hosted AX snapshots can take several seconds while selection
        // rebuilds the tile. Allow two observations without relaxing the
        // requirement that its frame is exactly unchanged.
        XCTAssertEqual(XCTWaiter.wait(for: [stable], timeout: 15), .completed)
    }

    @MainActor private func auditSelect(_ app: XCUIApplication, stage: String) throws {
        let select = app.navigationBars.buttons["Select"].firstMatch
        XCTAssertTrue(select.waitForExistence(timeout: 5))
        XCTAssertTrue(select.isEnabled); XCTAssertTrue(select.isHittable)
        attach(app, name: stage + " Select before contrast audit")
        try auditContrast(select, app: app, stage: stage + " Select")
    }

    @MainActor private func auditContrast(_ scope: XCUIElement, app: XCUIApplication, stage: String) throws {
        try auditAccessibility(scope, app: app, stage: stage, types: .contrast)
    }

    @MainActor private func auditAccessibility(_ scope: XCUIElement, app: XCUIApplication, stage: String,
                                              types: XCUIAccessibilityAuditType) throws {
        _ = try XCTUnwrap(scope.exists && !scope.frame.isEmpty ? scope : nil,
                          "An owning-region audit requires its actual visible native element")
        try app.performAccessibilityAudit(for: types) { issue in
            guard let element = issue.element else {
                let diagnostic = XCTAttachment(string: "\(stage): \(issue.compactDescription)\n\(issue.detailedDescription)\n\(app.debugDescription)")
                diagnostic.name = stage + " Accessibility issue without element"
                diagnostic.lifetime = .keepAlways; self.add(diagnostic)
                self.attach(app, name: stage + " Accessibility issue without element")
                return false
            }
            let belongs = element.elementType == scope.elementType && element.label == scope.label && element.frame == scope.frame
                || scope.descendants(matching: element.elementType)
                    .matching(NSPredicate(format: "label == %@", element.label)).allElementsBoundByIndex
                    .contains { $0.frame == element.frame }
            guard belongs else { return true }
            let hierarchy = XCTAttachment(string: app.debugDescription)
            hierarchy.name = stage + " Accessibility failure hierarchy"; hierarchy.lifetime = .keepAlways; self.add(hierarchy)
            self.attach(app, name: stage + " Accessibility failure")
            XCTFail("\(stage): \(issue.compactDescription), \(element.label) at \(element.frame). \(issue.detailedDescription)")
            return true
        }
    }

    @MainActor private func detailsViewport(_ details: XCUIElement, title: XCUIElement, app: XCUIApplication) -> CGRect {
        let frame = details.frame.intersection(app.frame)
        let footer = app.descendants(matching: .any)["offline-library-footer"].firstMatch
        XCTAssertTrue(footer.exists, "The details must have an opaque, unobstructed Done footer")
        let top = max(frame.minY, title.frame.maxY)
        let bottom = min(frame.maxY, footer.frame.minY)
        return CGRect(x: frame.minX, y: top, width: frame.width, height: max(0, bottom - top))
    }

    @MainActor private func revealDetailsText(_ text: XCUIElement, details: XCUIElement,
                                            title: XCUIElement, app: XCUIApplication) throws {
        let viewport = detailsViewport(details, title: title, app: app)
        for _ in 0..<12 {
            if text.exists, viewport.contains(text.frame) { break }
            if !text.exists { details.swipeDown(); continue }
            XCTAssertLessThanOrEqual(text.frame.height, viewport.height, "Sheet text must fit at the requested text size")
            let delta = text.frame.midY - viewport.midY
            let distance = min(abs(delta), viewport.height * 0.4)
            let direction: CGFloat = delta > 0 ? 1 : -1
            let origin = app.coordinate(withNormalizedOffset: .zero)
            let x = viewport.minX + viewport.width * 0.03
            origin.withOffset(CGVector(dx: x, dy: viewport.midY + direction * distance / 2))
                .press(forDuration: 0.1, thenDragTo: origin.withOffset(CGVector(dx: x, dy: viewport.midY - direction * distance / 2)),
                       withVelocity: .slow, thenHoldForDuration: 0.3)
            settle(text)
        }
        _ = try XCTUnwrap(text.exists ? text : nil, "Every sheet explanation and section heading must be reachable: \(app.debugDescription)")
        settle(text)
        XCTAssertTrue(viewport.contains(text.frame), "The complete sheet text \(text.label), \(text.frame), must fit \(viewport)")
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

    @MainActor func testRetainedOpeningAndClosingFramesAreVisibleAndRemovable() async throws {
        continueAfterFailure = false
        let machine = try FixtureMachine(referenceFixture: true, galleryPrints: 1, retainedFrameFixture: true)
        let port = try await machine.start()
        let app = XCUIApplication()
        defer { app.terminate(); machine.stop() }
        cleanUpFixture(machine, port: port, app: app)
        app.launch()
        pair(port, in: app, name: "Frame reuse workstation")
        XCTAssertTrue(app.navigateToDestination("Library", shortcut: "2"))
        let print = fixturePrint(in: app)
        XCTAssertTrue(print.waitForExistence(timeout: 10))
        print.tap()
        let more = app.buttons["More"].firstMatch
        XCTAssertTrue(more.waitForExistence(timeout: 5)); more.tap()
        app.buttons["Use These Settings"].firstMatch.tap()
        XCTAssertTrue(app.navigateToDestination("Generate", shortcut: "1"))
        let first = app.buttons["First frame"].firstMatch
        let last = app.buttons["Last frame"].firstMatch
        XCTAssertTrue(first.waitForExistence(timeout: 15), app.debugDescription)
        XCTAssertTrue(last.waitForExistence(timeout: 15), app.debugDescription)
        let form = app.scrollViews["phone-generate-form"]
        for _ in 0..<6 where !last.isHittable && form.exists { form.swipeUp() }
        XCTAssertTrue(last.isHittable)
        XCTAssertFalse(app.buttons["Last frame, empty"].exists)
        attach(app, name: "Reuse restores both retained boundary frames")
        last.tap()
        let remove = app.buttons["Remove"].firstMatch
        XCTAssertTrue(remove.waitForExistence(timeout: 5)); remove.tap()
        XCTAssertTrue(app.buttons["Last frame, empty"].waitForExistence(timeout: 5))
        XCTAssertTrue(app.buttons["First frame"].exists)
        XCTAssertTrue(machine.generationRequests.isEmpty, "Reuse and removal must never generate")
        attach(app, name: "Closing frame removal remains explicit")
    }

    @MainActor func testRetainedSourceReuseAppearsInWellAndCanBeRemoved() async throws {
        continueAfterFailure = false
        let app = try await populatedLibrary(retainedMediaFixture: true)
        // Start a new scene before the Library handoff, rather than relying
        // on a Generate view retained by an earlier test's navigation.
        app.terminate()
        app.launch()
        XCTAssertTrue(app.navigateToDestination("Library", shortcut: "2"))
        app.chooseLibraryShelf("UAT Drafts")
        XCTAssertTrue(app.navigationBars["UAT Drafts"].waitForExistence(timeout: 5))
        let print = fixturePrint(in: app)
        XCTAssertTrue(print.waitForExistence(timeout: 10))
        print.tap()
        let more = app.buttons["More"].firstMatch
        XCTAssertTrue(more.waitForExistence(timeout: 5))
        more.tap()
        app.buttons["Use These Settings"].firstMatch.tap()
        XCTAssertTrue(app.navigateToDestination("Generate", shortcut: "1"))
        let prompt = app.descendants(matching: .any)["generation-prompt"].firstMatch
        let restoredPrompt = XCTNSPredicateExpectation(predicate: NSPredicate(format: "value == %@", "Fixture 0"), object: prompt)
        XCTAssertEqual(XCTWaiter.wait(for: [restoredPrompt], timeout: 5), .completed,
                       "The Library handoff must restore its prompt before probing private reference media: \(app.debugDescription)")
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
