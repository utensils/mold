import XCTest

/// Populated screens need populated regression tests: first-run empty states
/// cannot reveal compressed option controls or a misleading model search.
final class PopulatedGenerationTests: XCTestCase {
    override func setUp() {
        super.setUp()
        acceptCompanionPermissions()
    }

    @MainActor func testVisibleModelUnloadingKeepsInstalledFiles() async throws {
        try await modelMemory(size: "UICTContentSizeCategoryL")
    }

    @MainActor func testModelMemoryExtraSmall() async throws {
        try await modelMemory(size: "UICTContentSizeCategoryXS")
    }

    @MainActor func testModelMemoryAccessibilityXXXL() async throws {
        try await modelMemory(size: "UICTContentSizeCategoryAccessibilityXXXL")
    }

    @MainActor private func modelMemory(size: String) async throws {
        continueAfterFailure = false
        let machine = try FixtureMachine(loadedModels: true)
        let port = try await machine.start()
        let app = XCUIApplication()
        defer { app.terminate(); machine.stop() }
        cleanUpFixture(machine, port: port, app: app)
        app.launchArguments = ["-UIPreferredContentSizeCategoryName", size]
        app.launch()
        XCTAssertTrue(app.navigateToDestination("Machines", shortcut: "5"))
        app.buttons["Add a Machine"].firstMatch.tap()
        app.buttons.matching(NSPredicate(format: "label BEGINSWITH 'Enter an Address'")).firstMatch.tap()
        app.textFields["machine-name"].tap()
        app.textFields["machine-name"].typeText("Unload Fixture")
        app.textFields["machine-address"].tap()
        app.textFields["machine-address"].typeText("127.0.0.1:\(port)")
        app.buttons["Add"].firstMatch.tap()
        let row = app.buttons.matching(NSPredicate(format:
            "identifier BEGINSWITH 'machine-card-' AND label MATCHES %@",
            ".*127\\.0\\.0\\.1:\(port)([^0-9].*|$)")).firstMatch
        for _ in 0..<15 where !row.exists || !row.isHittable { app.swipeUp() }
        guard row.waitForExistence(timeout: 10) else { XCTFail(app.debugDescription); return }
        row.tap()
        let modelsLink = app.descendants(matching: .any)["machine-models"].firstMatch
        guard modelsLink.waitForExistence(timeout: 5) else { XCTFail(app.debugDescription); return }
        for _ in 0..<5 where !modelsLink.isHittable { app.swipeUp() }
        let hadTabBar = app.tabBars.firstMatch.exists
        modelsLink.tap()
        let pane = app.buttons["models-pane"]
        if pane.waitForExistence(timeout: 5) {
            pane.tap()
            app.buttons["Installed"].firstMatch.tap()
        }
        if app.frame.width > 600 {
            XCTAssertFalse(app.tabBars.firstMatch.exists, "Per-machine Models must hide iPad floating tab chrome")
        }
        let one = app.buttons["unload-model-flux-dev:q4"]
        guard one.waitForExistence(timeout: 10) else { XCTFail(app.debugDescription); return }
        revealMemoryControl(one, in: app)
        XCTAssertGreaterThanOrEqual(one.frame.height, 44)
        one.tap()
        XCTAssertTrue(one.waitForNonExistence(timeout: 10))
        let remaining = app.buttons["unload-model-ltx-2.5-22b-distilled:bf16"]
        XCTAssertTrue(remaining.exists)
        let unloadAll = app.buttons["unload-all-models"]
        revealMemoryControl(unloadAll, in: app)
        XCTAssertGreaterThanOrEqual(unloadAll.frame.height, 44)
        unloadAll.tap()
        XCTAssertTrue(remaining.waitForNonExistence(timeout: 10))
        let retained = app.staticTexts["flux-dev:q4"].firstMatch
        revealMemoryControl(retained, in: app)
        XCTAssertTrue(retained.exists, "Unloading must retain installed inventory")
        await machine.restoreResidentModels()
        app.navigationBars.buttons.firstMatch.tap()
        if hadTabBar { XCTAssertTrue(app.tabBars.firstMatch.waitForExistence(timeout: 5), "Back must restore tab chrome") }
        XCTAssertTrue(modelsLink.waitForExistence(timeout: 5))
        for _ in 0..<5 where !modelsLink.isHittable { app.swipeUp() }
        modelsLink.tap()
        guard one.waitForExistence(timeout: 10) else { XCTFail(app.debugDescription); return }
        revealMemoryControl(one, in: app)
        revealMemoryControl(unloadAll, in: app)
        // Contrast samples settled pixels before Dynamic Type temporarily resizes the hierarchy.
        try app.performAccessibilityAudit(for: .contrast) { issue in
            guard let element = issue.element else { return false }
            // Scroll content under the system glass is covered by the existing
            // shell audit exemption. Never exempt controls that belong to it.
            let bar = app.tabBars.firstMatch
            let belongsToBar = bar.exists && ((element.elementType == .tabBar && element.frame == bar.frame)
                || bar.descendants(matching: element.elementType)
                    .matching(NSPredicate(format: "label == %@", element.label))
                    .allElementsBoundByIndex.contains { $0.frame == element.frame })
            let isListContent = app.collectionViews.allElementsBoundByIndex.contains { list in
                list.descendants(matching: element.elementType)
                    .matching(NSPredicate(format: "label == %@", element.label))
                    .allElementsBoundByIndex.contains { $0.frame == element.frame }
            }
            var edge = app.frame.maxY
            if bar.exists { edge = min(edge, bar.frame.minY) }
            let dimming = app.images["AdditionalDimmingOverlay"].firstMatch
            if dimming.exists { edge = min(edge, dimming.frame.minY) }
            if isListContent, !belongsToBar, element.frame.maxY > edge { return true }
            XCTFail("Model memory at \(size): \(issue.compactDescription), \(element.label) at \(element.frame)")
            return true
        }
        try app.performAccessibilityAudit(for: [.dynamicType, .textClipped, .hitRegion, .sufficientElementDescription, .trait, .elementDetection]) { issue in
            let element = issue.element.map { "\($0.elementType) \($0.label) at \($0.frame)" } ?? "unnamed element"
            XCTFail("Model memory at \(size): \(issue.compactDescription), \(element) [\(issue.detailedDescription)]")
            return true
        }
        attach(app)
    }

    @MainActor private func revealMemoryControl(_ element: XCUIElement, in app: XCUIApplication) {
        for _ in 0..<12 {
            let top = app.navigationBars.firstMatch.frame.maxY
            let bottom = app.tabBars.firstMatch.exists ? app.tabBars.firstMatch.frame.minY : app.frame.maxY
            if element.exists, element.isHittable, element.frame.minY >= top, element.frame.maxY <= bottom { return }
            if element.exists, element.frame.minY < top { app.swipeDown() }
            else { app.swipeUp() }
        }
        XCTFail("Model memory control must be fully reachable: \(element)")
    }

    @MainActor func testOptionsAndModelSearchWithAnInstalledModel() async throws {
        continueAfterFailure = false
        let machine = try FixtureMachine()
        let port = try await machine.start()
        let app = XCUIApplication()
        defer { app.terminate(); machine.stop() }
        cleanUpFixture(machine, port: port, app: app)
        app.launchArguments = ["-UIPreferredContentSizeCategoryName", "UICTContentSizeCategoryL"]
        app.launch()
        XCTAssertTrue(app.navigateToDestination("Machines", shortcut: "5"))
        app.buttons["Add a Machine"].firstMatch.tap()
        app.buttons.matching(NSPredicate(format: "label BEGINSWITH 'Enter an Address'")).firstMatch.tap()
        let name = app.textFields["machine-name"]
        XCTAssertTrue(name.waitForExistence(timeout: 5))
        name.tap()
        name.typeText("Fixture Machine")
        let address = app.textFields["machine-address"]
        address.tap()
        address.typeText("127.0.0.1:\(port)")
        app.buttons["Add"].firstMatch.tap()
        XCTAssertTrue(app.navigateToDestination("Generate", shortcut: "1"))
        let chooser = app.buttons["choose-model"]
        reveal(chooser, in: app)
        chooser.tap()
        let model = app.buttons["model-flux-dev:q4"]
        XCTAssertTrue(model.waitForExistence(timeout: 10))
        model.tap()
        try checkLandscapeComposer(app)

        app.terminate()
        app.launchArguments = ["-UIPreferredContentSizeCategoryName", "UICTContentSizeCategoryAccessibilityXXXL"]
        app.launch()
        XCTAssertTrue(app.navigateToDestination("Generate", shortcut: "1"))
        reveal(chooser, in: app)
        chooser.tap()
        let search = app.searchFields.firstMatch
        XCTAssertTrue(search.waitForExistence(timeout: 5))
        search.tap()
        search.typeText("zzzznomodel")
        XCTAssertTrue(app.staticTexts["No matching models"].waitForExistence(timeout: 5))
        let closeSearch = app.buttons["model-chooser-close"]
        XCTAssertTrue(closeSearch.isHittable, "Model search must have a visible exit with the keyboard open")
        attach(app)
        closeSearch.tap()
        let options = app.buttons["Options"].firstMatch
        reveal(options, in: app)
        options.tap()
        XCTAssertTrue(app.navigationBars["More Options"].waitForExistence(timeout: 5))
        for identifier in ["options-shape", "options-steps", "options-batch"] {
            let control = app.buttons[identifier]
            reveal(control, in: app)
            XCTAssertLessThan(control.frame.height, app.frame.height * 0.45,
                              "An option must remain readable words, not a screen-high column of letters")
            XCTAssertGreaterThanOrEqual(control.frame.minX, 0)
            XCTAssertLessThanOrEqual(control.frame.maxX, app.frame.maxX)
        }
        XCTAssertFalse(app.buttons["More Options"].exists, "The sheet must not offer an inert button to reopen itself")
        attach(app)
        app.buttons["Done"].firstMatch.tap()
    }

    @MainActor private func checkLandscapeComposer(_ app: XCUIApplication) throws {
        XCUIDevice.shared.orientation = .landscapeLeft
        defer { XCUIDevice.shared.orientation = .portrait }
        let rotated = NSPredicate { _, _ in app.frame.width > app.frame.height }
        XCTAssertEqual(XCTWaiter.wait(for: [XCTNSPredicateExpectation(predicate: rotated, object: app)], timeout: 5), .completed)
        // Rotation delivers its new geometry before the composer settles.
        try awaitRotationLayout()
        let composer = app.scrollViews["phone-generate-form"]
        let prompt = app.descendants(matching: .any)["generation-prompt"].firstMatch
        for _ in 0..<6 where !prompt.isHittable { composer.swipeDown() }
        XCTAssertTrue(prompt.isHittable)
        prompt.tap()
        XCTAssertTrue(app.keyboards.firstMatch.waitForExistence(timeout: 5))
        prompt.typeText("Landscape lighthouse")
        XCTAssertTrue((prompt.value as? String)?.contains("Landscape lighthouse") == true)
        app.buttons["Done"].firstMatch.tap()
        XCTAssertTrue(app.keyboards.firstMatch.waitForNonExistence(timeout: 5))
        let submit = app.buttons["submit-generation"]
        for _ in 0..<12 where !submit.isHittable { composer.swipeUp() }
        XCTAssertTrue(submit.isHittable, "Generate must be reachable in landscape; never tap it in UAT")
        let options = app.buttons.matching(NSPredicate(format: "label == 'Options' OR label == 'More Options'")).firstMatch
        reveal(options, in: app)
        XCTAssertTrue(options.isHittable)
        options.tap()
        XCTAssertTrue(app.navigationBars["More Options"].waitForExistence(timeout: 5))
        attach(app)
        app.buttons["Done"].firstMatch.tap()
    }

    @MainActor func testQueueDetailsControlsAndPromptHistory() async throws {
        XCUIDevice.shared.orientation = .portrait
        defer { XCUIDevice.shared.orientation = .portrait }
        continueAfterFailure = false
        let machine = try FixtureMachine(queueFixture: true, queueControls: true)
        let port = try await machine.start()
        let fixtureName = "Queue History Fixture \(port)"
        let app = XCUIApplication()
        defer { app.terminate(); machine.stop() }
        cleanUpFixture(machine, port: port, app: app)
        app.launchArguments = ["-UIPreferredContentSizeCategoryName", "UICTContentSizeCategoryL"]
        app.launch()
        XCTAssertTrue(app.navigateToDestination("Machines", shortcut: "5"))
        app.buttons["Add a Machine"].firstMatch.tap()
        app.buttons.matching(NSPredicate(format: "label BEGINSWITH 'Enter an Address'")).firstMatch.tap()
        let name = app.textFields["machine-name"]
        XCTAssertTrue(name.waitForExistence(timeout: 5)); name.tap(); name.typeText(fixtureName)
        let address = app.textFields["machine-address"]
        address.tap(); address.typeText("127.0.0.1:\(port)")
        app.buttons["Add"].firstMatch.tap()
        XCTAssertTrue(app.navigateToDestination("Queue", shortcut: "3"))
        XCTAssertTrue(app.collectionViews["queue-list"].waitForExistence(timeout: 5),
                      "Queue rows must belong to the identified native Queue List")
        let open = app.buttons["queue-open-fixture-video"]
        XCTAssertTrue(open.waitForExistence(timeout: 10))
        XCTAssertFalse(app.staticTexts["A legacy verbose title that must not appear"].exists)
        open.tap()
        XCTAssertTrue(app.navigationBars["Job Details"].waitForExistence(timeout: 5))
        let inspector = app.descendants(matching: .any)["queue-detail"].firstMatch
        XCTAssertTrue(inspector.waitForExistence(timeout: 5))
        let modelID = inspector.staticTexts["ltx-2.5-22b-distilled:bf16"].firstMatch
        guard revealInspector(modelID, in: inspector, app: app) else { return }
        XCTAssertTrue(modelID.exists)
        let seed = inspector.staticTexts["Seed"].firstMatch
        guard revealInspector(seed, in: inspector, app: app) else { return }
        XCTAssertTrue(seed.exists)
        let pause = inspector.buttons["Pause"].firstMatch
        guard revealInspector(pause, in: inspector, app: app, towardTopWhenMissing: true) else { return }
        pause.tap()
        let resume = inspector.buttons["Resume"].firstMatch
        guard revealInspector(resume, in: inspector, app: app, towardTopWhenMissing: true) else { return }
        XCTAssertTrue(machine.queueActionRequests().contains("/api/queue/fixture-video/pause"),
                      "The fixture must receive the pause POST")
        resume.tap()
        guard revealInspector(pause, in: inspector, app: app, towardTopWhenMissing: true) else { return }
        XCTAssertTrue(machine.queueActionRequests().contains("/api/queue/fixture-video/resume"))
        attach(app)
        app.buttons["Done"].firstMatch.tap()
        app.buttons["queue-open-fixture-held"].tap()
        XCTAssertTrue(app.navigationBars["Job Details"].waitForExistence(timeout: 5))
        let retry = inspector.buttons["Retry"].firstMatch
        guard revealInspector(retry, in: inspector, app: app) else { return }
        retry.tap()
        guard revealInspector(pause, in: inspector, app: app) else { return }
        XCTAssertTrue(machine.queueActionRequests().contains("/api/queue/fixture-held/retry"))
        app.buttons["Done"].firstMatch.tap()
        XCTAssertTrue(app.navigateToDestination("Generate", shortcut: "1"))
        let history = app.buttons["prompt-history"]
        reveal(history, in: app); history.tap()
        XCTAssertTrue(app.navigationBars["Prompt History"].waitForExistence(timeout: 5))
        app.buttons["history-machine"].tap()
        app.buttons[fixtureName].firstMatch.tap()
        let prompt = app.buttons.matching(NSPredicate(format: "label CONTAINS 'A lighthouse in winter'")).firstMatch
        guard prompt.waitForExistence(timeout: 10) else {
            let sheet = app.debugDescription
            app.buttons["Done"].firstMatch.tap()
            _ = app.navigateToDestination("Machines", shortcut: "5")
            XCTFail("Fixture requests: \(machine.requestLog()). Sheet: \(sheet). Machines: \(app.debugDescription)")
            return
        }
        var search = app.searchFields.firstMatch
        if !search.exists { app.buttons["Search"].firstMatch.tap(); search = app.searchFields.firstMatch }
        XCTAssertTrue(search.waitForExistence(timeout: 5)); search.tap(); search.typeText("winter")
        XCTAssertTrue(prompt.waitForExistence(timeout: 10))
        attach(app)
        prompt.tap()
        let field = app.textFields["generation-prompt"].firstMatch
        XCTAssertTrue(field.waitForExistence(timeout: 5))
        XCTAssertEqual(field.value as? String, "A lighthouse in winter")
        for category in ["UICTContentSizeCategoryXS", "UICTContentSizeCategoryAccessibilityXXXL"] {
            app.terminate(); app.launchArguments = ["-UIPreferredContentSizeCategoryName", category]; app.launch()
            XCTAssertTrue(app.navigateToDestination("Queue", shortcut: "3"))
            let row = app.buttons["queue-open-fixture-video"]
            _ = row.waitForExistence(timeout: 10)
            // A restored List can retain a scroll offset across size changes;
            // materialize the first known fixture row in that List, not by
            // swiping the surrounding screen or relaxing the row assertion.
            let list = app.collectionViews["queue-list"]
            XCTAssertTrue(list.waitForExistence(timeout: 5), "The native Queue List must be present")
            // Missing lazy rows can lie past a tall header or above a restored
            // bottom offset. Search upward through content, then reverse once.
            for step in 0..<12 where !row.exists || !row.isHittable {
                if row.exists {
                    let top = app.navigationBars["Queue"].frame.maxY
                    let bottom = app.tabBars.firstMatch.exists ? app.tabBars.firstMatch.frame.minY : list.frame.maxY
                    if row.frame.midY < (top + bottom) / 2 { list.swipeDown() }
                    else { list.swipeUp() }
                } else if step < 6 { list.swipeUp() }
                else { list.swipeDown() }
            }
            guard row.exists, row.isHittable else {
                attach(app)
                XCTFail("Fixture video must be reachable after \(category): requests \(machine.requestLog()); \(app.debugDescription)")
                return
            }
            row.tap()
            XCTAssertTrue(app.navigationBars["Job Details"].waitForExistence(timeout: 5))
            attach(app)
            app.buttons["Done"].firstMatch.tap()
            XCTAssertTrue(app.navigateToDestination("Generate", shortcut: "1"))
            let history = app.buttons["prompt-history"]; reveal(history, in: app); history.tap()
            app.buttons["history-machine"].tap(); app.buttons[fixtureName].firstMatch.tap()
            XCTAssertTrue(prompt.waitForExistence(timeout: 10)); attach(app)
            app.buttons["Done"].firstMatch.tap()
        }
        reveal(app.buttons["prompt-history"], in: app); app.buttons["prompt-history"].tap()
        app.buttons["history-machine"].tap(); app.buttons[fixtureName].firstMatch.tap()
        XCTAssertTrue(prompt.waitForExistence(timeout: 10))
        app.buttons["Clear"].firstMatch.tap()
        app.buttons["Clear All Prompts"].firstMatch.tap()
        XCTAssertTrue(app.staticTexts["No prompts yet on this machine."].waitForExistence(timeout: 10))
    }

    /// Scroll only inside the sheet's visible list. Full swipes overshoot a
    /// short action row and oscillate between positions on the taller iPad.
    @MainActor private func revealInspector(_ element: XCUIElement, in inspector: XCUIElement,
                                           app: XCUIApplication, towardTopWhenMissing: Bool = false) -> Bool {
        for _ in 0..<20 {
            let frame = inspector.frame.intersection(app.frame)
            var top = max(app.navigationBars["Job Details"].frame.maxY, frame.minY) + 12
            var bottom = frame.maxY - 20
            for overlay in app.images.matching(identifier: "AdditionalDimmingOverlay").allElementsBoundByIndex {
                let covered = overlay.frame.intersection(frame)
                guard !covered.isNull, !covered.isEmpty else { continue }
                if covered.midY < frame.midY { top = max(top, covered.maxY + 12) }
                else { bottom = min(bottom, covered.minY - 12) }
            }
            guard bottom > top + 44 else {
                XCTFail("Inspector needs a usable viewport: \(frame), safe bounds \(top)...\(bottom)")
                return false
            }
            let exists = element.exists
            let target = exists ? element.frame : .zero
            if exists, element.isHittable, target.midY >= top, target.midY <= bottom { return true }
            let down = exists ? target.midY < top : towardTopWhenMissing
            let height = bottom - top
            let distance = min(height * 0.25, max(44, exists ? abs(target.midY - (top + bottom) / 2) : height * 0.25))
            let center = (top + bottom) / 2
            let x = frame.minX + frame.width * 0.9
            let origin = app.coordinate(withNormalizedOffset: .zero)
            let start = origin.withOffset(CGVector(dx: x, dy: center + (down ? -distance / 2 : distance / 2)))
            let end = origin.withOffset(CGVector(dx: x, dy: center + (down ? distance / 2 : -distance / 2)))
            start.press(forDuration: 0.1, thenDragTo: end,
                        withVelocity: .slow, thenHoldForDuration: 0.1)
        }
        XCTFail("Inspector control must be hittable within its visible bounds: \(element). \(inspector.debugDescription)")
        return false
    }

    @MainActor func testCuratedDiscoveryAndQueuedSourceImage() async throws {
        XCUIDevice.shared.orientation = .portrait
        defer { XCUIDevice.shared.orientation = .portrait }
        continueAfterFailure = false
        let machine = try FixtureMachine(queueFixture: true)
        let port = try await machine.start()
        let app = XCUIApplication()
        defer { app.terminate(); machine.stop() }
        cleanUpFixture(machine, port: port, app: app)
        app.launchArguments = ["-UIPreferredContentSizeCategoryName", "UICTContentSizeCategoryL"]
        app.launch()
        XCTAssertTrue(app.navigateToDestination("Machines", shortcut: "5"))
        app.buttons["Add a Machine"].firstMatch.tap()
        app.buttons.matching(NSPredicate(format: "label BEGINSWITH 'Enter an Address'")).firstMatch.tap()
        let name = app.textFields["machine-name"]
        XCTAssertTrue(name.waitForExistence(timeout: 5))
        name.tap(); name.typeText("Media Fixture")
        let address = app.textFields["machine-address"]
        address.tap(); address.typeText("127.0.0.1:\(port)")
        app.buttons["Add"].firstMatch.tap()
        XCTAssertTrue(app.navigateToDestination("Queue", shortcut: "3"))
        let row = app.descendants(matching: .any)["queue-entry-fixture-video"].firstMatch
        XCTAssertTrue(row.waitForExistence(timeout: 10))
        XCTAssertTrue(app.staticTexts["A coastal path at sunrise"].waitForExistence(timeout: 5))
        XCTAssertTrue(app.images["queue-source-fixture-video"].waitForExistence(timeout: 10))
        XCTAssertTrue(app.staticTexts["LTX-2.5 Distilled BF16"].exists)
        XCTAssertLessThanOrEqual(row.frame.width, 860)
        attach(app)
        for category in ["UICTContentSizeCategoryXS", "UICTContentSizeCategoryAccessibilityXXXL"] {
            app.terminate()
            app.launchArguments = ["-UIPreferredContentSizeCategoryName", category]
            app.launch()
            XCTAssertTrue(app.navigateToDestination("Queue", shortcut: "3"))
            reveal(app.descendants(matching: .any)["queue-entry-fixture-video"].firstMatch, in: app)
            XCTAssertTrue(app.images["queue-source-fixture-video"].waitForExistence(timeout: 10))
            XCTAssertTrue(app.staticTexts["A coastal path at sunrise"].exists)
            attach(app)
        }
        app.terminate()
        app.launchArguments = ["-UIPreferredContentSizeCategoryName", "UICTContentSizeCategoryL"]
        app.launch()
        // Select the fixture explicitly: global Models follows the saved
        // preferred machine, which another full-suite test may leave offline.
        XCTAssertTrue(app.navigateToDestination("Machines", shortcut: "5"))
        let card = app.descendants(matching: .any).matching(NSPredicate(format:
            "identifier BEGINSWITH 'machine-card-' AND label CONTAINS %@", "127.0.0.1:\(port)")).firstMatch
        XCTAssertTrue(card.waitForExistence(timeout: 5))
        card.tap()
        XCTAssertTrue(app.navigationBars["Media Fixture"].waitForExistence(timeout: 5))
        // The installed count distinguishes this detail link from the iPad sidebar.
        let modelsLink = app.buttons.matching(NSPredicate(format:
            "label BEGINSWITH 'Models' AND label CONTAINS 'installed'")).firstMatch
        XCTAssertTrue(modelsLink.waitForExistence(timeout: 5))
        for _ in 0..<5 where !modelsLink.isHittable { app.swipeUp() }
        modelsLink.tap()
        app.buttons["models-pane"].tap()
        app.buttons["Discover"].firstMatch.tap()
        let curated = app.descendants(matching: .any)["curated-model-flux-dev:q4"].firstMatch
        XCTAssertTrue(curated.waitForExistence(timeout: 10))
        XCTAssertTrue(app.staticTexts["Hugging Face"].firstMatch.exists)
        let search = app.searchFields.firstMatch
        if !search.exists { app.navigationBars["Models"].buttons["Search"].tap() }
        XCTAssertTrue(search.waitForExistence(timeout: 5))
        search.tap(); search.typeText("black-forest")
        XCTAssertTrue(curated.waitForExistence(timeout: 5))
        XCTAssertFalse(app.descendants(matching: .any)["curated-model-ltx-2.5-22b-distilled:bf16"].exists)
        if app.buttons["Cancel"].firstMatch.isHittable { app.buttons["Cancel"].firstMatch.tap() }
        let get = app.buttons["Get FLUX.1 Dev Q4"]
        XCTAssertTrue(get.waitForExistence(timeout: 5))
        for _ in 0..<5 where !get.isHittable { app.swipeUp() }
        get.tap()
        for _ in 0..<50 where machine.installedRequests.isEmpty { try await Task.sleep(for: .milliseconds(100)) }
        XCTAssertEqual(machine.installedRequests, ["flux-dev:q4"], "Curated Get must target one exact manifest checkpoint")
        attach(app)
    }

    @MainActor private func awaitRotationLayout() throws {
        // UIKit's orientation animation is not included in XCTest's app-idle wait.
        RunLoop.current.run(until: Date().addingTimeInterval(1))
    }

    @MainActor private func reveal(_ element: XCUIElement, in app: XCUIApplication) {
        let form = app.scrollViews["phone-generate-form"]
        for _ in 0..<12 {
            if !form.exists {
                if element.isHittable { break }
                app.swipeUp()
                continue
            }
            if app.navigationBars["More Options"].exists {
                if element.isHittable { break }
                app.swipeUp()
                continue
            }
            let top = app.navigationBars.firstMatch.frame.maxY + 12
            let bottom = app.buttons["submit-generation"].frame.minY - 12
            if element.isHittable, element.frame.midY >= top + 20, element.frame.midY <= bottom - 20 { break }
            // Swipe the clear right edge. A centered swipe lands in the
            // horizontal picture wells and leaves the form where it was.
            let down = element.exists && element.frame.midY < top + 20
            // top/bottom are screen coordinates. Adding them to the form's
            // origin put the gesture below its viewport, over fixed chrome.
            let origin = app.coordinate(withNormalizedOffset: .zero)
            let x = form.frame.minX + form.frame.width * 0.9
            let upper = max(top, form.frame.minY) + 20
            let lower = min(bottom, form.frame.maxY) - 20
            XCTAssertGreaterThan(lower, upper, "The form needs a visible scrolling viewport")
            let high = upper + (lower - upper) * 0.2
            let low = upper + (lower - upper) * 0.7
            let start = origin.withOffset(CGVector(dx: x, dy: down ? high : low))
            let end = origin.withOffset(CGVector(dx: x, dy: down ? low : high))
            start.press(forDuration: 0.1, thenDragTo: end,
                        withVelocity: .slow, thenHoldForDuration: 0.1)
        }
        XCTAssertTrue(element.isHittable, "The control must be reachable within the Generate form: \(app.debugDescription)")
    }

    @MainActor private func attach(_ app: XCUIApplication) {
        let screenshot = XCTAttachment(screenshot: app.screenshot())
        screenshot.lifetime = .keepAlways
        add(screenshot)
    }
}
