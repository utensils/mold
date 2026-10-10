import XCTest

final class PromptEditorUITests: XCTestCase {
    override func setUp() { super.setUp(); acceptCompanionPermissions() }

    @MainActor func testLiveMultilineClearUndoHistoryAndRelaunch() async throws {
        continueAfterFailure = false
        let machine = try FixtureMachine(queueControls: true, promptExpansionFixture: true)
        let port = try await machine.start()
        let app = XCUIApplication()
        defer { app.terminate(); machine.stop() }
        cleanUpFixture(machine, port: port, app: app)
        app.launch()
        pair(port, in: app)
        XCTAssertTrue(app.navigateToDestination("Generate", shortcut: "1"))
        guard openEditor(in: app) else { return }
        let editor = app.textViews["prompt-editor-text"]
        guard editor.waitForExistence(timeout: 5) else { XCTFail("Prompt editor must open"); return }
        let text = (1...8).map { "Line \($0): a detailed landscape with mountains and soft morning light." }.joined(separator: "\n")
        if let existing = editor.value as? String, !existing.isEmpty {
            action("prompt-editor-clear", in: app).tap()
        }
        editor.tap(); editor.typeText(text)
        XCTAssertEqual(editor.value as? String, text)
        app.typeKey(XCUIKeyboardKey.leftArrow, modifierFlags: [])
        app.typeKey(XCUIKeyboardKey.return, modifierFlags: .command)
        XCTAssertEqual(editor.value as? String, text)
        XCTAssertTrue(machine.generationRequests.isEmpty)
        attach(app, "Long multiline prompt with keyboard")
        action("prompt-editor-clear", in: app).tap()
        XCTAssertEqual(editor.value as? String, "")
        action("prompt-editor-undo-clear", in: app).tap()
        XCTAssertEqual(editor.value as? String, text)
        showActions(in: app)
        app.buttons["Expand"].firstMatch.tap()
        guard editor.waitForValue("Expanded fixture prompt") else { XCTFail("Fixture rewrite must appear"); return }
        action("prompt-editor-clear", in: app).tap()
        action("prompt-editor-undo-clear", in: app).tap()
        XCTAssertTrue(editor.waitForValue("Expanded fixture prompt"))
        showActions(in: app)
        let undoExpand = app.buttons["Undo Expand"].firstMatch
        guard undoExpand.waitForExistence(timeout: 5) else { XCTFail("Clear/Undo must retain rewrite undo"); return }; undoExpand.tap()
        guard editor.waitForValue(text) else { XCTFail("Undo Expand must restore original multiline text"); return }
        showActions(in: app)
        app.buttons["Suggest Other Ways"].firstMatch.tap()
        let suggestion = app.buttons["Expanded fixture prompt"].firstMatch
        guard suggestion.waitForExistence(timeout: 5) else { XCTFail("Suggestions must present from fallback menu"); return }
        suggestion.tap()
        guard editor.waitForValue("Expanded fixture prompt") else { XCTFail("Chosen suggestion must update editable draft"); return }
        XCTAssertTrue(machine.requestLog().contains("POST /api/expand"))
        app.buttons["prompt-editor-done"].tap()
        guard openEditor(in: app) else { return }
        XCTAssertEqual(editor.value as? String, "Expanded fixture prompt")
        action("prompt-editor-history", in: app).tap()
        let search = app.searchFields.firstMatch
        XCTAssertTrue(search.waitForExistence(timeout: 5)); search.tap(); search.typeText("lighthouse")
        let history = app.buttons.matching(NSPredicate(format: "label CONTAINS 'A lighthouse in winter'")).firstMatch
        XCTAssertTrue(history.waitForExistence(timeout: 10)); history.tap()
        XCTAssertEqual(editor.value as? String, "A lighthouse in winter")
        app.buttons["prompt-editor-done"].tap()
        app.terminate(); app.launch()
        XCTAssertTrue(app.navigateToDestination("Generate", shortcut: "1"))
        guard openEditor(in: app) else { return }
        XCTAssertEqual(editor.value as? String, "A lighthouse in winter")
        XCTAssertTrue(machine.generationRequests.isEmpty, "Editing and newline entry must never generate")
        app.buttons["prompt-editor-done"].tap()
    }

    @MainActor func testLargestTextEditorAndPersistedDatePreference() async throws {
        continueAfterFailure = false
        let machine = try FixtureMachine(galleryPrints: 4)
        let port = try await machine.start()
        let app = XCUIApplication()
        defer { app.terminate(); machine.stop() }
        cleanUpFixture(machine, port: port, app: app)
        app.launchArguments = ["-UIPreferredContentSizeCategoryName", "UICTContentSizeCategoryAccessibilityXXXL"]
        app.launch(); pair(port, in: app)
        XCTAssertTrue(app.navigateToDestination("Generate", shortcut: "1"))
        guard openEditor(in: app) else { return }
        let editor = app.textViews["prompt-editor-text"]
        guard editor.waitForExistence(timeout: 5) else { XCTFail("Prompt editor must open"); return }
        XCTAssertTrue(app.buttons["prompt-editor-actions"].isHittable)
        if let existing = editor.value as? String, !existing.isEmpty {
            action("prompt-editor-clear", in: app).tap()
        }
        editor.tap(); editor.typeText("Large text\nSecond line")
        XCTAssertTrue(app.buttons["prompt-editor-done"].isHittable)
        XCTAssertGreaterThan(editor.frame.height, 200)
        attach(app, "Largest text prompt editor with keyboard")
        app.buttons["prompt-editor-done"].tap()
        XCTAssertTrue(app.navigateToDestination("Machines", shortcut: "5"))
        app.buttons["Settings"].firstMatch.tap()
        let toggle = app.switches["library-show-date-separators"]
        for _ in 0..<8 where !toggle.isHittable { app.swipeUp() }
        XCTAssertTrue(toggle.waitForExistence(timeout: 5))
        XCTAssertEqual(toggle.value as? String, "1")
        toggle.coordinate(withNormalizedOffset: CGVector(dx: 0.92, dy: 0.5)).tap()
        XCTAssertTrue(toggle.waitForValue("0"))
        attach(app, "Date separators disabled in Settings")
        app.terminate(); app.launch()
        XCTAssertTrue(app.navigateToDestination("Machines", shortcut: "5"))
        app.buttons["Settings"].firstMatch.tap()
        for _ in 0..<8 where !toggle.isHittable { app.swipeUp() }
        XCTAssertTrue(toggle.waitForValue("0"))
        app.buttons["Done"].firstMatch.tap()
        XCTAssertTrue(app.navigateToDestination("Library", shortcut: "2"))
        let print = app.buttons.matching(NSPredicate(format: "label BEGINSWITH 'Fixture 0,'")).firstMatch
        XCTAssertTrue(print.waitForExistence(timeout: 10))
        XCTAssertFalse(app.staticTexts["day-header"].exists)
        attach(app, "Continuous Library after relaunch")
        app.buttons["Settings"].firstMatch.tap()
        for _ in 0..<8 where !toggle.isHittable { app.swipeUp() }
        toggle.coordinate(withNormalizedOffset: CGVector(dx: 0.92, dy: 0.5)).tap()
        XCTAssertTrue(toggle.waitForValue("1"))
    }

    @MainActor private func openEditor(in app: XCUIApplication) -> Bool {
        let opener = app.buttons["prompt-editor-open"]
        guard opener.waitForExistence(timeout: 10) else { XCTFail("Edit prompt must exist"); return false }
        let queue = app.buttons["generate-queue-status"]
        let origin = app.coordinate(withNormalizedOffset: .zero)
        for _ in 0..<6 where !opener.isHittable || opener.frame.maxY >= queue.frame.minY {
            let start = origin.withOffset(CGVector(dx: app.frame.maxX - 8, dy: queue.frame.minY - 24))
            let end = origin.withOffset(CGVector(dx: app.frame.maxX - 8, dy: app.navigationBars.firstMatch.frame.maxY + 24))
            start.press(forDuration: 0.05, thenDragTo: end)
        }
        guard opener.isHittable, opener.frame.maxY < queue.frame.minY else {
            XCTFail("Edit prompt must be fully above pinned actions"); return false
        }
        opener.tap()
        return true
    }

    @MainActor private func showActions(in app: XCUIApplication) {
        let menu = app.buttons["prompt-editor-actions"]
        if menu.exists { menu.tap() }
    }

    @MainActor private func action(_ id: String, in app: XCUIApplication) -> XCUIElement {
        let button = app.buttons[id]
        if !button.exists { showActions(in: app) }
        XCTAssertTrue(button.waitForExistence(timeout: 5))
        return button
    }

    @MainActor private func pair(_ port: UInt16, in app: XCUIApplication) {
        XCTAssertTrue(app.navigateToDestination("Machines", shortcut: "5"))
        app.buttons["Add a Machine"].firstMatch.tap()
        app.buttons.matching(NSPredicate(format: "label BEGINSWITH 'Enter an Address'")).firstMatch.tap()
        let name = app.textFields["machine-name"]
        XCTAssertTrue(name.waitForExistence(timeout: 5)); name.tap(); name.typeText("Prompt Editor Fixture \(port)")
        let address = app.textFields["machine-address"]
        address.tap(); address.typeText("127.0.0.1:\(port)")
        app.buttons["Add"].firstMatch.tap()
    }

    @MainActor private func attach(_ app: XCUIApplication, _ name: String) {
        let attachment = XCTAttachment(screenshot: app.screenshot())
        attachment.name = name; attachment.lifetime = .keepAlways; add(attachment)
    }
}

private extension XCUIElement {
    func waitForValue(_ expected: String) -> Bool {
        XCTWaiter.wait(for: [XCTNSPredicateExpectation(predicate: NSPredicate(format: "value == %@", expected), object: self)], timeout: 5) == .completed
    }
}
