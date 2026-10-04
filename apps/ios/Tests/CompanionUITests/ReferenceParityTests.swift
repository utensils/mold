import XCTest

/// Real native chooser/import/reorder/submission, backed by disposable loopback media.
final class ReferenceParityTests: XCTestCase {
    override func setUp() { super.setUp(); acceptCompanionPermissions() }

    @MainActor func testMiniMaxMultipleReferencesReachAdmissionInOrder() async throws {
        try await referenceAdmission(size: "UICTContentSizeCategoryL")
    }
    @MainActor func testPopulatedReferencesExtraSmall() async throws {
        try await referenceAdmission(size: "UICTContentSizeCategoryXS")
    }
    @MainActor func testPopulatedReferencesAccessibilityXXXL() async throws {
        try await referenceAdmission(size: "UICTContentSizeCategoryAccessibilityXXXL")
    }
    @MainActor private func referenceAdmission(size: String) async throws {
        continueAfterFailure = false
        let machine = try FixtureMachine(referenceFixture: true, galleryPrints: 4)
        let port = try await machine.start()
        let app = XCUIApplication()
        defer { app.terminate(); machine.stop() }
        cleanUpFixture(machine, port: port, app: app); continueAfterFailure = false
        app.launchArguments = ["-UIPreferredContentSizeCategoryName", size]
        app.launch()
        try addMachine(app, port: port)
        try choose("minimax-h3-ref2va:comfy-pruned-int8-turbo-4step", in: app)
        let prompt = app.descendants(matching: .any)["generation-prompt"].firstMatch
        reveal(prompt, app: app); prompt.tap()
        let text = "A scene using image 1 and image 2"
        if let value = prompt.value as? String, !value.isEmpty, value != prompt.placeholderValue {
            prompt.tap(withNumberOfTaps: 3, numberOfTouches: 1)
        }
        prompt.typeText(text)
        XCTAssertEqual(prompt.value as? String, text, "Each run replaces the persisted prompt")
        let done = app.toolbars.buttons["Done"].firstMatch
        XCTAssertTrue(done.waitForExistence(timeout: 5), app.debugDescription)
        done.tap()
        XCTAssertTrue(app.keyboards.firstMatch.waitForNonExistence(timeout: 5))
        try pickLibrary("Add image, empty", image: 0, app: app)
        try pickLibrary("Add image, empty", image: 1, app: app)
        let order = app.buttons["Order reference 2"]
        reveal(order, app: app, geometryOnly: true); XCTAssertTrue(order.exists)
        let earlier = app.descendants(matching: .any).matching(NSPredicate(format: "label == 'Move earlier'")).firstMatch
        // Library dismissal and attachment decoding can replace the menu node.
        for _ in 0..<3 {
            let title = order.staticTexts["Order"].firstMatch
            let anchor = title.exists ? title : order
            reveal(anchor, app: app, geometryOnly: true)
            anchor.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.5)).tap()
            if earlier.waitForExistence(timeout: 2) { break }
        }
        XCTAssertTrue(earlier.exists, app.debugDescription)
        earlier.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.5)).tap()
        let references = app.scrollViews["generation-references"]
        reveal(references, app: app)
        let populated = XCTAttachment(screenshot: app.screenshot()); populated.name = "Populated MiniMax references \(size)"; populated.lifetime = .keepAlways; add(populated)
        let guidance = app.staticTexts["Use image 1, video 1, or audio 1 in your prompt. Reference order is preserved."].firstMatch
        try centerForAudit(guidance, app: app)
        try auditContrast(guidance, app: app, size: size)
        try centerForAudit(references, app: app)
        try auditContrast(references, app: app, size: size)
        let label = app.buttons["Order reference 1"].staticTexts["Order"].firstMatch
        XCTAssertTrue(label.exists, app.debugDescription)
        XCTAssertTrue(references.frame.contains(label.frame), "Order label \(label.frame) must fit reference row \(references.frame)")
        let count = app.staticTexts["2 of 12 references"].firstMatch
        XCTAssertGreaterThanOrEqual(count.frame.minY, label.frame.maxY, "Count must not overlap Order")
        let settled = XCTAttachment(screenshot: app.screenshot()); settled.name = "Audited MiniMax reference row \(size)"; settled.lifetime = .keepAlways; add(settled)
        try app.performAccessibilityAudit(for: [.dynamicType, .textClipped, .hitRegion, .sufficientElementDescription]) { issue in
            // Match ShellAccessibilityTests' narrow exemption: XCUITest reports
            // hidden caption2 tile badges although the well speaks their ordinal.
            if issue.auditType == .dynamicType, issue.element?.identifier == "tile-badge" { return true }
            XCTFail("Populated references at \(size): \(issue.compactDescription), \(issue.element?.label ?? "unnamed")")
            return true
        }
        let submit = app.buttons["submit-generation"]
        reveal(submit, app: app); XCTAssertTrue(submit.isEnabled); submit.tap()
        for _ in 0..<50 where machine.generationRequests.isEmpty { try await Task.sleep(for: .milliseconds(100)) }
        let data = try XCTUnwrap(machine.generationRequests.first)
        let body = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let requests = try XCTUnwrap(body["requests"] as? [[String: Any]])
        let refs = try XCTUnwrap(requests.first?["references"] as? [[String: Any]])
        XCTAssertEqual(refs.count, 2)
        XCTAssertEqual(refs.map { ($0["provenance"] as? [String: Any])?["name"] as? String }, ["fixture-1.png", "fixture-0.png"])
        XCTAssertTrue(refs.allSatisfy { ($0["media"] as? [String: Any])?["authority"] as? String == "inline" })
        XCTAssertNil(requests.first?["edit_images"])
        XCTAssertNil(requests.first?["source_image"])
        let shot = XCTAttachment(screenshot: app.screenshot()); shot.name = "MiniMax two references native"; shot.lifetime = .keepAlways; add(shot)
    }

    @MainActor func testNamedViewsAndBoundaryWellsFollowModelCapabilities() async throws {
        continueAfterFailure = false
        let machine = try FixtureMachine(referenceFixture: true, galleryPrints: 4)
        let port = try await machine.start(); let app = XCUIApplication()
        defer { app.terminate(); machine.stop() }
        cleanUpFixture(machine, port: port, app: app); continueAfterFailure = false; app.launch(); try addMachine(app, port: port)
        try choose("hunyuan3d-2mv:fp16", in: app)
        XCTAssertTrue(app.buttons["Front, empty"].waitForExistence(timeout: 5))
        try pickLibrary("Front, empty", image: 0, app: app)
        XCTAssertTrue(app.buttons["Front"].exists)
        try choose("wan22-ti2v-5b:fp16", in: app)
        XCTAssertTrue(app.buttons["First frame, empty"].waitForExistence(timeout: 5))
        try pickLibrary("First frame, empty", image: 0, app: app)
        try pickLibrary("Last frame, empty", image: 1, app: app)
        XCTAssertFalse(app.buttons["Start from, empty"].exists)
        try choose("minimax-h3-fl2va:comfy-pruned-int8-turbo-4step-768p", in: app)
        XCTAssertTrue(app.buttons["Last frame, empty"].exists || app.buttons["Last frame"].exists)
        try choose("qwen-image-edit-2511:q4", in: app)
        XCTAssertTrue(app.buttons["Picture to edit, empty"].exists)
        XCTAssertFalse(app.buttons["Start from, empty"].exists)
    }

    @MainActor private func addMachine(_ app: XCUIApplication, port: UInt16) throws {
        XCTAssertTrue(app.navigateToDestination("Machines", shortcut: "5"))
        app.buttons["Add a Machine"].firstMatch.tap()
        app.buttons.matching(NSPredicate(format: "label BEGINSWITH 'Enter an Address'")).firstMatch.tap()
        let name = app.textFields["machine-name"]; XCTAssertTrue(name.waitForExistence(timeout: 5)); name.tap(); name.typeText("Reference UAT")
        let address = app.textFields["machine-address"]; address.tap(); address.typeText("127.0.0.1:\(port)")
        app.buttons["Add"].firstMatch.tap()
        XCTAssertTrue(app.navigateToDestination("Generate", shortcut: "1"))
    }
    @MainActor private func choose(_ model: String, in app: XCUIApplication) throws {
        if app.keyboards.firstMatch.exists, app.toolbars.buttons["Done"].firstMatch.exists {
            app.toolbars.buttons["Done"].firstMatch.tap()
            XCTAssertTrue(app.keyboards.firstMatch.waitForNonExistence(timeout: 5))
        }
        let chooser = app.buttons["choose-model"]
        let heading = chooser.staticTexts["Model"].firstMatch
        reveal(heading.exists ? heading : chooser, app: app)
        if heading.exists { heading.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.5)).tap() }
        else { tapMenu(chooser) }
        let sheet = app.otherElements["model-chooser"].firstMatch
        XCTAssertTrue(sheet.waitForExistence(timeout: 5), app.debugDescription)
        let title = model.hasPrefix("hunyuan") ? "3-D object" : (model.hasPrefix("minimax") || model.hasPrefix("wan")) ? "Short clip" : "Still picture"
        let kind = sheet.buttons.matching(NSPredicate(format: "label IN %@", ["Still picture", "Short clip", "3-D object"])).firstMatch
        XCTAssertTrue(kind.waitForExistence(timeout: 5), app.debugDescription)
        if kind.label != title {
            tapMenu(kind)
            let nested = app.popUpButtons.matching(NSPredicate(format: "label BEGINSWITH 'Kind'")).firstMatch
            if nested.waitForExistence(timeout: 2) { nested.tap() }
            let entry = app.buttons.matching(NSPredicate(format: "label == %@", title)).allElementsBoundByIndex.last { $0.isHittable }
            let option = try XCTUnwrap(entry, app.debugDescription)
            option.tap()
        }
        let row = app.buttons["model-" + model]
        for _ in 0..<10 where !row.exists || !row.isHittable { app.swipeUp() }
        XCTAssertTrue(row.waitForExistence(timeout: 10)); row.tap()
    }
    @MainActor private func pickLibrary(_ label: String, image: Int, app: XCUIApplication) throws {
        let well = app.buttons[label]; reveal(well, app: app); XCTAssertTrue(well.exists); well.tap()
        app.buttons["Choose from Library…"].firstMatch.tap()
        let tile = app.buttons.matching(NSPredicate(format: "label CONTAINS %@", "Fixture \(image)")).firstMatch
        XCTAssertTrue(tile.waitForExistence(timeout: 10)); tile.tap()
        XCTAssertTrue(app.navigationBars["Choose from Library"].waitForNonExistence(timeout: 10))
    }
    @MainActor private func tapMenu(_ element: XCUIElement) {
        // SwiftUI List's wrapping button includes blank row space outside Menu's label.
        let target = element.descendants(matching: .button).allElementsBoundByIndex
            .filter { !$0.frame.isEmpty }.min { $0.frame.width * $0.frame.height < $1.frame.width * $1.frame.height } ?? element
        target.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.5)).tap()
    }
    @MainActor private func form(in app: XCUIApplication) -> XCUIElement {
        let phone = app.scrollViews["phone-generate-form"]
        if phone.exists { return phone }
        return app.descendants(matching: .any)["bottom-chrome"].firstMatch.scrollViews.firstMatch
    }
    @MainActor private func auditContrast(_ scope: XCUIElement, app: XCUIApplication, size: String) throws {
        // Audit each region once, fully visible. Other form regions may be
        // scrolled through system glass while this region is sampled.
        try app.performAccessibilityAudit(for: .contrast) { issue in
            guard let element = issue.element else { return false }
            let belongs = element.frame == scope.frame && element.label == scope.label
                || scope.descendants(matching: element.elementType).matching(NSPredicate(format: "label == %@", element.label))
                    .allElementsBoundByIndex.contains { $0.frame == element.frame }
            guard belongs else { return true }
            XCTFail("References at \(size): \(issue.compactDescription), \(element.label) at \(element.frame)")
            return true
        }
    }
    @MainActor private func centerForAudit(_ scope: XCUIElement, app: XCUIApplication) throws {
        reveal(scope, app: app)
        let scroll = form(in: app)
        guard scroll.exists else { return }
        let top = max(app.navigationBars.firstMatch.frame.maxY + 64, scroll.frame.minY)
        let bottom = min(app.buttons["submit-generation"].frame.minY - 24, scroll.frame.maxY)
        let center = (top + bottom) / 2
        for _ in 0..<8 {
            let delta = center - scope.frame.midY
            if abs(delta) <= 20 { break }
            let travel = (delta < 0 ? -1.0 : 1.0) * max(40, min(200, abs(delta)))
            let start = scroll.coordinate(withNormalizedOffset: CGVector(dx: 0.03, dy: 0.5))
            start.press(forDuration: 0.1, thenDragTo: start.withOffset(CGVector(dx: 0, dy: travel)),
                        withVelocity: .slow, thenHoldForDuration: 0.3)
        }
        XCTAssertGreaterThanOrEqual(scope.frame.minY, app.navigationBars.firstMatch.frame.maxY)
        XCTAssertLessThanOrEqual(scope.frame.maxY, app.buttons["submit-generation"].frame.minY)
    }
    @MainActor private func reveal(_ element: XCUIElement, app: XCUIApplication, geometryOnly: Bool = false) {
        for _ in 0..<10 {
            let top = app.navigationBars.firstMatch.exists ? app.navigationBars.firstMatch.frame.maxY : app.frame.minY
            var bottom = app.tabBars.firstMatch.exists ? app.tabBars.firstMatch.frame.minY : app.frame.maxY
            let submit = app.buttons["submit-generation"]
            if element.identifier != "submit-generation", submit.exists { bottom = min(bottom, submit.frame.minY) }
            if element.exists, (geometryOnly || element.elementType == .scrollView || element.isHittable), element.frame.minY >= top, element.frame.maxY <= bottom { return }
            if element.exists, element.frame.minX >= app.frame.maxX {
                app.scrollViews["generation-references"].swipeLeft()
            } else if element.exists, element.frame.maxX <= app.frame.minX {
                app.scrollViews["generation-references"].swipeRight()
            } else {
                let form = form(in: app)
                if form.exists {
                    // The center can fall inside the multiline prompt and scroll
                    // that text field. Drag the form's observed padding instead.
                    let delta = element.exists
                        ? (element.frame.minY < top ? top - element.frame.minY + 8 : bottom - element.frame.maxY - 8)
                        : -200
                    let start = form.coordinate(withNormalizedOffset: CGVector(dx: 0.03, dy: 0.5))
                    let travel = (delta < 0 ? -1.0 : 1.0) * max(40, min(250, abs(delta)))
                    let end = start.withOffset(CGVector(dx: 0, dy: travel))
                    start.press(forDuration: 0.1, thenDragTo: end, withVelocity: .slow, thenHoldForDuration: 0.3)
                } else if element.exists, element.frame.minY < top { app.swipeDown() } else { app.swipeUp() }
            }
        }
    }
}
