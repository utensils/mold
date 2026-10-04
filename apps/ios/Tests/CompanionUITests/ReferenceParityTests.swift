import XCTest

/// Real native chooser/import/reorder/submission, backed by disposable loopback media.
final class ReferenceParityTests: XCTestCase {
    override func setUp() { super.setUp(); acceptCompanionPermissions() }

    @MainActor func testMiniMaxMultipleReferencesReachAdmissionInOrder() async throws {
        continueAfterFailure = false
        let machine = try FixtureMachine(referenceFixture: true, galleryPrints: 4)
        let port = try await machine.start()
        let app = XCUIApplication()
        defer { app.terminate(); machine.stop() }
        cleanUpFixture(machine, port: port, app: app)
        app.launch()
        try addMachine(app, port: port)
        try choose("minimax-h3-ref2va:comfy-pruned-int8-turbo-4step", in: app)
        let prompt = app.textViews["generation-prompt"]
        reveal(prompt, app: app); prompt.tap(); prompt.typeText("A scene using image 1 and image 2")
        app.buttons["Hide keyboard"].firstMatch.exists ? app.buttons["Hide keyboard"].firstMatch.tap() : app.tap()
        try pickLibrary("Add reference image, empty", image: 0, app: app)
        try pickLibrary("Add reference image, empty", image: 1, app: app)
        let order = app.buttons["Order reference 2"]
        reveal(order, app: app); XCTAssertTrue(order.exists); order.tap()
        app.buttons["Move earlier"].firstMatch.tap()
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
        cleanUpFixture(machine, port: port, app: app); app.launch(); try addMachine(app, port: port)
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
        let chooser = app.buttons["choose-model"]; reveal(chooser, app: app); chooser.tap()
        let kind = app.buttons.matching(NSPredicate(format: "label IN %@", ["Still picture", "Short clip", "3-D object"])).firstMatch
        if kind.exists {
            kind.tap()
            let title = model.hasPrefix("hunyuan") ? "3-D object" : (model.hasPrefix("minimax") || model.hasPrefix("wan")) ? "Short clip" : "Still picture"
            var entry = app.descendants(matching: .any).matching(NSPredicate(format: "label == %@", title)).allElementsBoundByIndex.first { $0.isHittable }
            if entry == nil {
                let picker = app.popUpButtons.matching(NSPredicate(format: "label BEGINSWITH 'Kind'")).firstMatch
                if picker.waitForExistence(timeout: 2) { picker.tap() }
                entry = app.descendants(matching: .any).matching(NSPredicate(format: "label == %@", title)).allElementsBoundByIndex.first { $0.isHittable }
            }
            XCTAssertNotNil(entry, app.debugDescription); entry?.tap()
        }
        let search = app.searchFields.firstMatch
        if search.waitForExistence(timeout: 3) { search.tap(); search.typeText(model) }
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
    @MainActor private func reveal(_ element: XCUIElement, app: XCUIApplication) {
        for _ in 0..<10 where !element.isHittable { app.swipeUp() }
    }
}
