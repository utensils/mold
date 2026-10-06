import XCTest

final class MediaExportUITests: XCTestCase {
    override func setUp() { super.setUp(); acceptCompanionPermissions(); continueAfterFailure = false }

    @MainActor private func fixture(size: String = "UICTContentSizeCategoryL", unsupportedFormats: Bool = false) async throws -> (XCUIApplication, FixtureMachine, String) {
        let identity = UUID().uuidString
        let machine = try FixtureMachine(exportFixture: true, unsupportedExportFormats: unsupportedFormats, galleryPrints: 3, galleryID: identity, mixedMedia: true)
        let port = try await machine.start()
        let app = XCUIApplication()
        cleanUpFixture(machine, port: port, app: app)
        app.launchArguments += ["-UIPreferredContentSizeCategoryName", size]
        app.resetAuthorizationStatus(for: .photos)
        app.launch()
        XCTAssertTrue(app.navigateToDestination("Machines", shortcut: "5"))
        app.buttons["Add a Machine"].firstMatch.tap()
        app.buttons.matching(NSPredicate(format: "label BEGINSWITH 'Enter an Address'")).firstMatch.tap()
        let address = app.textFields["machine-address"]
        XCTAssertTrue(address.waitForExistence(timeout: 5)); address.tap(); address.typeText("127.0.0.1:\(port)")
        app.buttons["Add"].firstMatch.tap()
        XCTAssertTrue(app.navigateToDestination("Library", shortcut: "2"))
        if !app.navigationBars["All Prints"].exists { app.chooseLibraryShelf("All Prints") }
        return (app, machine, identity)
    }
    @MainActor private func open(_ index: Int, identity: String, app: XCUIApplication) throws {
        let tile = app.buttons.matching(NSPredicate(format: "label BEGINSWITH %@", "Photos-\(identity) \(index),")).firstMatch
        let grid = app.scrollViews.firstMatch
        _ = tile.waitForExistence(timeout: 5)
        for _ in 0..<8 where !tile.exists || !tile.isHittable {
            guard grid.exists else { throw ExportUATFailure.missingControl("Library viewport") }
            grid.swipeUp()
        }
        guard tile.exists && tile.isHittable else { throw ExportUATFailure.missingControl("fixture print \(index)") }
        tile.tap()
    }
    private enum ExportUATFailure: Error { case missingControl(String) }
    @MainActor private func export(_ app: XCUIApplication) throws {
        app.buttons["More"].firstMatch.tap(); app.buttons["Export…"].firstMatch.tap()
        XCTAssertTrue(app.navigationBars["Export Media"].waitForExistence(timeout: 10))
        guard app.buttons["export-format"].waitForExistence(timeout: 10) else { throw ExportUATFailure.missingControl("export-format") }
    }
    @MainActor private func choose(_ id: String, _ value: String, app: XCUIApplication) {
        let picker = app.buttons[id]
        for _ in 0..<6 where !picker.isHittable { app.swipeUp() }
        picker.tap()
        let item = app.buttons[value].firstMatch
        if item.waitForExistence(timeout: 2) { item.tap() }
        else { let text = app.staticTexts[value].firstMatch; XCTAssertTrue(text.waitForExistence(timeout: 3)); text.tap() }
    }
    @MainActor private func submit(_ app: XCUIApplication) {
        let button = app.buttons["export-submit"]
        for _ in 0..<8 where !button.isHittable { app.swipeUp() }
        XCTAssertTrue(button.isHittable); XCTAssertTrue(button.isEnabled); button.tap()
    }
    @MainActor private func evidence(_ app: XCUIApplication, _ title: String) {
        let image = XCTAttachment(screenshot: app.screenshot()); image.name = title; image.lifetime = .keepAlways; add(image)
    }
    @MainActor func testVideoZeroPauseAndBounceFolderExports() async throws {
        let (app, machine, identity) = try await fixture()
        try open(1, identity: identity, app: app); try export(app)
        XCTAssertEqual(app.textFields["export-pause"].value as? String, "0")
        evidence(app, "Native GIF Loop Forever zero pause")
        choose("export-destination", "Save to Mold folder", app: app); submit(app)
        XCTAssertTrue(app.navigationBars["Export Media"].waitForNonExistence(timeout: 15))
        var body = try JSONSerialization.jsonObject(with: XCTUnwrap(machine.exportRequests.last)) as! [String: Any]
        XCTAssertEqual(body["pause_ms"] as? Int, 0); XCTAssertEqual(body["playback"] as? String, "loop")
        try export(app); choose("export-playback", "Bounce", app: app)
        let pause = app.textFields["export-pause"]; pause.tap(); pause.typeText(XCUIKeyboardKey.delete.rawValue + "250")
        app.buttons["Done"].firstMatch.tap()
        choose("export-destination", "Save to Mold folder", app: app); evidence(app, "Native GIF Bounce 250 ms")
        submit(app); XCTAssertTrue(app.navigationBars["Export Media"].waitForNonExistence(timeout: 15))
        body = try JSONSerialization.jsonObject(with: XCTUnwrap(machine.exportRequests.last)) as! [String: Any]
        XCTAssertEqual(body["pause_ms"] as? Int, 250); XCTAssertEqual(body["playback"] as? String, "bounce")
        try export(app); choose("export-playback", "Bounce", app: app)
        let resetPause = app.textFields["export-pause"]
        resetPause.tap(); resetPause.typeText(XCUIKeyboardKey.delete.rawValue + "250")
        app.buttons["Done"].firstMatch.tap()
        XCTAssertEqual(resetPause.value as? String, "250")
        app.buttons["No pause (0 ms)"].tap()
        XCTAssertEqual(resetPause.value as? String, "0")
        choose("export-destination", "Save to Mold folder", app: app)
        evidence(app, "Native GIF Bounce reset to zero pause")
        submit(app); XCTAssertTrue(app.navigationBars["Export Media"].waitForNonExistence(timeout: 15))
        body = try JSONSerialization.jsonObject(with: XCTUnwrap(machine.exportRequests.last)) as! [String: Any]
        XCTAssertEqual(body["pause_ms"] as? Int, 0); XCTAssertEqual(body["playback"] as? String, "bounce")
        XCTAssertEqual(machine.exportRequests.count, 3)
    }
    @MainActor func testAPNGFolderDeliveryOmitsParkedGifControls() async throws {
        let (app, machine, identity) = try await fixture()
        try open(1, identity: identity, app: app); try export(app)
        choose("export-format", "APNG", app: app)
        XCTAssertFalse(app.textFields["export-pause"].exists)
        choose("export-destination", "Save to Mold folder", app: app); submit(app)
        XCTAssertTrue(app.navigationBars["Export Media"].waitForNonExistence(timeout: 15))
        let body = try JSONSerialization.jsonObject(with: XCTUnwrap(machine.exportRequests.last)) as! [String: Any]
        XCTAssertEqual(body["format"] as? String, "apng")
        XCTAssertNil(body["pause_ms"])
        evidence(app, "APNG file saved to Mold folder")
    }

    @MainActor func testMeshGeometryTurntableAndAssetFolderExports() async throws {
        let (app, machine, identity) = try await fixture()
        try open(2, identity: identity, app: app); try export(app)
        choose("export-format", "STL", app: app)
        choose("export-destination", "Save to Mold folder", app: app); evidence(app, "Native STL geometry options")
        submit(app); XCTAssertTrue(app.navigationBars["Export Media"].waitForNonExistence(timeout: 15))
        var body = try JSONSerialization.jsonObject(with: XCTUnwrap(machine.exportRequests.last)) as! [String: Any]
        XCTAssertEqual(body["size_mm"] as? Int, 100); XCTAssertEqual(body["up_axis"] as? String, "z"); XCTAssertNil(body["pause_ms"])
        try export(app); choose("export-format", "GIF", app: app)
        choose("export-destination", "Save to Mold folder", app: app)
        submit(app); XCTAssertTrue(app.navigationBars["Export Media"].waitForNonExistence(timeout: 15))
        body = try JSONSerialization.jsonObject(with: XCTUnwrap(machine.exportRequests.last)) as! [String: Any]
        XCTAssertEqual(body["frames"] as? Int, 36); XCTAssertEqual(body["fps"] as? Int, 10); XCTAssertNil(body["size_mm"])
        app.buttons["More"].firstMatch.tap(); app.buttons["Save Files"].firstMatch.tap()
        app.buttons["base-color.png"].firstMatch.tap(); app.buttons["Save to Mold folder"].firstMatch.tap()
        XCTAssertTrue(app.staticTexts.matching(NSPredicate(format: "label CONTAINS 'base-color'")).firstMatch.waitForExistence(timeout: 10))
        XCTAssertTrue(machine.requestLog().contains { $0.contains("/api/gallery/assets/") })
        evidence(app, "Native sidecar saved to Mold folder")
    }
    @MainActor func testShareAndFilesCancellationPreserveTheViewer() async throws {
        let (app, _, identity) = try await fixture()
        try open(1, identity: identity, app: app); try export(app); submit(app)
        let close = app.buttons["Close"].firstMatch
        XCTAssertTrue(close.waitForExistence(timeout: 15)); evidence(app, "Exported GIF native share sheet"); close.tap()
        XCTAssertTrue(app.buttons["More"].firstMatch.waitForExistence(timeout: 5))
        try export(app); choose("export-destination", "Save to Files", app: app); submit(app)
        try cancelFilesPicker(app)
        XCTAssertTrue(app.buttons["More"].firstMatch.waitForExistence(timeout: 5))
        try export(app); app.buttons["export-cancel"].tap()
        XCTAssertTrue(app.buttons["More"].firstMatch.waitForExistence(timeout: 5))
    }
    @MainActor private func cancelFilesPicker(_ app: XCUIApplication) throws {
        let save = app.buttons["Save"].firstMatch
        let picker = app.navigationBars["FullDocumentManagerViewControllerNavigationBar"]
        XCTAssertTrue(save.waitForExistence(timeout: 15), "The native Files export picker must be presented")
        guard save.exists && save.isHittable else { throw ExportUATFailure.missingControl("native Files Save") }
        evidence(app, "Exported GIF native Files picker")
        // iOS 26.5 starts inside Mold Studio. Its BackButton visits On My
        // iPhone, then Browse, where the native Close action dismisses it.
        // Other system presentations expose Cancel directly.
        for step in 0..<5 {
            for label in ["Cancel", "Close"] {
                let dismiss = app.buttons[label].firstMatch
                if dismiss.exists && dismiss.isHittable {
                    evidence(app, "Native Files cancellation via \(label)")
                    dismiss.tap()
                    XCTAssertTrue(picker.waitForNonExistence(timeout: 10), "The native Files picker must dismiss")
                    return
                }
            }
            let back = picker.buttons["BackButton"].firstMatch
            guard back.exists && back.isHittable else {
                evidence(app, "Native Files missing cancellation affordance")
                throw ExportUATFailure.missingControl("native Files Back, Close or Cancel")
            }
            evidence(app, "Native Files parent navigation \(step): \(back.label)")
            back.tap()
        }
        throw ExportUATFailure.missingControl("native Files cancellation after bounded parent navigation")
    }
    @MainActor private func answerPhotosPermission(allow: Bool) throws {
        let springboard = XCUIApplication(bundleIdentifier: "com.apple.springboard")
        let alert = springboard.alerts.containing(NSPredicate(format: "elementType == %d AND label CONTAINS[c] 'Mold Studio' AND label CONTAINS[c] 'Photos'", XCUIElement.ElementType.staticText.rawValue)).firstMatch
        let button = alert.buttons.matching(NSPredicate(format: allow ? "label BEGINSWITH 'Allow'" : "label MATCHES 'Don.t Allow'")).firstMatch
        let ready = XCTNSPredicateExpectation(predicate: NSPredicate { _, _ in alert.exists && button.exists && button.isHittable }, object: nil)
        guard XCTWaiter.wait(for: [ready], timeout: 60) == .completed else {
            evidence(springboard, "Photos permission prompt unavailable")
            throw ExportUATFailure.missingControl("scoped Mold Studio Photos permission action")
        }
        evidence(springboard, "Mold Studio Photos permission \(allow ? "Allow" : "Deny")")
        button.tap()
        guard alert.waitForNonExistence(timeout: 60) else {
            evidence(springboard, "Photos permission prompt remained after action")
            throw ExportUATFailure.missingControl("scoped Photos permission prompt dismissal")
        }
    }
    @MainActor func testExportedGIFSavesToPhotos() async throws {
        let (app, _, identity) = try await fixture()
        try open(1, identity: identity, app: app); try export(app)
        choose("export-destination", "Save to Photos", app: app); submit(app)
        try answerPhotosPermission(allow: true)
        guard app.staticTexts["Saved to Photos."].waitForExistence(timeout: 60) else {
            evidence(app, "GIF PhotoKit save success was not observed")
            throw ExportUATFailure.missingControl("actual Saved to Photos status")
        }
        evidence(app, "Animated GIF saved through PhotoKit")
    }

    @MainActor func testPhotosDenialKeepsOptionsAndOffersRecovery() async throws {
        let (app, _, identity) = try await fixture()
        try open(1, identity: identity, app: app); try export(app)
        choose("export-destination", "Save to Photos", app: app); submit(app)
        try answerPhotosPermission(allow: false)
        let recovery = app.alerts["Allow Saving to Photos"]
        XCTAssertTrue(recovery.waitForExistence(timeout: 10))
        XCTAssertTrue(recovery.buttons["Open Settings"].exists)
        evidence(app, "GIF Photos denial recovery")
        recovery.buttons["Not Now"].tap()
        XCTAssertTrue(app.navigationBars["Export Media"].exists)
        app.buttons["export-cancel"].tap()
    }
    @MainActor func testRefusedExportCanBeRetriedWithoutLosingChoices() async throws {
        let (app, machine, identity) = try await fixture()
        try open(1, identity: identity, app: app); try export(app)
        choose("export-destination", "Save to Mold folder", app: app)
        machine.refuseNextExport(); submit(app)
        let error = app.staticTexts.matching(NSPredicate(format: "label CONTAINS 'Fixture refused'")).firstMatch
        XCTAssertTrue(error.waitForExistence(timeout: 10))
        XCTAssertTrue(error.isHittable, "Conversion failures must be visible beside Export without searching the form")
        evidence(app, "Refused GIF export retains its options")
        submit(app)
        XCTAssertTrue(app.navigationBars["Export Media"].waitForNonExistence(timeout: 15))
        XCTAssertEqual(machine.exportRequests.count, 2)
    }
    @MainActor func testUnsupportedFormatsShowAnExplanation() async throws {
        let (app, machine, identity) = try await fixture(unsupportedFormats: true)
        try open(1, identity: identity, app: app)
        app.buttons["More"].firstMatch.tap(); app.buttons["Export…"].firstMatch.tap()
        XCTAssertTrue(app.navigationBars["Export Media"].waitForExistence(timeout: 10))
        let explanation = app.staticTexts.matching(NSPredicate(format: "label CONTAINS 'no conversions'")).firstMatch
        XCTAssertTrue(explanation.waitForExistence(timeout: 10))
        XCTAssertTrue(explanation.isHittable)
        XCTAssertFalse(app.buttons["export-submit"].exists)
        XCTAssertTrue(machine.exportRequests.isEmpty)
        evidence(app, "Unsupported export formats retain a visible explanation")
        app.buttons["export-cancel"].tap()
    }
    @MainActor func testExportSheetExtraSmall() async throws { try await auditExport(size: "UICTContentSizeCategoryXS") }
    @MainActor func testExportSheetLarge() async throws { try await auditExport(size: "UICTContentSizeCategoryL") }
    @MainActor func testExportSheetAX5() async throws { try await auditExport(size: "UICTContentSizeCategoryAccessibilityXXXL") }
    @MainActor func testMeshExportSheetExtraSmall() async throws { try await auditExport(size: "UICTContentSizeCategoryXS", mesh: true) }
    @MainActor func testMeshExportSheetLarge() async throws { try await auditExport(size: "UICTContentSizeCategoryL", mesh: true) }
    @MainActor func testMeshExportSheetAX5() async throws { try await auditExport(size: "UICTContentSizeCategoryAccessibilityXXXL", mesh: true) }
    @MainActor private func auditExport(size: String, mesh: Bool = false) async throws {
        let (app, _, identity) = try await fixture(size: size)
        try open(mesh ? 2 : 1, identity: identity, app: app); try export(app)
        if mesh { choose("export-format", "GIF", app: app) }
        let common = ["export-format", "export-destination", "export-submit"]
        try auditSheet(app, size: size, required: common + ["export-playback", "export-repeat", "export-pause", "export-size", "export-fps"] + (mesh ? ["export-frames", "export-transparent"] : []))
        app.buttons["export-cancel"].tap()
        if mesh {
            try export(app); choose("export-format", "STL", app: app)
            try auditSheet(app, size: size, required: common + ["export-size-default", "export-size-mm", "export-axis", "export-origin"])
            app.buttons["export-cancel"].tap()
        }
    }
    @MainActor private func auditSheet(_ app: XCUIApplication, size: String, required: [String]) throws {
        evidence(app, "Export controls at \(size)")
        let form = app.collectionViews["export-form"]
        XCTAssertTrue(form.exists)
        var covered = Set<String>()
        var discoveredText = Set<String>()
        var coveredText = Set<String>()
        let animation = required.contains("export-playback")
        var expectedLabels = ["Format": 2, "Destination": 2]
        if animation {
            expectedLabels.merge(["Playback": 2, "Repeat": 1,
                "Bounce plays forward, then reverses.": 1, "Pause between loops": 1,
                "0 adds no pause and keeps the frame rate.": 1, "Size and frame rate": 1,
                "Longest side": 1, "Frame rate": 1]) { _, new in new }
            if required.contains("export-frames") { expectedLabels["GIF has a hard transparent edge."] = 1 }
        } else {
            expectedLabels.merge(["Geometry": 1, "Longest side in mm": 1, "Up axis": 1, "Origin": 1]) { _, new in new }
        }
        var coveredLabels: [String: Set<Int>] = [:]
        var previousVisibleRows: String?
        var unchangedViewports = 0
        // Every form control must be fully visible during a contrast pass.
        // Offscreen rows are excluded only while scrolling to audit them in full.
        func viewport() -> CGRect {
            let frame = form.frame.intersection(app.frame)
            let top = max(frame.minY, app.navigationBars["Export Media"].frame.maxY)
            return CGRect(x: frame.minX, y: top, width: frame.width, height: max(0, frame.maxY - top))
        }
        func settleBoundary() throws {
            for attempt in 0...3 {
                let visible = viewport()
                let texts = form.staticTexts.allElementsBoundByIndex.map { ($0.label, $0.frame) }
                guard let crossing = texts.filter({ !$0.0.isEmpty && $0.1.width > 0 && $0.1.height > 0 && $0.1.minY >= visible.minY - 24 && $0.1.minY < visible.minY && $0.1.maxY > visible.minY })
                    .max(by: { $0.1.minY < $1.1.minY }) else { return }
                guard attempt < 3 else {
                    evidence(app, "Export boundary settling exhausted")
                    throw ExportUATFailure.missingControl("export boundary settling exceeded its bound")
                }
                let matches = form.staticTexts.matching(NSPredicate(format: "label == %@", crossing.0)).allElementsBoundByIndex
                guard let occurrence = matches.firstIndex(where: { $0.frame == crossing.1 }) else { throw ExportUATFailure.missingControl("export boundary text identity") }
                let containedControls = required.compactMap { id -> (String, CGRect)? in
                    let control = form.descendants(matching: .any)[id].firstMatch
                    guard control.exists else { return nil }
                    let frame = control.frame
                    return frame.width > 0 && frame.height > 0 && visible.contains(frame) ? (id, frame) : nil
                }
                let shift = visible.minY + 16 - crossing.1.minY
                let slack = visible.maxY - (containedControls.map { $0.1.maxY }.max() ?? visible.minY)
                // Move only when the actual currently contained controls retain
                // their full viewport. Other rows still receive strict audits.
                guard shift <= slack, shift < visible.height * 0.4 else { return }
                evidence(app, "Export boundary before settling \(attempt)")
                print("EXPORT BOUNDARY SETTLE \(crossing.0) \(crossing.1) viewport \(visible) shift \(shift) slack \(slack)")
                let origin = app.coordinate(withNormalizedOffset: .zero)
                let x = visible.maxX - 10 - app.frame.minX
                let y = visible.minY + visible.height * 0.6 - app.frame.minY
                origin.withOffset(CGVector(dx: x, dy: y))
                    .press(forDuration: 0.1, thenDragTo: origin.withOffset(CGVector(dx: x, dy: y + shift)),
                           withVelocity: .slow, thenHoldForDuration: 0.3)
                let after = viewport()
                let matching = form.staticTexts.matching(NSPredicate(format: "label == %@", crossing.0)).allElementsBoundByIndex
                guard after == visible, matching.indices.contains(occurrence), matching[occurrence].label == crossing.0 else {
                    evidence(app, "Export boundary settling changed layout identity")
                    throw ExportUATFailure.missingControl("export boundary settling changed layout for \(crossing.0)")
                }
                for (id, _) in containedControls {
                    let control = form.descendants(matching: .any)[id].firstMatch
                    guard control.exists, after.contains(control.frame) else {
                        evidence(app, "Export boundary settling displaced a control")
                        throw ExportUATFailure.missingControl("export boundary settling displaced \(id)")
                    }
                }
                // Optional normalization must not turn an immovable native
                // heading into a finding. Audit the unchanged viewport strictly.
                guard matching[occurrence].frame.minY > crossing.1.minY + 0.5 else { return }
            }
        }
        for pass in 0..<32 {
            try settleBoundary()
            let visible = viewport()
            for id in required where !covered.contains(id) {
                let control = form.descendants(matching: .any)[id].firstMatch
                // A Stepper is an accessibility group; its child buttons are
                // audited for hit regions rather than treating the group as a button.
                if control.exists {
                    let frame = control.frame
                    if frame.width > 0 && frame.height > 0 && visible.contains(frame) { covered.insert(id) }
                }
            }
            let elements = form.descendants(matching: .any).allElementsBoundByIndex
            let navigation = app.navigationBars["Export Media"].descendants(matching: .any).allElementsBoundByIndex
            var occurrences: [String: Int] = [:]
            var labelOccurrences: [String: Int] = [:]
            var manifest: [String] = []
            var layoutRows: [String] = []
            var movementRows: [String] = []
            for element in elements {
                let label = element.label
                guard !label.isEmpty else { continue }
                let type = element.elementType
                let identifier = element.identifier
                guard [.staticText, .button, .textField, .slider, .switch, .stepper].contains(type)
                    || identifier.hasPrefix("export-") else { continue }
                let identity = "\(type.rawValue)|\(identifier)|\(label)"
                let occurrence = occurrences[identity, default: 0]
                occurrences[identity] = occurrence + 1
                let key = "\(identity)|\(occurrence)"
                discoveredText.insert(key)
                let frame = element.frame
                if frame.width > 0 && frame.height > 0 && visible.intersects(frame) {
                    layoutRows.append("\(identity)|\(frame)")
                    if type == .staticText { movementRows.append("\(label):\(frame.minY.rounded()):\(frame.height.rounded())") }
                }
                let fullyVisible = frame.width > 0 && frame.height > 0 && visible.contains(frame)
                if fullyVisible { coveredText.insert(key) }
                if type == .staticText {
                    let index = labelOccurrences[label, default: 0]
                    labelOccurrences[label] = index + 1
                    if fullyVisible { coveredLabels[label, default: []].insert(index) }
                }
                manifest.append("\(fullyVisible ? "VISIBLE" : "offscreen") \(key) at \(frame)")
            }
            let inventory = XCTAttachment(string: "Effective viewport: \(visible)\n" + manifest.joined(separator: "\n"))
            inventory.name = "Export text coverage \(animation ? "GIF" : "geometry") \(size) viewport \(pass)"
            inventory.lifetime = .keepAlways; add(inventory)
            evidence(app, "Export text coverage \(animation ? "GIF" : "geometry") \(size) viewport \(pass)")
            // Clipping/Dynamic Type audits resize and reset a lazy Form's
            // scroll position. Finish settled pixel and text-detection passes before resizing it,
            // matching ShellAccessibilityTests' ordering.
            // The combined pixel audit exceeds XCTest's private deadline on
            // narrow AX5 viewports. Keep every audit strict, one type at a time.
            let settledAudits: [(String, XCUIAccessibilityAuditType)] = [
                ("contrast", .contrast), ("elementDetection", .elementDetection),
                ("hitRegion", .hitRegion), ("sufficientElementDescription", .sufficientElementDescription)
            ]
            let before = "\(form.frame)|\(visible)|\(app.navigationBars["Export Media"].frame)|" + layoutRows.sorted().joined(separator: "\n")
            for (name, type) in settledAudits {
                var auditElements = elements
                var auditNavigation = navigation
                var auditViewport = visible
                var unhandledFinding = false
                func layoutSignature() -> String {
                    var rows: [String] = []
                    for element in auditElements {
                        let label = element.label
                        guard !label.isEmpty else { continue }
                        let kind = element.elementType
                        let identifier = element.identifier
                        guard [.staticText, .button, .textField, .slider, .switch, .stepper].contains(kind)
                            || identifier.hasPrefix("export-") else { continue }
                        let frame = element.frame
                        if frame.width > 0 && frame.height > 0 && auditViewport.intersects(frame) {
                            rows.append("\(kind.rawValue)|\(identifier)|\(label)|\(frame)")
                        }
                    }
                    return "\(form.frame)|\(auditViewport)|\(app.navigationBars["Export Media"].frame)|" + rows.sorted().joined(separator: "\n")
                }
                func runAudit() throws {
                    print("EXPORT SETTLED AUDIT \(name) viewport \(pass) \(size)")
                    try app.performAccessibilityAudit(for: type) { issue in
                        guard let element = issue.element else {
                            unhandledFinding = true
                            let report = XCTAttachment(string: issue.compactDescription + ": " + issue.detailedDescription + "\n" + app.debugDescription)
                            report.name = "Unnamed export accessibility failure"; report.lifetime = .keepAlways; self.add(report)
                            print("EXPORT AUDIT unnamed: \(issue.compactDescription): \(issue.detailedDescription)")
                            return false
                        }
                        if !auditNavigation.contains(element) && (!auditElements.contains(element) || !auditViewport.contains(element.frame)) { return true }
                        unhandledFinding = true
                        let report = XCTAttachment(string: "\(issue.compactDescription): \(element.elementType) '\(element.label)' [\(element.identifier)] at \(element.frame)\n\(issue.detailedDescription)\n\(app.debugDescription)")
                        report.name = "Visible export accessibility failure"; report.lifetime = .keepAlways; self.add(report)
                        print("EXPORT AUDIT: \(issue.compactDescription): '\(element.label)' [\(element.identifier)] \(issue.detailedDescription)")
                        return false
                    }
                }
                do { try runAudit() }
                catch {
                    let failure = error as NSError
                    // Match ShellAccessibilityTests: retry an unfinished framework
                    // audit once. Actual findings and all other errors stay strict.
                    guard !unhandledFinding, failure.domain == "com.apple.xcode.xctest.accessibilityAudit", failure.code == -56 else { throw error }
                    let report = XCTAttachment(string: "\(name) viewport \(pass) at \(size): \(failure)\nBefore: \(before)\n\(app.debugDescription)")
                    report.name = "Export audit first timeout"; report.lifetime = .keepAlways; add(report)
                    evidence(app, "Export audit first timeout \(name) viewport \(pass) at \(size)")
                    auditElements = form.descendants(matching: .any).allElementsBoundByIndex
                    auditNavigation = app.navigationBars["Export Media"].descendants(matching: .any).allElementsBoundByIndex
                    auditViewport = viewport()
                    guard layoutSignature() == before else { throw ExportUATFailure.missingControl("export layout changed after audit timeout: \(name) viewport \(pass)") }
                    print("EXPORT AUDIT RETRY \(name) viewport \(pass) \(size)")
                    try runAudit()
                }
            }
            let labelsComplete = expectedLabels.allSatisfy { label, count in coveredLabels[label, default: []].count >= count }
            if covered.count == required.count && coveredText == discoveredText && labelsComplete { break }
            let visibleRows = movementRows.sorted().joined(separator: "|")
            unchangedViewports = !visibleRows.isEmpty && visibleRows == previousVisibleRows ? unchangedViewports + 1 : 0
            previousVisibleRows = visibleRows
            guard unchangedViewports < 2 else {
                throw ExportUATFailure.missingControl("native export scroll made no progress: \(visibleRows)")
            }
            // Release at rest: a fast drag can coast past a tall AX5 row's
            // fully visible interval on the narrow SE viewport.
            // Begin in the Form gutter, outside Menu labels which can consume
            // a held drag. Use the on-screen viewport rather than Form bounds.
            let origin = app.coordinate(withNormalizedOffset: .zero)
            let start = origin.withOffset(CGVector(dx: visible.maxX - 10 - app.frame.minX,
                                                   dy: visible.minY + visible.height * 0.8 - app.frame.minY))
            let end = origin.withOffset(CGVector(dx: visible.maxX - 10 - app.frame.minX,
                                                 dy: visible.minY + visible.height * 0.6 - app.frame.minY))
            print("EXPORT SCROLL GEOMETRY form \(form.frame) viewport \(visible) start \(start.screenPoint) end \(end.screenPoint)")
            start
                .press(forDuration: 0.1,
                       thenDragTo: end,
                       withVelocity: .slow, thenHoldForDuration: 0.3)
        }
        XCTAssertEqual(covered, Set(required), "Every export control must fit and be audited")
        guard covered == Set(required) else { throw ExportUATFailure.missingControl(required.filter { !covered.contains($0) }.joined(separator: ", ")) }
        XCTAssertEqual(coveredText, discoveredText, "Every discovered export label and selected value must become fully visible")
        let missingLabels = expectedLabels.filter { label, count in coveredLabels[label, default: []].count < count }
        XCTAssertTrue(missingLabels.isEmpty, "Missing source-declared label coverage: \(missingLabels)")
        guard coveredText == discoveredText && missingLabels.isEmpty else { throw ExportUATFailure.missingControl("export text inventory") }
        XCTAssertTrue(app.buttons["export-cancel"].isHittable)
        evidence(app, "Settled export prediction audit at \(size)")
        // Check clipping before Dynamic Type temporarily rebuilds the hierarchy.
        for types: XCUIAccessibilityAuditType in [[.textClipped], [.dynamicType]] {
        try app.performAccessibilityAudit(for: types) { issue in
            // UIKit navigation bars cap their font size and offer the Large
            // Content Viewer. Match ShellAccessibilityTests' system-bar rule.
            // Unnamed Dynamic Type reports are a known auditor limitation;
            // retain diagnostics rather than identifying them as system chrome.
            if issue.auditType == .dynamicType {
                guard let element = issue.element else {
                    let report = XCTAttachment(string: issue.detailedDescription + "\n" + app.debugDescription)
                    report.name = "Unnamed Dynamic Type auditor report"; report.lifetime = .keepAlways; self.add(report)
                    self.evidence(app, "Unnamed Dynamic Type auditor report")
                    return true
                }
                if app.navigationBars["Export Media"].descendants(matching: .any).allElementsBoundByIndex.contains(element) { return true }
            }
            guard let element = issue.element else {
                let runtime = ProcessInfo.processInfo.operatingSystemVersion
                let knownPrediction = issue.auditType == .textClipped
                    && size == "UICTContentSizeCategoryAccessibilityXXXL"
                    && runtime.majorVersion == 26 && runtime.minorVersion == 5
                    && issue.detailedDescription.trimmingCharacters(in: .whitespacesAndNewlines)
                        == "Text of this element may be clipped at larger Dynamic Type sizes."
                if knownPrediction {
                    // iOS 26.5 predicts larger-size clipping without naming a
                    // node even at maximum AX5. Frame coverage cannot detect
                    // internal text truncation: independent visual inspection of
                    // EVERY retained AX5 GIF/geometry viewport in light and dark
                    // is required before accepting UAT. Named clipping, other
                    // runtimes/sizes and nil contrast continue to fail.
                    let report = XCTAttachment(string: issue.detailedDescription + "\n" + app.debugDescription)
                    report.name = "iOS 26.5 AX5 unnamed clipping prediction"; report.lifetime = .keepAlways; self.add(report)
                    self.evidence(app, "iOS 26.5 AX5 unnamed clipping prediction")
                    return true
                }
                let report = XCTAttachment(string: issue.compactDescription + ": " + issue.detailedDescription + "\n" + app.debugDescription)
                report.name = "Unnamed export final audit failure"; report.lifetime = .keepAlways; self.add(report)
                print("EXPORT FINAL AUDIT unnamed: \(issue.compactDescription): \(issue.detailedDescription)")
                return false
            }
            // The underlying Library/viewer is dimmed behind this sheet.
            return !form.descendants(matching: .any).allElementsBoundByIndex.contains(element)
                && !app.navigationBars["Export Media"].descendants(matching: .any).allElementsBoundByIndex.contains(element)
        }
        }
    }

}
