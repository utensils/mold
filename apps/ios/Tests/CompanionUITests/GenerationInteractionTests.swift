import XCTest
import UIKit

/// Exercise a saved-machine screen, not just first-run empty states.
final class GenerationInteractionTests: XCTestCase {
    override func setUp() {
        super.setUp()
        acceptCompanionPermissions()
    }

    @MainActor private func launch(size: String = "UICTContentSizeCategoryL") -> XCUIApplication {
        continueAfterFailure = false
        let app = XCUIApplication()
        app.launchArguments = ["-UIPreferredContentSizeCategoryName", "UICTContentSizeCategoryL"]
        app.launch()
        app.buttons["Generate"].firstMatch.tap()
        if app.buttons["Add a Machine…"].firstMatch.exists {
            app.buttons["Add a Machine…"].firstMatch.tap()
            app.buttons.matching(NSPredicate(format: "label BEGINSWITH 'Enter an Address'")).firstMatch.tap()
            let name = app.textFields["machine-name"]
            XCTAssertTrue(name.waitForExistence(timeout: 5))
            name.tap()
            name.typeText("UAT Machine")
            let address = app.textFields["machine-address"]
            address.tap()
            address.typeText("127.0.0.1:9")
            app.buttons["Add"].firstMatch.tap()
            app.buttons["Generate"].firstMatch.tap()
        }
        if size != "UICTContentSizeCategoryL" {
            app.terminate()
            app.launchArguments = ["-UIPreferredContentSizeCategoryName", size]
            app.launch()
            app.buttons["Generate"].firstMatch.tap()
        }
        return app
    }

    @MainActor func testAccessibilityComposerAndMachineCardsFitTheScreen() throws {
        try XCTSkipIf(UIDevice.current.userInterfaceIdiom == .pad,
                      "Phone composer width contract; iPad composition is covered by the shell audit")
        let app = launch(size: "UICTContentSizeCategoryAccessibilityXXXL")
        let form = app.scrollViews["phone-generate-form"]
        XCTAssertTrue(form.waitForExistence(timeout: 5))
        let submit = app.buttons["submit-generation"]
        XCTAssertTrue(submit.isHittable)
        XCTAssertGreaterThan(submit.frame.width, form.frame.width * 0.7,
                             "Large-text Generate should occupy the form width")
        capture(app)
        XCTAssertTrue(app.navigateToDestination("Machines", shortcut: "5"))
        let card = app.descendants(matching: .any).matching(NSPredicate(format: "identifier BEGINSWITH 'machine-card-'")).firstMatch
        XCTAssertTrue(card.waitForExistence(timeout: 5))
        XCTAssertGreaterThanOrEqual(card.frame.minX, app.frame.minX)
        XCTAssertLessThanOrEqual(card.frame.maxX, app.frame.maxX)
        capture(app)
    }

    @MainActor func testOfflineQueueExplanationScrollsAtLargestText() throws {
        let app = launch(size: "UICTContentSizeCategoryAccessibilityXXXL")
        Self.removeOfflineQueueTestMachine(from: app)
        addTeardownBlock {
            await MainActor.run { Self.removeOfflineQueueTestMachine(from: app) }
        }
        app.buttons["Add a Machine"].firstMatch.tap()
        let enterAddress = app.buttons.matching(NSPredicate(format: "label BEGINSWITH 'Enter an Address'")).firstMatch
        XCTAssertTrue(enterAddress.waitForExistence(timeout: 5))
        enterAddress.tap()
        let name = app.textFields["machine-name"]
        XCTAssertTrue(name.waitForExistence(timeout: 5))
        name.tap()
        name.typeText("Offline Queue Test")
        let address = app.textFields["machine-address"]
        address.tap()
        address.typeText("127.0.0.1:65534")
        app.buttons["Add"].firstMatch.tap()
        XCTAssertTrue(name.waitForNonExistence(timeout: 5), "Adding the test machine must dismiss its sheet")
        XCTAssertTrue(app.navigateToDestination("Queue", shortcut: "3"))
        let message = app.staticTexts["Some machines could not provide their queues. Check Machines to reconnect, then pull to refresh."]
        XCTAssertTrue(message.waitForExistence(timeout: 5))
        let action = app.buttons["Check Machines"]
        for _ in 0..<8 where message.frame.maxY > action.frame.minY {
            // A full-screen swipe starts on the pinned button on small phones.
            // Drag the visible explanation instead, as a person would.
            let origin = app.coordinate(withNormalizedOffset: .zero)
            let start = origin.withOffset(CGVector(dx: app.frame.midX, dy: action.frame.minY - 24))
            let end = origin.withOffset(CGVector(dx: app.frame.midX, dy: app.navigationBars.firstMatch.frame.maxY + 24))
            start.press(forDuration: 0.05, thenDragTo: end)
        }
        capture(app)
        XCTAssertLessThanOrEqual(message.frame.maxY, action.frame.minY,
                                 "The final line must scroll above the pinned action")
        XCTAssertTrue(action.isHittable)
    }

    /// Use the detail page: a long press can select the card's address text
    /// instead of opening its context menu. Clear stale fixtures before setup
    /// and after every outcome so light/dark runs cannot share a duplicate.
    @MainActor private static func removeOfflineQueueTestMachine(from app: XCUIApplication) {
        if app.textFields["machine-name"].exists { app.buttons["Cancel"].firstMatch.tap() }
        XCTAssertTrue(app.navigateToDestination("Machines", shortcut: "5"))
        let card = app.buttons.matching(NSPredicate(format: "label CONTAINS %@", "127.0.0.1:65534")).firstMatch
        for _ in 0..<8 where !card.isHittable { app.swipeUp() }
        guard card.exists else { return }
        XCTAssertTrue(card.isHittable)
        card.tap()
        let remove = app.buttons["Remove…"].firstMatch
        for _ in 0..<8 where !remove.isHittable { app.swipeUp() }
        XCTAssertTrue(remove.isHittable)
        remove.tap()
        let confirm = app.buttons["Remove"].firstMatch
        XCTAssertTrue(confirm.waitForExistence(timeout: 5))
        confirm.tap()
        XCTAssertTrue(app.navigationBars["Machines"].waitForExistence(timeout: 5))
        XCTAssertFalse(card.exists, "The offline queue fixture must not survive cleanup")
    }

    @MainActor func testFloatingBarKeepsModelsInSidebar() throws {
        let app = launch(size: "UICTContentSizeCategoryAccessibilityXXXL")
        guard app.buttons["ToggleSideBar"].exists else { return }
        let favourites = app.descendants(matching: .any)["Favourites"].firstMatch
        if favourites.exists, favourites.isHittable { app.buttons["ToggleSideBar"].tap() }
        XCTAssertFalse(app.buttons["Models"].firstMatch.exists)
        XCTAssertFalse(app.buttons["Next Page"].firstMatch.exists)
        XCTAssertTrue(app.navigateToDestination("Machines", shortcut: "5"))
        XCTAssertTrue(app.navigateToDestination("Models", shortcut: "4"))
    }

    @MainActor func testPromptAcceptsTypingAndKeepsKeyboard() throws {
        try checkPrompt(size: "UICTContentSizeCategoryL")
    }

    @MainActor func testLargestTextPromptKeepsKeyboard() throws {
        try checkPrompt(size: "UICTContentSizeCategoryAccessibilityXXXL")
    }

    @MainActor private func checkPrompt(size: String) throws {
        let app = launch(size: size)
        let prompt = app.descendants(matching: .any)["generation-prompt"].firstMatch
        XCTAssertTrue(prompt.waitForExistence(timeout: 5))
        prompt.tap()
        XCTAssertTrue(app.keyboards.firstMatch.waitForExistence(timeout: 5))
        let before = prompt.value as? String
        if let before, !before.isEmpty, before != prompt.placeholderValue {
            prompt.tap(withNumberOfTaps: 3, numberOfTouches: 1)
        }
        let text = "A lighthouse at dusk \(UUID().uuidString.prefix(4))"
        XCTAssertNotEqual(before, text, "A persisted prompt must not satisfy the input assertion")
        prompt.typeText(text)
        // Match the observed exact AX value directly; a closure can spend its
        // deadline fetching snapshots on a loaded runner. Never type twice.
        let typed = XCTNSPredicateExpectation(
            predicate: NSPredicate(format: "value == %@", text), object: prompt
        )
        XCTAssertEqual(XCTWaiter.wait(for: [typed], timeout: 5), .completed,
                       "Prompt must retain typed text; actual value: \(String(describing: prompt.value))")
        XCTAssertTrue(app.keyboards.firstMatch.exists, "Typing must not replace the focused composer")
        capture(app)
    }

    @MainActor func testAppearanceChangesLiveAndPersistsAcrossLaunch() throws {
        let app = launch()
        defer { app.terminate() }
        app.buttons["Settings"].firstMatch.tap()
        let picker = app.descendants(matching: .any)["appearance-picker"].firstMatch
        XCTAssertTrue(picker.waitForExistence(timeout: 5))
        @discardableResult func choose(_ title: String, dark: Bool?) throws -> Double {
            picker.tap()
            let choice = app.buttons.matching(NSPredicate(format: "label == %@", title)).firstMatch
            XCTAssertTrue(choice.waitForExistence(timeout: 5), app.debugDescription)
            choice.tap()
            XCTAssertTrue(picker.label.contains(title) || (picker.value as? String)?.contains(title) == true,
                          "Appearance choice must be visible: \(picker.label), \(picker.value ?? "")")
            let brightness = try assertAppearancePixels(app, scope: picker.frame, dark: dark)
            let shot = XCTAttachment(screenshot: app.screenshot())
            shot.name = "Settings appearance \(title)"; shot.lifetime = .keepAlways; add(shot)
            return brightness
        }
        // Record the actual system palette; this test runs in both appearances.
        let systemBrightness = try choose("System", dark: nil)
        try choose("Dark", dark: true)
        app.buttons["Done"].firstMatch.tap()
        try assertAppearancePixels(app, scope: app.navigationBars.firstMatch.frame, dark: true, fraction: 0.25)
        app.terminate(); app.launch()
        XCTAssertTrue(app.navigateToDestination("Generate", shortcut: "1"))
        app.buttons["Settings"].firstMatch.tap()
        XCTAssertTrue(picker.waitForExistence(timeout: 5))
        XCTAssertTrue(picker.label.contains("Dark") || (picker.value as? String)?.contains("Dark") == true)
        try assertAppearancePixels(app, scope: picker.frame, dark: true)
        try choose("Light", dark: false)
        app.buttons["Done"].firstMatch.tap()
        try assertAppearancePixels(app, scope: app.navigationBars.firstMatch.frame, dark: false, fraction: 0.25)
        app.buttons["Settings"].firstMatch.tap()
        XCTAssertTrue(picker.waitForExistence(timeout: 5))
        let resetBrightness = try choose("System", dark: nil)
        XCTAssertEqual(resetBrightness, systemBrightness, accuracy: 0.1,
                       "System must restore the device palette")
        app.buttons["Done"].firstMatch.tap()
    }

    @MainActor @discardableResult private func assertAppearancePixels(
        _ app: XCUIApplication, scope: CGRect, dark: Bool?, fraction: CGFloat = 0.5
    ) throws -> Double {
        // Sample the blank center between a Form row's label/value, or the
        // navigation bar's left padding. Canonical RGBA avoids PNG byte-order assumptions.
        let image = try XCTUnwrap(app.screenshot().image.cgImage)
        let scale = CGFloat(image.width) / app.frame.width
        let context = try XCTUnwrap(CGContext(data: nil, width: image.width, height: image.height,
            bitsPerComponent: 8, bytesPerRow: image.width * 4, space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
        context.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
        let rgba = try XCTUnwrap(context.data).assumingMemoryBound(to: UInt8.self)
        let point = CGPoint(x: scope.minX + scope.width * fraction, y: scope.midY)
        let x = min(image.width - 1, max(0, Int(point.x * scale)))
        let y = min(image.height - 1, max(0, Int(point.y * scale)))
        let offset = (y * image.width + x) * 4
        let brightness = (Double(rgba[offset]) + Double(rgba[offset + 1]) + Double(rgba[offset + 2])) / 765
        if let dark {
            if dark { XCTAssertLessThan(brightness, 0.4, "Dark surface must paint dark pixels") }
            else { XCTAssertGreaterThan(brightness, 0.7, "Light surface must paint light pixels") }
        }
        return brightness
    }

    @MainActor func testSettingsPresentationAndAddMachineRoute() throws {
        let app = launch()
        XCTAssertTrue(app.navigateToDestination("Machines", shortcut: "5"))
        app.buttons["Settings"].firstMatch.tap()
        XCTAssertTrue(app.buttons["Done"].firstMatch.waitForExistence(timeout: 5))
        app.buttons["Done"].firstMatch.tap()
        XCTAssertTrue(app.navigationBars["Machines"].waitForExistence(timeout: 5))
        app.buttons["Settings"].firstMatch.tap()
        app.buttons["Add a Machine…"].firstMatch.tap()
        XCTAssertTrue(app.buttons["Cancel"].firstMatch.waitForExistence(timeout: 5))
        app.buttons["Cancel"].firstMatch.tap()
    }

    @MainActor func testLibraryShelvesAndSettingsAreReachableOnPhone() throws {
        try XCTSkipIf(UIDevice.current.userInterfaceIdiom == .pad,
                      "Phone Settings toolbar route; iPad sidebar Settings is covered by the shell audit")
        let app = launch(size: "UICTContentSizeCategoryAccessibilityXXXL")
        XCTAssertTrue(app.navigateToDestination("Library", shortcut: "2"))
        let shelves = app.navigationBars.buttons["All Prints"].firstMatch
        XCTAssertTrue(shelves.waitForExistence(timeout: 5))
        XCTAssertTrue(shelves.isHittable)
        shelves.tap()
        XCTAssertTrue(app.buttons["Favourites"].firstMatch.waitForExistence(timeout: 5))
        app.buttons["Favourites"].firstMatch.tap()
        XCTAssertTrue(app.navigationBars["Favourites"].waitForExistence(timeout: 5))
        app.buttons["Settings"].firstMatch.tap()
        XCTAssertTrue(app.navigationBars["Settings"].waitForExistence(timeout: 5))
        app.buttons["Done"].firstMatch.tap()
        XCTAssertTrue(app.navigationBars.buttons["Favourites"].firstMatch.exists)
    }

    @MainActor func testModelChooserAndDownloadRoute() throws {
        let app = launch()
        let chooser = app.buttons["choose-model"]
        for _ in 0..<4 where !chooser.isHittable { app.scrollViews["phone-generate-form"].swipeUp() }
        XCTAssertTrue(chooser.isHittable)
        chooser.tap()
        XCTAssertTrue(app.navigationBars["Choose a Model"].waitForExistence(timeout: 5))
        capture(app)
        let more = app.buttons["Get More Models…"].firstMatch
        for _ in 0..<8 where !more.isHittable { app.swipeUp() }
        XCTAssertTrue(more.isHittable)
        more.tap()
        XCTAssertTrue(app.navigationBars["Models"].waitForExistence(timeout: 5))
        capture(app)
    }

    @MainActor private func capture(_ app: XCUIApplication) {
        let shot = XCTAttachment(screenshot: app.screenshot())
        shot.lifetime = .keepAlways
        add(shot)
    }
}

@MainActor extension XCUIApplication {
    /// iPadOS can page the floating bar on the first tap at large text sizes.
    /// Require the destination, then retry the now-visible tab if necessary.
    func navigateToDestination(_ title: String, shortcut: String) -> Bool {
        let bar = navigationBars[title == "Library" ? "All Prints" : title]
        for _ in 0..<3 {
            let tab = buttons[title].firstMatch
            if tab.waitForExistence(timeout: 2), tab.isHittable {
                tab.tap()
            } else if buttons["Next Page"].firstMatch.exists, buttons["Next Page"].firstMatch.isHittable {
                buttons["Next Page"].firstMatch.tap()
                continue
            } else {
                typeKey(shortcut, modifierFlags: .command)
            }
            if bar.waitForExistence(timeout: 2) { return true }
        }
        return bar.exists
    }
}
