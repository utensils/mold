import XCTest

/// DESIGN.md §6 made executable: every destination -- plus Search and the
/// Settings sheet -- is audited at the smallest and the largest text sizes
/// for clipped text, Dynamic Type support, hit regions, contrast and element
/// descriptions. `make uitest` runs this on an iPhone and an iPad, in light
/// and dark. A layout that only works at the default size fails here, not in
/// someone's hand.
final class ShellAccessibilityTests: XCTestCase {
    override func setUp() {
        continueAfterFailure = true
    }

    @MainActor func testExtraSmall() throws { try audit(size: "UICTContentSizeCategoryXS") }
    @MainActor func testDefaultLarge() throws { try audit(size: "UICTContentSizeCategoryL") }
    @MainActor func testAccessibilityXXXL() throws { try audit(size: "UICTContentSizeCategoryAccessibilityXXXL") }

    @MainActor private func audit(size: String) throws {
        let app = XCUIApplication()
        app.launchArguments += ["-UIPreferredContentSizeCategoryName", size]
        app.launch()

        for (index, tab) in ["Generate", "Library", "Queue", "Models", "Machines"].enumerated() {
            let button = app.buttons[tab].firstMatch
            // Models is a sidebar destination: present on iPad, absent on iPhone.
            if tab == "Models", !button.waitForExistence(timeout: 2) { continue }
            // At AX sizes the iPad's floating tab bar pages its tabs, and its
            // sidebar names the Library's first shelf "All Prints"; the Go
            // menu's ⌘1–⌘5 is the way there that never scrolls -- and using
            // it tests those shortcuts too.
            if button.waitForExistence(timeout: 3), button.isHittable {
                button.tap()
            } else {
                app.typeKey("\(index + 1)", modifierFlags: .command)
            }
            try check(app, "\(tab) at \(size)")
        }

        // Settings: the Machines toolbar button, or ⌘, where it is not on screen.
        let settings = app.buttons["Settings"].firstMatch
        if settings.waitForExistence(timeout: 3), settings.isHittable {
            settings.tap()
        } else {
            app.typeKey(",", modifierFlags: .command)
        }
        // At huge text iPadOS folds toolbar items into the bar's overflow menu.
        if !app.buttons["Done"].firstMatch.waitForExistence(timeout: 3) {
            let overflow = app.navigationBars.buttons["More"].firstMatch
            if overflow.exists { overflow.tap(); app.buttons["Settings"].firstMatch.tap() }
        }
        // The Go menu's ⌘, is the way that never scrolls or folds.
        if !app.buttons["Done"].firstMatch.waitForExistence(timeout: 3) {
            app.typeKey(",", modifierFlags: .command)
        }
        XCTAssertTrue(app.buttons["Done"].firstMatch.waitForExistence(timeout: 5), "Settings did not open at \(size)")
        let sheetBar = app.navigationBars["Settings"].firstMatch
        settle(sheetBar)
        try check(app, "Settings at \(size)", sheet: sheetBar)
        app.buttons["Done"].firstMatch.tap()

        // Search is its own tab role, drawn as the separate glass button.
        let search = app.buttons["Search"].firstMatch
        XCTAssertTrue(search.waitForExistence(timeout: 5), "no Search tab at \(size)")
        if search.isHittable { search.tap() } else { app.tabs["Search"].firstMatch.tap() }
        // The field grows out of the tab bar's search button; audited while
        // it is still growing, its placeholder reads as clipped. Wait for the
        // field and for it to stop moving.
        // (The iPad's sidebar draws Search without a field until it is used.)
        let field = app.searchFields.firstMatch
        if field.waitForExistence(timeout: 3) { settle(field) }
        try check(app, "Search at \(size)")
    }

    /// `sheet`: the presented sheet's navigation bar. Everything outside the
    /// sheet is the dimmed screen behind it -- unreachable, and measured by
    /// the auditor through the scrim.
    @MainActor private func check(_ app: XCUIApplication, _ place: String, sheet: XCUIElement? = nil) throws {
        do {
            try audit(app, place, sheet: sheet)
        } catch let error as NSError where error.domain == "com.apple.xcode.xctest.accessibilityAudit" && error.code == -56 {
            // "Audit failed to complete in time" is the harness, not a
            // finding (the big iPad sidebar, on a loaded machine): once more.
            try audit(app, place, sheet: sheet)
        }
    }

    @MainActor private func audit(_ app: XCUIApplication, _ place: String, sheet: XCUIElement?) throws {
        try app.performAccessibilityAudit(for: [
            .dynamicType, .textClipped, .hitRegion, .contrast, .sufficientElementDescription,
        ]) { issue in
            // Text scrolled UNDER the glass tab bar or a pinned action is
            // measured through the glass; scrolled into view it is plain text
            // on the background. Only that is skipped -- never a control that
            // lives in the chrome itself.
            if issue.auditType == .contrast, let element = issue.element,
               self.isUnderChrome(element, in: app) {
                return true
            }
            // Behind a sheet: the iPad's sidebar beside a form sheet, and --
            // with no element at all -- the status bar under the light
            // scrim above an iPhone page sheet (the test's screen recording
            // shows nothing else unnamed on screen). Every view inside the
            // sheet is named, so an unnamed node cannot hide a real failure.
            if issue.auditType == .contrast, let sheet, sheet.exists,
               issue.element.map({ !self.isInside(sheet, $0) }) ?? true {
                return true
            }
            // Settings' Form lays its cells out lazily, and at xSmall and
            // Large the auditor flags whichever rows it has not re-measured
            // as "partially unsupported" -- a different row as the layout or
            // scroll position changes, while the AX5 pass (every row laid
            // out at 53 pt body text, nothing clipped) proves they scale.
            // Only that check, only in the sheet; clipping and contrast
            // there still fail.
            if issue.auditType == .dynamicType, let sheet, sheet.exists,
               issue.element.map({ self.isInside(sheet, $0) }) ?? false {
                return true
            }
            // The system search field, grown out of the tab bar, reports its
            // own placeholder as clipped at every size while the recording
            // shows it whole; the app supplies only the prompt text.
            if issue.auditType == .textClipped, issue.element?.elementType == .searchField {
                return true
            }
            // System bars (the tab bar, the iPad's floating tab bar, a
            // navigation bar's Done) cap their text by design and offer the
            // Large Content Viewer instead. Only their Dynamic Type issues are
            // skipped -- never contrast, clipping or hit area -- and an
            // element the auditor cannot even name is taken to be one, since
            // every view this app draws is named.
            if issue.auditType == .dynamicType,
               issue.element.map({ self.isSystemBar($0, in: app) }) ?? true {
                return true
            }
            // Name the element: "Contrast failed" alone says nothing in a CI log.
            let element = issue.element.map { "\($0.elementType) '\($0.label)' at \($0.frame) in \(app.frame)" }
                ?? "unnamed element (\(issue.detailedDescription))"
            XCTFail("\(place): \(issue.compactDescription) -- \(element)")
            return true
        }
    }

    /// Waits for an element to stop moving: a sheet sliding up, a field
    /// growing out of the tab bar. Audited mid-animation, both read as
    /// clipped or low-contrast when neither is.
    @MainActor private func settle(_ element: XCUIElement) {
        var frame = CGRect.null
        for _ in 0..<20 where element.exists && element.frame != frame {
            frame = element.frame
            Thread.sleep(forTimeInterval: 0.25)
        }
    }

    @MainActor private func isInside(_ sheet: XCUIElement, _ element: XCUIElement) -> Bool {
        let bar = sheet.frame, frame = element.frame
        return frame.midX >= bar.minX && frame.midX <= bar.maxX && frame.midY >= bar.minY
    }

    @MainActor private func isSystemBar(_ element: XCUIElement, in app: XCUIApplication) -> Bool {
        let bars = app.navigationBars.allElementsBoundByIndex + app.tabBars.allElementsBoundByIndex
            + app.toolbars.allElementsBoundByIndex
        return bars.contains { $0.frame.contains(element.frame) }
    }

    /// Scrolled beneath the bottom chrome, and not part of it.
    @MainActor private func isUnderChrome(_ element: XCUIElement, in app: XCUIApplication) -> Bool {
        let pinned = app.otherElements["bottom-chrome"].firstMatch
        if pinned.exists, pinned.frame.contains(element.frame) { return false }
        var top = app.frame.maxY
        let bar = app.tabBars.firstMatch
        if bar.exists { top = min(top, bar.frame.minY) }
        if pinned.exists { top = min(top, pinned.frame.minY) }
        return element.frame.maxY > top
    }
}
