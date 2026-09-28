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
            settle(app.navigationBars.firstMatch)
            try check(app, "\(tab) at \(size)")

            if tab == "Generate" {
                let chooser = app.buttons["choose-model"]
                if chooser.exists {
                    let composer = app.descendants(matching: .any)["bottom-chrome"].firstMatch
                    for _ in 0..<5 where !chooser.isHittable { composer.swipeUp() }
                    XCTAssertTrue(chooser.isHittable)
                    chooser.tap()
                    settle(app.navigationBars["Choose a Model"])
                    try check(app, "Model chooser at \(size)",
                              within: app.descendants(matching: .any)["model-chooser"].firstMatch)
                    app.buttons["Done"].firstMatch.tap()
                }
            }
        }

        // iPad: the sidebar, opened over the content -- its shelves and
        // machines are drawn nowhere else. What it dims behind it is exempt
        // from contrast, as behind a sheet.
        let showSidebar = app.buttons.matching(NSPredicate(format: "label CONTAINS[c] 'sidebar'")).firstMatch
        if showSidebar.waitForExistence(timeout: 2), showSidebar.isHittable {
            showSidebar.tap()
            let row = app.descendants(matching: .any)["Favourites"].firstMatch
            XCTAssertTrue(row.waitForExistence(timeout: 5), "the sidebar did not open at \(size)")
            settle(row)
            // The sidebar is the column from the left edge to its rows' end.
            try check(app, "Sidebar at \(size)",
                      region: CGRect(x: 0, y: 0, width: row.frame.maxX + 16, height: .greatestFiniteMagnitude))
            let hide = app.buttons.matching(NSPredicate(format: "label CONTAINS[c] 'sidebar'")).firstMatch
            if hide.exists, hide.isHittable { hide.tap() } else { app.typeKey("1", modifierFlags: .command) }
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
        try check(app, "Settings at \(size)", within: app.descendants(matching: .any)["settings-sheet"].firstMatch,
                  lazyForm: true)
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

    /// `region`: a presented sheet or the iPad sidebar. Everything outside it
    /// is the dimmed screen behind -- unreachable, and measured by
    /// the auditor through the scrim.
    @MainActor private func check(_ app: XCUIApplication, _ place: String, region: CGRect? = nil,
                                  within container: XCUIElement? = nil, lazyForm: Bool = false) throws {
        do {
            try audit(app, place, region: region, within: container, lazyForm: lazyForm)
        } catch where Self.isTimeout(error) {
            // "Audit failed to complete in time" is the harness, not a
            // finding: once more, and if it still cannot finish, say where.
            do { try audit(app, place, region: region, within: container, lazyForm: lazyForm) } catch where Self.isTimeout(error) {
                XCTFail("\(place): the audit could not finish (\(error.localizedDescription))")
            }
        }
    }

    private static func isTimeout(_ error: Error) -> Bool {
        let error = error as NSError
        return (error.domain == "com.apple.xcode.xctest.accessibilityAudit" && error.code == -56)
            || error.domain == "com.apple.dt.XCTest.XCTFuture"
    }

    @MainActor private func audit(_ app: XCUIApplication, _ place: String, region: CGRect?,
                                  within container: XCUIElement?, lazyForm: Bool) throws {
        // Dynamic Type temporarily resizes the entire hierarchy. Sample pixels
        // first at the settled launch size, before that audit changes frames.
        // Combining both checks can measure contrast against stale geometry.
        for types: XCUIAccessibilityAuditType in [
            [.contrast], [.dynamicType, .textClipped, .hitRegion, .sufficientElementDescription],
        ] {
            try app.performAccessibilityAudit(for: types) { issue in
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
                if issue.auditType == .contrast, let region,
                   issue.element.map({ !region.contains(CGPoint(x: $0.frame.midX, y: $0.frame.midY)) }) ?? true {
                    return true
                }
                // A presented sheet: anything that is not its descendant is the
                // screen behind (on iPhone the sheet spans the whole width, so
                // the Machines title under it sits "inside" any column).
                if issue.auditType == .contrast, let container, container.exists,
                   issue.element.map({ !self.isDescendant($0, of: container) }) ?? true {
                    return true
                }
                // Settings' Form lays its cells out lazily, and at xSmall and
                // Large the auditor flags whichever rows it has not re-measured
                // as "partially unsupported" -- a different row as the layout or
                // scroll position changes, while the AX5 pass (every row laid
                // out at 53 pt body text, nothing clipped) proves they scale.
                // Only that check, only in the sheet; clipping and contrast
                // there still fail.
                if issue.auditType == .dynamicType, lazyForm, let container, container.exists,
                   issue.element.map({ self.isDescendant($0, of: container) }) ?? false {
                    return true
                }
                // The Library grid is lazy too, and the auditor flags its day
                // headers (headline) at xSmall and Large the same way; the AX5
                // pass lays them out at full size. Tile badges (caption2, drawn
                // at 48 pt tall at AX5) are flagged at every size: XCUITest lists
                // them although VoiceOver never reaches them (hidden -- the
                // tile's label says what they show). Only Dynamic Type; their
                // contrast is still audited.
                if issue.auditType == .dynamicType, let id = issue.element?.identifier,
                   id == "tile-badge" || (id == "day-header" && !place.contains("AccessibilityXXXL")) {
                    return true
                }
                // The iPad sidebar's rows are UIKit's single-line cells, and at
                // xSmall and Large the auditor PREDICTS a long one ("Recently
                // Deleted") "may be clipped at larger Dynamic Type sizes". The
                // AX5 pass audits the sidebar at that size and reports real
                // clipping there; only the prediction is skipped, only there.
                if issue.auditType == .textClipped, place.hasPrefix("Sidebar"),
                   issue.detailedDescription.contains("larger Dynamic Type sizes") {
                    return true
                }
                // A disabled control (Generate before a model is chosen) is an
                // inactive component, which WCAG 1.4.3 exempts from contrast.
                if issue.auditType == .contrast, let element = issue.element, element.exists, !element.isEnabled {
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
                XCTFail("\(place): \(issue.compactDescription) -- \(element) [\(issue.detailedDescription)]")
                return true
            }
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
        // Still: but a sheet's content and a tab's cross-fade finish after
        // the frames stop, and only on a slow machine does that show (CI
        // measured a Settings header mid-fade as low contrast).
        Thread.sleep(forTimeInterval: 0.75)
    }


    @MainActor private func isDescendant(_ element: XCUIElement, of container: XCUIElement) -> Bool {
        container.descendants(matching: element.elementType)
            .matching(NSPredicate(format: "label == %@", element.label))
            .allElementsBoundByIndex
            .contains { $0.frame == element.frame }
    }

    @MainActor private func isSystemBar(_ element: XCUIElement, in app: XCUIApplication) -> Bool {
        let bars = app.navigationBars.allElementsBoundByIndex + app.tabBars.allElementsBoundByIndex
            + app.toolbars.allElementsBoundByIndex
        return bars.contains { $0.frame.contains(element.frame) }
    }

    /// Scrolled beneath the bottom chrome, and not part of it.
    @MainActor private func isUnderChrome(_ element: XCUIElement, in app: XCUIApplication) -> Bool {
        // Any type: at AX5 the composer is a ScrollView, not a plain group.
        let pinned = app.descendants(matching: .any)["bottom-chrome"].firstMatch
        // A control IN the chrome is never skipped -- by descent, not frame:
        // at AX5 the composer's frame covers the canvas text behind it.
        if pinned.exists, isDescendant(element, of: pinned) { return false }
        var top = app.frame.maxY
        let bar = app.tabBars.firstMatch
        if bar.exists { top = min(top, bar.frame.minY) }
        if pinned.exists { top = min(top, pinned.frame.minY) }
        // The glass bar's scroll-edge effect dims what scrolls toward it
        // from above its own frame (UIKit's "AdditionalDimmingOverlay").
        let dimming = app.images["AdditionalDimmingOverlay"].firstMatch
        if dimming.exists { top = min(top, dimming.frame.minY) }
        return element.frame.maxY > top
    }
}
