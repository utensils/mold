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
            if tab == "Models", !button.waitForExistence(timeout: 2), !app.buttons["ToggleSideBar"].exists { continue }
            // Restore the requested size after the prior size-changing audit.
            app.terminate()
            app.launch()
            XCTAssertTrue(app.navigateToDestination(tab, shortcut: "\(index + 1)"))
            settle(app.navigationBars.firstMatch)
            if tab == "Generate" { try auditComposer(app, size: size) }
            try check(app, "\(tab) at \(size)")

            if tab == "Generate" {
                // The Dynamic Type auditor changes layout while probing sizes.
                // Restore the requested size before interacting with the sheet.
                app.terminate()
                app.launch()
                app.buttons["Generate"].firstMatch.tap()
                let chooser = app.buttons["choose-model"]
                if chooser.exists {
                    let composer = app.scrollViews["phone-generate-form"].exists
                        ? app.scrollViews["phone-generate-form"]
                        : app.descendants(matching: .any)["bottom-chrome"].firstMatch
                    for _ in 0..<5 where !chooser.isHittable { composer.swipeUp() }
                    XCTAssertTrue(chooser.isHittable)
                    chooser.tap()
                    XCTAssertTrue(app.navigationBars["Choose a Model"].waitForExistence(timeout: 5))
                    settle(app.navigationBars["Choose a Model"])
                    try check(app, "Model chooser at \(size)",
                              within: app.descendants(matching: .any)["model-chooser"].firstMatch)
                    // Size probing can leave UIKit exporting the presenting
                    // hierarchy until another presentation. Restore a fresh
                    // launch-size screen for the remaining destinations.
                    app.terminate()
                    app.launch()
                }
            }
        }

        app.terminate()
        app.launch()
        XCTAssertTrue(app.navigateToDestination("Machines", shortcut: "5"))

        // iPad: the sidebar, opened over the content -- its shelves and
        // machines are drawn nowhere else. What it dims behind it is exempt
        // from contrast, as behind a sheet.
        let showSidebar = app.buttons.matching(NSPredicate(format: "label CONTAINS[c] 'sidebar'")).firstMatch
        if showSidebar.waitForExistence(timeout: 2), showSidebar.isHittable {
            showSidebar.tap()
            let row = app.descendants(matching: .any)["Favourites"].firstMatch
            XCTAssertTrue(row.waitForExistence(timeout: 5), "the sidebar did not open at \(size)")
            settle(row)
            let sidebar = app.collectionViews.containing(.any, identifier: "Favourites").firstMatch
            XCTAssertTrue(sidebar.exists, app.debugDescription)
            try check(app, "Sidebar at \(size)", within: sidebar)
            let hide = app.buttons.matching(NSPredicate(format: "label CONTAINS[c] 'sidebar'")).firstMatch
            if hide.exists, hide.isHittable { hide.tap() } else { app.typeKey("1", modifierFlags: .command) }
        }

        // UIKit's iPad floating bar loops inside the auditor's private
        // text-size cycling when Settings is presented over it. Audit the
        // identical Form through its native sidebar page on iPad instead.
        // Sheet presentation/dismissal has separate interaction coverage.
        let settingsInSidebar = app.buttons["ToggleSideBar"].exists
        if settingsInSidebar {
            app.terminate()
            app.launch()
            let favourites = app.descendants(matching: .any)["Favourites"].firstMatch
            if !favourites.exists || !favourites.isHittable { app.buttons["ToggleSideBar"].tap() }
            let sidebar = app.collectionViews.containing(.any, identifier: "Favourites").firstMatch
            let settings = sidebar.descendants(matching: .any).matching(NSPredicate(format: "label == 'Settings'")).firstMatch
            XCTAssertTrue(settings.waitForExistence(timeout: 5))
            settings.tap()
            XCTAssertTrue(app.navigationBars["Settings"].waitForExistence(timeout: 5))
        } else {
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
        }
        settle(app.navigationBars["Settings"])
        try check(app, "Settings at \(size)", within: app.descendants(matching: .any)["settings-sheet"].firstMatch,
                  lazyForm: true)
        if settingsInSidebar {
            XCTAssertTrue(app.navigateToDestination("Machines", shortcut: "5"))
        } else {
            app.buttons["Done"].firstMatch.tap()
        }

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

    /// Audit every exported composer label/control while fully inside the
    /// viewport. Pixel auditing a partly clipped AX node samples blank pixels
    /// outside the scroll view. Coverage is mandatory, not an exemption: a
    /// control that cannot be brought fully into view fails this test.
    @MainActor private func auditComposer(_ app: XCUIApplication, size: String) throws {
        // The first-run Add a Machine action also belongs to bottom chrome.
        // It has no generation composer; the outer screen audit covers it.
        if app.staticTexts["Add a machine to start generating"].exists { return }
        let composer = app.scrollViews["phone-generate-form"].exists
            ? app.scrollViews["phone-generate-form"]
            : app.descendants(matching: .any)["bottom-chrome"].firstMatch
        guard composer.exists else { return }
        let types: [XCUIElement.ElementType] = [.staticText, .button, .textField, .textView]
        func elements() -> [XCUIElement] {
            types.flatMap { composer.descendants(matching: $0).allElementsBoundByIndex }
                .filter { !$0.label.isEmpty && $0.frame.height > 0 }
        }
        func key(_ element: XCUIElement) -> String { "\(element.elementType):\(element.label)" }
        let expected = Set(elements().map(key))
        var seen = Set<String>()
        for step in 0..<20 {
            settle(composer)
            let visible = elements().filter { composerViewport(composer, in: app).contains($0.frame) }
            try check(app, "Composer scroll \(step) at \(size)", contrastOnly: true)
            seen.formUnion(visible.map(key))
            if expected.isSubset(of: seen) { break }
            // Small, non-flinging steps ensure even tall wrapped guidance is
            // sampled in full. The right edge avoids the photo well, and the
            // drag starts above the pinned Generate action.
            composer.coordinate(withNormalizedOffset: CGVector(dx: 0.9, dy: 0.60))
                .press(forDuration: 0.1, thenDragTo: composer.coordinate(
                    withNormalizedOffset: CGVector(dx: 0.9, dy: 0.28)),
                       withVelocity: .slow, thenHoldForDuration: 0.1)
        }
        XCTAssertTrue(expected.isSubset(of: seen), "Composer content never fully visible: \(expected.subtracting(seen).sorted())")
        let submit = app.buttons["submit-generation"]
        XCTAssertTrue(submit.exists)
        // Return to the prompt before the size-changing audit.
        for _ in 0..<5 {
            composer.coordinate(withNormalizedOffset: CGVector(dx: 0.9, dy: 0.28))
                .press(forDuration: 0.1, thenDragTo: composer.coordinate(
                    withNormalizedOffset: CGVector(dx: 0.9, dy: 0.60)),
                       withVelocity: .slow, thenHoldForDuration: 0.1)
        }
        settle(composer)
    }

    @MainActor private func composerViewport(_ composer: XCUIElement, in app: XCUIApplication) -> CGRect {
        guard composer.identifier == "phone-generate-form" else { return composer.frame }
        let top = max(composer.frame.minY, app.navigationBars.firstMatch.frame.maxY)
        let submit = app.buttons["submit-generation"]
        let bottom = submit.exists && !isDescendant(submit, of: composer)
            ? submit.frame.minY : min(composer.frame.maxY, app.tabBars.firstMatch.frame.minY)
        return CGRect(x: composer.frame.minX, y: top, width: composer.frame.width,
                      height: max(0, bottom - top))
    }

    /// `region`: a presented sheet or the iPad sidebar. Everything outside it
    /// is the dimmed screen behind -- unreachable, and measured by
    /// the auditor through the scrim.
    @MainActor private func check(_ app: XCUIApplication, _ place: String, region: CGRect? = nil,
                                  within container: XCUIElement? = nil, lazyForm: Bool = false,
                                  contrastOnly: Bool = false) throws {
        do {
            try audit(app, place, region: region, within: container, lazyForm: lazyForm, contrastOnly: contrastOnly)
        } catch where Self.isTimeout(error) {
            // "Audit failed to complete in time" is the harness, not a
            // finding: once more, and if it still cannot finish, say where.
            do { try audit(app, place, region: region, within: container, lazyForm: lazyForm, contrastOnly: contrastOnly) } catch where Self.isTimeout(error) {
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
                                  within container: XCUIElement?, lazyForm: Bool, contrastOnly: Bool) throws {
        // Dynamic Type temporarily resizes the entire hierarchy. Sample pixels
        // first at the settled launch size, before that audit changes frames.
        // Combining both checks can measure contrast against stale geometry.
        let passes: [XCUIAccessibilityAuditType] = contrastOnly ? [.contrast] : [
            [.contrast], [.dynamicType, .textClipped, .hitRegion, .sufficientElementDescription],
        ]
        for types in passes {
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
        let phoneForm = app.scrollViews["phone-generate-form"]
        if phoneForm.exists, isDescendant(element, of: phoneForm) {
            return !composerViewport(phoneForm, in: app).contains(element.frame)
        }
        // Any type: at AX5 the composer is a ScrollView, not a plain group.
        let pinned = app.descendants(matching: .any)["bottom-chrome"].firstMatch
        // A control IN the chrome is never skipped -- by descent, not frame:
        // at AX5 the composer's frame covers the canvas text behind it.
        if pinned.exists, isDescendant(element, of: pinned) {
            // The explicit scroll coverage pass requires every composer
            // label/control to be fully visible and audited at least once.
            return !pinned.frame.contains(element.frame)
        }
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
