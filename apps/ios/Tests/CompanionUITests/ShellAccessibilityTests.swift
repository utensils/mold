import XCTest

/// DESIGN.md §6 made executable: every destination is audited at the smallest
/// and the largest text sizes for clipped text, Dynamic Type support, hit
/// regions, contrast and element descriptions. A layout that only works at the
/// default size fails here, not in someone's hand.
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

        for tab in ["Generate", "Library", "Queue", "Machines"] {
            let button = app.buttons[tab].firstMatch
            XCTAssertTrue(button.waitForExistence(timeout: 5), "no \(tab) tab at \(size)")
            button.tap()
            try app.performAccessibilityAudit(for: [
                .dynamicType, .textClipped, .hitRegion, .contrast, .sufficientElementDescription,
            ]) { issue in
                // Text scrolled under the glass tab bar or a pinned action is
                // measured THROUGH the glass; scrolled into view it is plain
                // text on the background. Only that case is skipped.
                if issue.auditType == .contrast, let element = issue.element,
                   element.frame.maxY > self.bottomChromeTop(app) {
                    return true
                }
                // Name the element: "Contrast failed" alone says nothing in a CI log.
                let element = issue.element.map { "\($0.elementType) '\($0.label)'" } ?? "unknown element"
                XCTFail("\(tab) at \(size): \(issue.compactDescription) -- \(element)")
                return true
            }
        }
    }

    /// Where the bottom chrome begins: the tab bar, or an empty state's pinned
    /// action bar (`bottom-chrome`) above it.
    @MainActor private func bottomChromeTop(_ app: XCUIApplication) -> CGFloat {
        var top = app.frame.maxY
        let bar = app.tabBars.firstMatch
        if bar.exists { top = min(top, bar.frame.minY) }
        let pinned = app.otherElements["bottom-chrome"].firstMatch
        if pinned.exists { top = min(top, pinned.frame.minY) }
        return top
    }
}
