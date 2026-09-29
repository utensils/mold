import XCTest

extension XCTestCase {
    /// Fresh simulators can ask about Local Network or Notifications during
    /// an otherwise unrelated tap. Accept the app's own permissions so the
    /// interaction under test remains the one XCTest is exercising.
    func acceptCompanionPermissions() {
        addUIInterruptionMonitor(withDescription: "Mold Studio permissions") { alert in
            let allow = alert.buttons["Allow"]
            guard allow.exists else { return false }
            allow.tap()
            return true
        }
    }
}
