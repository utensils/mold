import Foundation

/// Where preferences live.
///
/// `MOLD_NATIVE_FRESH` swaps in a scratch suite so a first-launch run can be
/// exercised without throwing away the real machine list — the difference
/// between testing onboarding and losing your setup.
enum AppStorageSuite {
    static let name = "io.utensils.mold.native.fresh"

    static var defaults: UserDefaults {
        guard ProcessInfo.processInfo.environment["MOLD_NATIVE_FRESH"] != nil,
              let scratch = UserDefaults(suiteName: name)
        else { return .standard }
        return scratch
    }

    static var isFresh: Bool {
        ProcessInfo.processInfo.environment["MOLD_NATIVE_FRESH"] != nil
    }
}
