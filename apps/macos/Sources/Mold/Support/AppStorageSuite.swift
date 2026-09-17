import Foundation

/// Where preferences live.
///
/// `MOLD_NATIVE_FRESH` swaps in a scratch suite so a first-launch run can be
/// exercised without throwing away the real machine list — the difference
/// between testing onboarding and losing your setup. `make uat` sets it,
/// alongside a throwaway `MOLD_HOME`.
enum AppStorageSuite {
    static let name = "io.utensils.mold.native.fresh"

    static var defaults: UserDefaults {
        guard NativeUAT.fresh.isSet(),
              let scratch = UserDefaults(suiteName: name)
        else { return .standard }
        return scratch
    }
}
