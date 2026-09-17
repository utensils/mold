import Foundation
import Testing

@testable import MoldClient

/// A bootstrap pointer that exists but is damaged.
///
/// **Fails today**: `MoldHome.resolve` returns `~/.mold` for an empty or
/// relative pointer and nothing else asks, so the embedded engine forced that
/// answer into `MOLD_HOME` and mold's own fail-closed guard could never fire
/// (review 05-M5).
struct MoldHomePointerTests {
    private func pointer(_ contents: String?) -> [String: String] {
        let url = FileManager.default.temporaryDirectory
            .appending(path: "mold-home-pointer-\(UUID().uuidString)")
        if let contents {
            try? contents.write(to: url, atomically: true, encoding: .utf8)
        }
        return ["MOLD_HOME_POINTER_PATH": url.path(percentEncoded: false)]
    }

    @Test func noPointerAtAllIsAFirstRunAndNotARefusal() {
        #expect(MoldHome.pointerRefusal(environment: pointer(nil)) == nil)
    }

    @Test func aGoodPointerIsNotARefusal() {
        #expect(MoldHome.pointerRefusal(environment: pointer("/Volumes/Big/mold\n")) == nil)
    }

    @Test(arguments: ["", "   \n", "relative/mold", "~/mold"])
    func aDamagedPointerRefusesRatherThanFallingBackTo_mold(_ contents: String) {
        let environment = pointer(contents)
        #expect(MoldHome.pointerRefusal(environment: environment) != nil)
        // `resolve` still answers -- which is exactly why the refusal has to
        // be asked for separately.
        #expect(MoldHome.resolve(environment: environment).source == .fallback)
    }

    @Test func anExplicitHomeNeverConsultsThePointer() {
        var environment = pointer("")
        environment["MOLD_HOME"] = "/tmp/somewhere"
        #expect(MoldHome.pointerRefusal(environment: environment) == nil)
    }
}
