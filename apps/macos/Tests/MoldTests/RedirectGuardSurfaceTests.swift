import Foundation
import Testing

@testable import Mold

/// `RedirectGuard.swift`'s rule: every request that carries `X-Api-Key`
/// attaches the guard, so a redirect off the origin never hands the key
/// away. The app's own senders are the engine's loopback probe and its
/// shutdown -- exactly where something squatting the port before the engine
/// binds it could answer with a redirect.
///
/// **Fails today**: both engine requests go out with no delegate.
@Test func everyFileThatSendsTheKeyAttachesTheRedirectGuard() throws {
    let root = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        .appending(path: "Sources/Mold")
    let files = FileManager.default.enumerator(at: root, includingPropertiesForKeys: nil)
    let offenders = (files?.allObjects as? [URL] ?? [])
        .filter { $0.pathExtension == "swift" }
        .compactMap { url -> String? in
            guard let text = try? String(contentsOf: url, encoding: .utf8),
                  text.contains("\"X-Api-Key\""), !text.contains("RedirectGuard(")
            else { return nil }
            return url.lastPathComponent
        }
    #expect(offenders.isEmpty, "\(offenders)")
}
