import Foundation
import MoldClient

// The mesh arm's two helpers: where its bytes come from, and what its file
// actions mean here. Split from `RunCanvas+Result` for size.
extension RunCanvas {
    /// The stored GLB from the machine that rendered it, bounded by the
    /// reader's own cap. Throws, so the canvas puts the reason on screen.
    func meshBytes(_ filename: String) async throws -> Data {
        guard let host else { throw MeshViewFailure.transport("That machine isn't connected.") }
        let bytes = try await hosts.backend(for: host).media(filename, trashed: false)
        return try ResponseCeiling.checked(
            bytes, ceiling: min(ResponseCeiling.media, GLB.maximumBytes), what: "that mesh")
    }

    /// The mesh canvas's file actions, answered by the result bar's own.
    func perform(_ action: MeshViewAction, on result: BatchResult?) {
        guard let result else { return }
        switch action {
        case .save: actions.save(result)
        case .showInLibrary: actions.showInLibrary()
        default: break
        }
    }

}
