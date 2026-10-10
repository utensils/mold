import Foundation

/// One serialized writer for the app icon across windows and refresh tasks.
/// A permission request can suspend; write the latest gallery count afterward.
final class AppIconBadge {
    private let write: (Int) async -> Void
    private let authorize: () async -> Void
    private var pending: Int?
    private var mayPrompt = false
    private var task: Task<Void, Never>?

    init(write: @escaping (Int) async -> Void, authorize: @escaping () async -> Void) {
        self.write = write
        self.authorize = authorize
    }

    func update(_ count: Int, allowPrompt: Bool) {
        pending = count
        mayPrompt = mayPrompt || (allowPrompt && count > 0)
        guard task == nil else { return }
        task = Task {
            while pending != nil {
                if mayPrompt {
                    mayPrompt = false
                    await authorize()
                }
                guard let count = pending else { continue }
                pending = nil
                await write(count)
            }
            task = nil
        }
    }

    func flush() async { await task?.value }
}
