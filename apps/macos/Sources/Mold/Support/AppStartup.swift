import Foundation
import Observation
import Synchronization

/// A single pre-store worker. A blocked volume must never occupy the main actor.
/// Cancellation cannot interrupt a filesystem syscall, but revokes every later step.
@MainActor @Observable
final class AppStartup<Value: Sendable> {
    private(set) var isStarting = false
    private(set) var failure: String?
    private var started = false
    private let cancellation = StartupCancellation()
    private let resolve: @Sendable () throws -> Value
    private let bootstrap: @Sendable (Value) throws -> Value
    private let ready: (Value) -> Void

    init(resolve: @escaping @Sendable () throws -> Value,
         bootstrap: @escaping @Sendable (Value) throws -> Value,
         ready: @escaping (Value) -> Void) {
        self.resolve = resolve
        self.bootstrap = bootstrap
        self.ready = ready
    }

    deinit { cancellation.cancel() }

    func start() {
        guard !started else { return }
        started = true
        isStarting = true
        let resolve = resolve, bootstrap = bootstrap, cancellation = cancellation
        DispatchQueue(label: "io.utensils.mold.startup", qos: .userInitiated).async { [weak self] in
            guard !cancellation.isCancelled else { return }
            let result: Result<Value, Error>
            do {
                let value = try resolve()
                guard !cancellation.isCancelled else { return }
                result = .success(try bootstrap(value))
            } catch { result = .failure(error) }
            Task { @MainActor [weak self] in
                guard let self, !cancellation.isCancelled else { return }
                isStarting = false
                switch result {
                case let .success(value): ready(value)
                case let .failure(error):
                    failure = (error as? EngineLaunchRefusal)?.reason
                        ?? "Mold couldn't prepare its engine. Quit and relaunch to try again."
                }
            }
        }
    }

    func cancel() {
        cancellation.cancel()
        started = true
        isStarting = false
    }
}

private nonisolated final class StartupCancellation: Sendable {
    private let value = Mutex(false)
    var isCancelled: Bool { value.withLock { $0 } }
    func cancel() { value.withLock { $0 = true } }
}

@MainActor @Observable
final class NativeAppStartup {
    private(set) var stores: AppStores?
    private(set) var resolvedHome: URL?
    private var cancelled = false
    let homeAccess = StartupHomeAccess()
    @ObservationIgnored lazy var preparation = AppStartup<EngineLaunch?>(
        resolve: { [weak self] in
            try EnginePreparation.resolve { [weak self] home in
                Task { @MainActor [weak self] in
                    guard let self, !cancelled else { return }
                    resolvedHome = home
                }
            }
        },
        bootstrap: { try EnginePreparation.bootstrap($0) },
        ready: { [weak self] launch in self?.stores = AppStores(preparedLaunch: launch) })

    func cancel() {
        cancelled = true
        preparation.cancel()
        homeAccess.cancel()
    }
}
