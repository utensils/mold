import Foundation
import Synchronization
import Testing
@testable import Mold

@MainActor
struct AppStartupTests {
    @Test func blockedResolutionLeavesMainActorUsableAndCreatesNothing() async throws {
        let resolver = StartupTestResolver()
        var stores = 0
        let startup = AppStartup<Int>(resolve: { resolver.resolve() }, bootstrap: { value in
            resolver.bootstrapped.withLock { $0 += 1 }; return value
        }, ready: { _ in stores += 1 })
        startup.start()
        await resolver.waitForEntry()
        startup.start()
        #expect(stores == 0)
        #expect(resolver.bootstrapped.withLock { $0 } == 0)
        #expect(startup.isStarting)
        resolver.release.signal()
        for _ in 0..<100 where startup.isStarting { try await Task.sleep(for: .milliseconds(10)) }
        #expect(stores == 1)
        #expect(resolver.bootstrapped.withLock { $0 } == 1)
    }

    @Test func cancelledBlockedResolutionCannotBootstrapOrPublishLateStores() async throws {
        let resolver = StartupTestResolver()
        var stores = 0
        let startup = AppStartup<Int>(resolve: { resolver.resolve() }, bootstrap: { value in
            resolver.bootstrapped.withLock { $0 += 1 }; return value
        }, ready: { _ in stores += 1 })
        startup.start()
        await resolver.waitForEntry()
        startup.cancel()
        resolver.release.signal()
        try await Task.sleep(for: .milliseconds(100))
        #expect(stores == 0)
        #expect(resolver.bootstrapped.withLock { $0 } == 0)
        #expect(!startup.isStarting)
        startup.start()
        #expect(!startup.isStarting)
    }

    @Test func resolutionFailureCreatesNoStoresOrBootstrap() async throws {
        var stores = 0
        let startup = AppStartup<Int>(resolve: { throw EngineLaunchRefusal.home("Drive unavailable") },
                                     bootstrap: { _ in Issue.record("must not bootstrap"); return 1 },
                                     ready: { _ in stores += 1 })
        startup.start()
        for _ in 0..<100 where startup.isStarting { try await Task.sleep(for: .milliseconds(10)) }
        #expect(startup.failure == "Drive unavailable")
        #expect(stores == 0)
    }

    @Test func cancelledBootstrapCannotPublishLateStores() async throws {
        let bootstrap = StartupTestResolver()
        var stores = 0
        let startup = AppStartup<Int>(resolve: { 42 }, bootstrap: { _ in bootstrap.resolve() },
                                     ready: { _ in stores += 1 })
        startup.start()
        await bootstrap.waitForEntry()
        startup.cancel()
        bootstrap.release.signal()
        try await Task.sleep(for: .milliseconds(100))
        #expect(stores == 0)
        #expect(startup.failure == nil)
    }

    @Test func failedBootstrapCannotConstructStoresAndDoesNotExposePrivateErrors() async throws {
        struct PrivateError: Error {}
        var stores = 0
        let startup = AppStartup<Int>(resolve: { 42 }, bootstrap: { _ in throw PrivateError() },
                                     ready: { _ in stores += 1 })
        startup.start()
        for _ in 0..<100 where startup.isStarting { try await Task.sleep(for: .milliseconds(10)) }
        #expect(startup.failure == "Mold couldn't prepare its engine. Quit and relaunch to try again.")
        #expect(stores == 0)
    }

}

private nonisolated final class StartupTestResolver: Sendable {
    let entered = Mutex(false)
    let bootstrapped = Mutex(0)
    let release = DispatchSemaphore(value: 0)
    func resolve() -> Int {
        entered.withLock { $0 = true }
        release.wait()
        return 42
    }
    func waitForEntry() async {
        for _ in 0..<200 {
            if entered.withLock({ $0 }) { return }
            try? await Task.sleep(for: .milliseconds(5))
        }
        Issue.record("Startup worker did not enter within one second")
    }
}
