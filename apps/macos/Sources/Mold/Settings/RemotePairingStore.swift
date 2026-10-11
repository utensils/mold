import Foundation
import MoldClient

/// Owns opt-in access independently of a Settings window or QR sheet.
@MainActor @Observable
final class RemotePairingStore {
    enum State: Equatable { case off, preparing, ready(URL), failed(String) }
    struct Dependencies {
        var load: () throws -> ManagedRelayEnrollment?
        var loadEnabled: () -> Bool
        var saveEnabled: (Bool) -> Void
        var save: (ManagedRelayEnrollment?) throws -> Void
        var enroll: (ManagedRelayEnrollment?) async throws -> ManagedRelayEnrollment
        var start: (ManagedRelayEnrollment, UInt16) throws -> Void
        var stop: () -> Void
        var isAlive: () -> Bool
        var advertise: (URL?) throws -> Void
        var verify: (PairingSession, URL) async throws -> Void
        var localNetwork: (Bool) -> Void = { _ in }
    }

    internal(set) var state: State = .off
    private(set) var enrollment: ManagedRelayEnrollment?
    private(set) var enabled = false
    private(set) var isStopping = false
    let pairing: PairingStore
    let dependencies: Dependencies
    @ObservationIgnored private var pending: Task<MoldHost, Error>?
    @ObservationIgnored var supervisor: Task<Void, Never>?
    @ObservationIgnored private var generation = 0

    init(pairing: PairingStore, dependencies: Dependencies = .live) {
        self.pairing = pairing
        self.dependencies = dependencies
        do { enrollment = try dependencies.load(); enabled = dependencies.loadEnabled() }
        catch { state = .failed("Mold couldn’t read its saved remote connection. Try pairing again.") }
    }

    func prepare(_ host: MoldHost, renew: Bool = false) async throws -> MoldHost {
        guard !isStopping else { throw ManagedRelayFailure.stopping }
        guard host.id == MoldEngine.localHostID, host.baseURL.host == "127.0.0.1",
              let port = host.baseURL.port, let localPort = UInt16(exactly: port) else {
            throw ManagedRelayFailure.engineNotRunning
        }
        if let pending { return try await pending.value }
        dependencies.localNetwork(true)
        let alreadyConnected = dependencies.isAlive() && enrollment != nil
        let previousOwner = enrollment
        generation += 1
        let current = generation
        state = .preparing
        let task = Task { @MainActor [self] () throws -> MoldHost in
            let owner: ManagedRelayEnrollment
            if alreadyConnected, !renew, let saved = enrollment {
                owner = saved
            } else {
                owner = try await dependencies.enroll(enrollment)
            }
            try Task.checkCancellation()
            try owner.validate()
            try dependencies.save(owner)
            dependencies.saveEnabled(true)
            enrollment = owner
            enabled = true
            // Opening a QR must not interrupt an already running tunnel.
            if !alreadyConnected || previousOwner?.hostId != owner.hostId
                || previousOwner?.relayUrl != owner.relayUrl {
                dependencies.stop()
                dependencies.localNetwork(enabled)
                try dependencies.start(owner, localPort)
            }
            try dependencies.advertise(owner.publicUrl)
            if PairingSheet.needsFreshCode(session: pairing.session, sessionHost: pairing.sessionHost,
                                          host: host.id, now: .now) {
                await pairing.createSession(on: host.id)
            }
            try Task.checkCancellation()
            guard let session = pairing.session, pairing.sessionHost == host.id else {
                throw ManagedRelayFailure.notReady
            }
            try await dependencies.verify(session, owner.publicUrl)
            try Task.checkCancellation()
            var phoneHost = host
            phoneHost.baseURL = owner.publicUrl
            return phoneHost
        }
        pending = task
        do {
            let phoneHost = try await task.value
            guard current == generation else { throw CancellationError() }
            state = .ready(phoneHost.baseURL)
            pending = nil
            return phoneHost
        } catch {
            if current == generation {
                pending = nil
                dependencies.stop()
                dependencies.localNetwork(enabled)
                try? dependencies.advertise(nil)
                state = .failed(error is CancellationError ? "Pairing was cancelled. Try again." : error.failureSentence)
            }
            throw error
        }
    }

    /// Withdraw transport before removing public connection metadata.
    func disable() async {
        guard !isStopping else { return }
        isStopping = true
        defer { isStopping = false }
        stopTransport()
        dependencies.saveEnabled(false)
        enabled = false
        state = .off
        do {
            try dependencies.save(nil)
            enrollment = nil
        } catch {
            state = .failed("Remote access is stopped. Its saved connection couldn’t be removed; try Stop Remote Access again.")
        }
    }

    /// Quitting keeps the enrollment so paired phones work after a relaunch.
    func shutdown() {
        supervisor?.cancel()
        supervisor = nil
        stopTransport()
    }

    func stopTransport() {
        generation += 1
        pending?.cancel()
        pending = nil
        dependencies.stop()
        dependencies.localNetwork(false)
        try? dependencies.advertise(nil)
    }

    var canStopRemoteAccess: Bool { enabled || enrollment != nil }
}
