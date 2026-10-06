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
        var revoke: (ManagedRelayEnrollment) async throws -> Void
        var start: (ManagedRelayEnrollment, UInt16) throws -> Void
        var stop: () -> Void
        var isAlive: () -> Bool
        var advertise: (URL?) throws -> Void
        var verify: (PairingSession, URL) async throws -> Void
    }

    internal(set) var state: State = .off
    private(set) var enrollment: ManagedRelayEnrollment?
    private(set) var enabled = false
    let pairing: PairingStore
    let dependencies: Dependencies
    @ObservationIgnored private var pending: Task<MoldHost, Error>?
    @ObservationIgnored var supervisor: Task<Void, Never>?
    @ObservationIgnored private var generation = 0

    init(pairing: PairingStore, dependencies: Dependencies = .live) {
        self.pairing = pairing
        self.dependencies = dependencies
        do { enrollment = try dependencies.load(); enabled = dependencies.loadEnabled() && enrollment != nil }
        catch { state = .failed("Mold couldn’t read its saved remote connection. Try pairing again.") }
    }

    func prepare(_ host: MoldHost, renew: Bool = false) async throws -> MoldHost {
        guard host.id == MoldEngine.localHostID, host.baseURL.host == "127.0.0.1",
              let port = host.baseURL.port, let localPort = UInt16(exactly: port) else {
            throw ManagedRelayFailure.engineNotRunning
        }
        if let pending { return try await pending.value }
        let alreadyConnected = dependencies.isAlive() && enrollment != nil
        let previousOwner = enrollment
        generation += 1
        let current = generation
        state = .preparing
        let task = Task { @MainActor [self] () throws -> MoldHost in
            let owner: ManagedRelayEnrollment
            if alreadyConnected, !renew, let saved = enrollment,
               TimeInterval(saved.expiresAt) > Date().timeIntervalSince1970 + 86_400 {
                owner = saved
            } else {
                owner = try await dependencies.enroll(enrollment)
            }
            if Task.isCancelled {
                // A response can arrive after Stop/quit. Release only a newly
                // created namespace; a saved owner must survive normal quitting.
                if previousOwner?.hostId != owner.hostId { await releaseUnretained(owner) }
                throw CancellationError()
            }
            try owner.validate()
            do { try dependencies.save(owner) }
            catch {
                if previousOwner?.hostId != owner.hostId { await releaseUnretained(owner) }
                throw error
            }
            dependencies.saveEnabled(true)
            enrollment = owner
            enabled = true
            // Opening a QR must not interrupt an already running tunnel.
            if !alreadyConnected || previousOwner?.hostId != owner.hostId
                || previousOwner?.relayUrl != owner.relayUrl || previousOwner?.token != owner.token {
                dependencies.stop()
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
                try? dependencies.advertise(nil)
                state = .failed(error is CancellationError ? "Pairing was cancelled. Try again." : error.failureSentence)
            }
            throw error
        }
    }

    /// Withdraw transport first. A failed control request cannot leave access on.
    func disable() async {
        stopTransport()
        dependencies.saveEnabled(false)
        enabled = false
        state = .off
        let owner = enrollment
        var cleared = true
        do { try dependencies.save(nil) } catch { cleared = false }
        var revoked = true
        if let owner {
            do { try await dependencies.revoke(owner) } catch { revoked = false }
        }
        // Keep ownership for another cleanup attempt; the independent opt-out
        // preference prevents a stale credential from enabling access on launch.
        if cleared && revoked { enrollment = nil }
        else { state = .failed("Remote access is stopped. Its saved connection couldn’t be fully removed; try Stop Remote Access again when online.") }
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
        try? dependencies.advertise(nil)
    }

    private func releaseUnretained(_ owner: ManagedRelayEnrollment) async {
        // An unstructured task does not inherit cancellation. Foundation must
        // be allowed to send the bounded cleanup request after Stop/quit.
        let cleanup = Task { @MainActor [dependencies] in try? await dependencies.revoke(owner) }
        await cleanup.value
    }

    var canStopRemoteAccess: Bool { enabled || enrollment != nil }
}
