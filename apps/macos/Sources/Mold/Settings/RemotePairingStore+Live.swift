import Foundation
import MoldClient

extension RemotePairingStore.Dependencies {
    static var live: Self {
        let client = ManagedRelayClient()
        return Self(load: { try ManagedRelayEnrollment.load() },
                    loadEnabled: { AppStorageSuite.defaults.bool(forKey: "managedRemoteAccessEnabled") },
                    saveEnabled: { AppStorageSuite.defaults.set($0, forKey: "managedRemoteAccessEnabled") },
                    save: { try ManagedRelayEnrollment.save($0) },
                    enroll: { _ in
                        #if MOLD_EMBEDDED_ENGINE
                        var bytes = [UInt8](repeating: 0, count: 32)
                        guard mold_secure_host_namespace(&bytes, bytes.count) == 0 else { throw ManagedRelayFailure.engineNotRunning }
                        return try await client.discover(namespace: String(decoding: bytes, as: UTF8.self))
                        #else
                        throw ManagedRelayFailure.engineNotRunning
                        #endif
                    },
                    start: { owner, port in
                        #if MOLD_EMBEDDED_ENGINE
                        let code = owner.relayUrl.absoluteString.withCString { endpoint in
                                owner.hostId.withCString { host in
                                    owner.publicUrl.absoluteString.withCString { origin in
                                        mold_relay_start(endpoint, host, port, origin)
                                    }
                                }
                        }
                        guard code == 0 else { throw ManagedRelayFailure.connector }
                        #else
                        throw ManagedRelayFailure.engineNotRunning
                        #endif
                    }, stop: {
                        #if MOLD_EMBEDDED_ENGINE
                        _ = mold_relay_stop()
                        #endif
                    }, isAlive: {
                        #if MOLD_EMBEDDED_ENGINE
                        mold_relay_is_alive()
                        #else
                        false
                        #endif
                    }, advertise: { origin in
                        #if MOLD_EMBEDDED_ENGINE
                        let code = origin.map { value in value.absoluteString.withCString { mold_relay_set_public_origin($0) } }
                            ?? mold_relay_set_public_origin(nil)
                        guard code == 0 else { throw ManagedRelayFailure.connector }
                        #endif
                    }, verify: { session, origin in
                        guard let invitation = session.secureInvitation,
                              let namespace = origin.host?.split(separator: ".").first,
                              invitation.namespace == String(namespace) else { throw ManagedRelayFailure.notReady }
                        for _ in 0..<12 {
                            try Task.checkCancellation()
                            #if MOLD_EMBEDDED_ENGINE
                            if mold_relay_is_ready() { return }
                            #endif
                            try await Task.sleep(for: .seconds(1))
                        }
                        throw ManagedRelayFailure.notReady
                    }, localNetwork: { enabled in
                        #if MOLD_EMBEDDED_ENGINE
                        mold_local_network_set_enabled(enabled)
                        #endif
                    })
    }
}

extension RemotePairingStore {
    func followEngine(_ engine: MoldEngine) {
        guard supervisor == nil else { return }
        supervisor = Task { @MainActor [weak self, weak engine] in
            var lastRenewal = Date.distantPast
            while !Task.isCancelled {
                guard let self, let engine else { return }
                if self.enabled, let host = engine.host {
                    self.dependencies.localNetwork(true)
                    if self.pendingOperationNeeded(lastRenewal: lastRenewal) {
                        _ = try? await self.prepare(host, renew: true)
                        lastRenewal = .now
                    }
                } else if self.enabled, self.state != .off {
                    self.stopTransport()
                    self.state = .failed("This Mac’s engine is stopped. Remote access will reconnect when it starts.")
                }
                do { try await Task.sleep(for: .seconds(30)) } catch { return }
            }
        }
    }

    func pendingOperationNeeded(lastRenewal: Date) -> Bool {
        switch state {
        case .preparing: false
        case .ready: !dependencies.isAlive() || Date().timeIntervalSince(lastRenewal) > 86_400
        case .off, .failed: true
        }
    }
}
