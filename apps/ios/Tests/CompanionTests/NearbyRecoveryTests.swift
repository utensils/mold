import Foundation
import MoldClient
import MoldClientTesting
import Network
import Testing
@testable import MoldCompanion

@MainActor
struct NearbyRecoveryTests {
    private let key = "mold_pair_abcdefghijklmnopqrstuvwxyz0123456789ABCDEFG"
    private var found: NearbyBrowser.Machine {
        .init(name: "halcyon-7680", instanceID: "halcyon-id",
              endpoint: .hostPort(host: "192.0.2.1", port: 7680))
    }
    private func store() -> (HostStore, FakeBackend, MoldHost) {
        let fake = FakeBackend()
        let file = HostListFile(url: FileManager.default.temporaryDirectory.appending(path: "nearby-\(UUID()).json"))
        let hosts = HostStore(list: file, credentials: HostStoreTests.MemoryCredentials(), makeBackend: { _ in fake })
        let host = MoldHost(name: "halcyon", baseURL: URL(string: "https://relay.example")!, apiKey: key,
                            connectionEndpoints: [ConnectionEndpoint(url: "https://relay.example", kind: .relay)],
                            connectionInstanceID: "halcyon-id", secure: SecureHost(
                                binding: .init(signingKey: "pin", noiseKey: "noise", signature: "signature"), namespace: "namespace",
                                websocketURL: URL(string: "wss://relay.example")!,
                                grant: .init(id: "grant", expiresAt: 4_000_000_000, epoch: 0, signature: "signature")))
        hosts.setHosts([host])
        return (hosts, fake, host)
    }

    @Test func offlinePairedMachineIsNotOfferedAsADuplicateNearby() {
        let (hosts, _, host) = store()
        hosts.setReachability(.down("Unavailable"), for: host.id)
        #expect(hosts.knows(found))
    }

    @Test func nearbyResolutionEndsWhenCancelled() async {
        let browser = NearbyBrowser()
        let task = Task { try await browser.address(of: found) }
        task.cancel()
        do {
            _ = try await task.value
            Issue.record("Cancelled Nearby lookup must not succeed")
        } catch { #expect(error is CancellationError) }
    }

    @Test func staleBonjourLookupHasAFiniteDeadline() async {
        let start = ContinuousClock.now
        do {
            _ = try await NearbyBrowser().address(of: found, timeout: .milliseconds(20))
            Issue.record("The unreachable documentation address must not resolve")
        } catch {
            #expect(start.duration(to: .now) < .seconds(1))
        }
    }

    @Test func completedProofCannotRestoreARemovedMachine() async {
        let (hosts, fake, host) = store()
        var resolved = host
        resolved.baseURL = URL(string: "http://192.168.1.142:7680")!
        let answer = resolved
        fake.stub("resolvedConnection()") { _ in
            await MainActor.run { hosts.remove(host.id) }
            return Optional(answer)
        }
        await hosts.recoverNearby(found, address: resolved.baseURL.absoluteString)
        #expect(hosts.hosts.isEmpty)
        #expect(!fake.calls.contains { $0.route == "status()" })
    }

    @Test func discoveredRouteRequiresProofBeforeSavingOrAuthenticatedReads() async {
        let (hosts, fake, host) = store()
        fake.stub("resolvedConnection()", returning: Optional<MoldHost>.none)
        await hosts.recoverNearby(found, address: "http://192.168.1.142:7680")
        #expect(hosts.host(host.id) == host)
        #expect(fake.count("resolvedConnection()") == 1)
        #expect(!fake.calls.contains { $0.route == "status()" })
    }

    @Test func matchingDiscoveryRecoversSamePairedMachineWithoutReplacingItsKey() async throws {
        let (hosts, fake, host) = store()
        var resolved = host
        resolved.baseURL = URL(string: "http://192.168.1.142:7680")!
        let answer = resolved
        fake.stub("resolvedConnection()") { _ in
            fake.stub("resolvedConnection()", returning: Optional<MoldHost>.none)
            return Optional(answer)
        }
        fake.stub("status()", returning: try MoldJSON.decoder.decode(ServerStatus.self,
            from: Data(#"{"version":"0.32.0","busy":false,"uptime_secs":1,"instance_id":"halcyon-id"}"#.utf8)))
        await hosts.recoverNearby(found, address: resolved.baseURL.absoluteString)
        #expect(hosts.hosts.count == 1)
        #expect(hosts.host(host.id)?.apiKey == key)
        #expect(hosts.host(host.id)?.baseURL == resolved.baseURL)
        #expect(hosts.host(host.id)?.connectionOriginalURL == host.baseURL)
        #expect(hosts.host(host.id)?.connectionEndpoints?.contains { $0.kind == .lan } == true)
        #expect(hosts.isUp(resolved))
    }

    @Test func operatorKeysNeverFollowBonjourAdvertisements() async {
        let (hosts, fake, original) = store()
        var host = original
        host.apiKey = "operator-secret"
        host.secure = nil
        hosts.setHosts([host])
        await hosts.recoverNearby(found, address: "http://192.168.1.142:7680")
        #expect(fake.calls.isEmpty)
        #expect(hosts.host(host.id) == host)
    }
}
