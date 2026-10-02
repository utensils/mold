import Foundation
import MoldClient
import MoldClientTesting
import Testing

@testable import MoldCompanion

/// The machine list: adding, refusing duplicates, the Default, removal taking
/// the key with it, reachability words, and pairing -- all against
/// `FakeBackend` and an in-memory credential store, never the network.
@MainActor
struct HostStoreTests {
    final class MemoryCredentials: CredentialStore, @unchecked Sendable {
        var keys: [UUID: String] = [:]
        func apiKey(for host: UUID) throws -> String? { keys[host] }
        func setAPIKey(_ key: String, for host: UUID) throws { keys[host] = key.isEmpty ? nil : key }
        func clearAPIKey(for host: UUID) throws { keys[host] = nil }
    }

    private func store(_ fake: FakeBackend = FakeBackend()) -> (HostStore, MemoryCredentials, HostListFile) {
        let file = HostListFile(url: FileManager.default.temporaryDirectory
            .appending(path: "hosts-\(UUID().uuidString).json"))
        let credentials = MemoryCredentials()
        return (HostStore(list: file, credentials: credentials, makeBackend: { _ in fake }), credentials, file)
    }

    private func status(instance: String = "inst-1") throws -> ServerStatus {
        let json = #"{"version":"0.32.0","busy":false,"uptime_secs":5,"instance_id":"\#(instance)","#
            + #""gpus":[{"ordinal":0,"name":"NVIDIA L40S"},{"ordinal":1,"name":"NVIDIA L40S"}]}"#
        return try MoldJSON.decoder.decode(ServerStatus.self, from: Data(json.utf8))
    }

    @Test func addingSavesTheListWithoutTheKeyAndTheKeyToTheKeychain() throws {
        let (hosts, credentials, file) = store()
        let host = try hosts.add(name: "workstation", address: "10.0.0.4", apiKey: "k1", makeDefault: false)
        #expect(host.baseURL.absoluteString == "http://10.0.0.4:7680")
        #expect(credentials.keys[host.id] == "k1")
        #expect(hosts.defaultMachine == host.id, "the first machine is the Default")
        let onDisk = try String(contentsOf: file.url, encoding: .utf8)
        #expect(!onDisk.contains("k1"), "a key must never reach hosts.json")
        #expect(file.load().entries.map(\.name) == ["workstation"])
    }

    @Test func theSameAddressTwiceIsRefusedByName() throws {
        let (hosts, _, _) = store()
        try hosts.add(name: "workstation", address: "10.0.0.4", apiKey: nil, makeDefault: false)
        #expect(throws: HostEditError.duplicate("workstation")) {
            try hosts.add(name: "again", address: "http://10.0.0.4:7680", apiKey: nil, makeDefault: false)
        }
    }

    @Test func removingTakesTheKeyAndTheDefaultWithIt() throws {
        let (hosts, credentials, _) = store()
        let host = try hosts.add(name: "a", address: "10.0.0.4", apiKey: "k", makeDefault: true)
        hosts.remove(host.id)
        #expect(hosts.hosts.isEmpty)
        #expect(credentials.keys[host.id] == nil)
        #expect(hosts.defaultMachine == nil)
    }

    @Test func aMachineThatAnswersIsUpAndCountsItsModels() async throws {
        let fake = FakeBackend()
        fake.stub("status()", returning: try status())
        fake.stub("capabilities()", throwing: MoldClientError.malformedResponse)
        fake.stub("models()", returning: [Model]())
        let (hosts, _, _) = store(fake)
        let host = try hosts.add(name: "a", address: "10.0.0.4", apiKey: nil, makeDefault: false)
        await hosts.refresh(host)
        #expect(hosts.isUp(host))
        #expect(hosts.reachability(of: host).summary == "Ready · 0.32.0")
        #expect(hosts.installed[host.id] == 0)
    }

    @Test(arguments: [false, true])
    func recheckingAnAnsweredMachineKeepsItsControlsUntilTheAnswer(fails: Bool) async throws {
        let fake = FakeBackend()
        let answer = try status()
        fake.stub("status()", returning: answer)
        fake.stub("models()", returning: [Model]())
        let (hosts, _, _) = store(fake)
        let host = try hosts.add(name: "a", address: "10.0.0.4", apiKey: nil, makeDefault: false)
        await hosts.refresh(host)
        hosts.stopWatching()
        fake.stub("status()") { _ in
            await MainActor.run {
                #expect(hosts.isUp(host), "A routine refresh must not remove source wells and their presented picker")
                #expect(hosts.instanceID(of: host.id) == answer.instanceId)
            }
            if fails { throw MoldClientError.unauthorized }
            return answer
        }
        await hosts.refresh(host)
        hosts.stopWatching()
        #expect(hosts.isUp(host) == !fails)
        if fails { #expect(hosts.reachability(of: host) == .needsKey) }
    }

    @Test func aRefusalReadsAsNeedsAKeyNotAsDown() async throws {
        let fake = FakeBackend()
        fake.stub("status()", throwing: MoldClientError.unauthorized)
        let (hosts, _, _) = store(fake)
        let host = try hosts.add(name: "a", address: "10.0.0.4", apiKey: nil, makeDefault: false)
        await hosts.refresh(host)
        #expect(hosts.reachability(of: host) == .needsKey)
        #expect(hosts.reachability(of: host).sentence == "This machine is there but wants an API key.")
    }

    @Test func pairingAddsTheMachineWithTheClaimedKey() async throws {
        let (hosts, credentials, _) = store()
        let payload = try MobilePairingPayload.parse(
            "mold://pair?version=1&base_url=http%3A%2F%2Fbox%3A7680&token=t&expires_at=4000000000&instance_id=inst-1&name=box")
        var sent: (String, String)?
        let host = try await hosts.pair(payload, claim: { _, name, kind in
            sent = (name, kind)
            return try MoldJSON.decoder.decode(PairingClaim.self, from: Data(
                #"{"api_key":"mold_pair_x","instance_id":"inst-1","hostname":"box"}"#.utf8))
        }, client: PairingClient(name: "Mold Studio on iPhone", kind: "iphone"))
        #expect(host.name == "box")
        #expect(credentials.keys[host.id] == "mold_pair_x")
        #expect(sent?.1 == "iphone")
    }

    @Test func pairingAMachineAlreadyListedReplacesItsKeyInsteadOfAddingARow() async throws {
        let (hosts, credentials, _) = store()
        let existing = try hosts.add(name: "workstation", address: "box", apiKey: "old", makeDefault: false)
        let payload = try MobilePairingPayload.parse(
            "mold://pair?version=1&base_url=http%3A%2F%2Fbox%3A7680&token=t&expires_at=4000000000&instance_id=inst-1&name=box")
        try await hosts.pair(payload, claim: { _, _, _ in
            try MoldJSON.decoder.decode(PairingClaim.self, from: Data(
                #"{"api_key":"new","instance_id":"inst-1","hostname":"box"}"#.utf8))
        })
        #expect(hosts.hosts.count == 1)
        #expect(hosts.hosts.first?.name == "workstation")
        #expect(credentials.keys[existing.id] == "new")
    }

    @Test func hardwareIsSaidTheWayAPersonWould() throws {
        #expect(try status().hardware == "2× NVIDIA L40S")
    }
}
