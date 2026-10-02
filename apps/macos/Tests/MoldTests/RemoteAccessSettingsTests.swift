import Foundation
import MoldClient
import Testing

@testable import Mold

struct RemoteAccessSettingsTests {
    @Test func remoteAccessHasItsOwnSettingsDestination() {
        let tab = SettingsTab(rawValue: "remoteAccess")
        #expect(tab?.title == "Remote Access")
        #expect(SettingsUAT.initialTab(environment: [SettingsUAT.envVar: "remoteAccess"]) == tab)
        let tabs = SettingsTab.allCases
        #expect(tabs.firstIndex(of: tab!) == tabs.firstIndex(of: .machines)! + 1)
    }

    @Test func localEngineCannotOfferAnUnreachablePairingCode() {
        #expect(!RemoteAccessSettings.canPair(MoldHost(id: MoldEngine.localHostID, name: "This Mac", baseURL: URL(string: "http://127.0.0.1:7680")!)))
        #expect(!RemoteAccessSettings.canPair(MoldHost(name: "Local", baseURL: URL(string: "http://localhost:7680")!)))
        #expect(RemoteAccessSettings.canPair(MoldHost(name: "Relay", baseURL: URL(string: "https://relay.example")!)))
    }
}

@MainActor
struct NativeConnectionPersistenceTests {
    @Test func routeChangesPreserveTheMachineAndItsCredential() {
        let original = MoldHost(name: "HAL9000", baseURL: URL(string: "http://192.168.1.2:7680")!, apiKey: "test-key")
        let hosts = HostStore(hosts: [original])
        var routed = original
        routed.baseURL = URL(string: "https://relay.example")!
        routed.connectionInstanceID = "hal-identity"
        routed.connectionEndpoints = [ConnectionEndpoint(url: routed.baseURL.absoluteString, kind: .relay)]
        hosts.applyConnection(routed, expectedURL: original.baseURL)
        #expect(hosts.hosts.count == 1)
        #expect(hosts.host(original.id)?.id == original.id)
        #expect(hosts.host(original.id)?.apiKey == "test-key")
        #expect(hosts.host(original.id)?.baseURL == routed.baseURL)
        #expect(hosts.host(original.id)?.connectionInstanceID == "hal-identity")
    }

    @Test func staleRouteRefreshCannotOverwriteAnEditedAddress() {
        let original = MoldHost(name: "HAL9000", baseURL: URL(string: "http://192.168.1.2:7680")!, apiKey: "test-key")
        var edited = original
        edited.baseURL = URL(string: "https://manually-edited.example")!
        let hosts = HostStore(hosts: [edited])
        var stale = original
        stale.baseURL = URL(string: "https://old-relay.example")!
        hosts.applyConnection(stale, expectedURL: original.baseURL)
        #expect(hosts.host(original.id)?.baseURL == edited.baseURL)
    }
}
