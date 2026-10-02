import Foundation
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
        #expect(!RemoteAccessSettings.canPair(hostID: MoldEngine.localHostID))
        #expect(RemoteAccessSettings.canPair(hostID: UUID()))
    }
}
