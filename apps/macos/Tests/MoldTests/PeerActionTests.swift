import Foundation
import MoldClient
import Testing

@testable import Mold

/// `PeerAction.resolve` is the pure decision behind `PeerSection`: what to do
/// about one discovered peer, with no `HostStore` in the loop. "Already have"
/// is asked twice -- by origin and by fleet identity -- because a machine
/// found again at a second address is not a new machine.
@MainActor
struct PeerActionTests {
    private let noneKnown: (URL) -> Bool = { _ in false }
    private let noInstanceKnown: (String) -> Bool = { _ in false }

    @Test func aPeerWeAlreadyTalkToAtAnotherAddressIsNotOffered() {
        let peer = FakeFixtures.discoveryPeer("workstation", url: "http://10.0.0.9:7680", instanceId: "abc-123")
        let action = PeerAction.resolve(peer, known: noneKnown, knownInstance: { $0 == "abc-123" })
        #expect(action == .skip)
    }

    @Test func aPeerAtAnOriginWeAlreadyHaveIsNotOffered() {
        let peer = FakeFixtures.discoveryPeer("workstation", url: "http://workstation:7680")
        let action = PeerAction.resolve(peer, known: { _ in true }, knownInstance: noInstanceKnown)
        #expect(action == .skip)
    }

    @Test func thisMachineIsNeverOfferedAsAPeer() {
        let peer = FakeFixtures.discoveryPeer("this-mac", url: "http://localhost:7680", isThisMachine: true)
        let action = PeerAction.resolve(peer, known: noneKnown, knownInstance: noInstanceKnown)
        #expect(action == .skip)
    }

    @Test func aPeerThatWantsAKeyOpensTheEditorRatherThanBeingAddedSilently() {
        let locked = FakeFixtures.discoveryPeer("workstation", url: "http://workstation:7680", authRequired: true)
        let open = FakeFixtures.discoveryPeer("bender", url: "http://bender:7680", authRequired: false)

        guard case let .edit(name, address) = PeerAction.resolve(locked, known: noneKnown, knownInstance: noInstanceKnown)
        else {
            Issue.record("expected .edit for a peer that requires a key")
            return
        }
        #expect(name == "workstation")
        #expect(address == "workstation:7680")

        guard case let .add(name2, url2) = PeerAction.resolve(open, known: noneKnown, knownInstance: noInstanceKnown)
        else {
            Issue.record("expected .add for a peer that needs no key")
            return
        }
        #expect(name2 == "bender")
        #expect(url2.absoluteString == "http://bender:7680")
    }

    @Test func aPeerWhoseAddressDoesNotNormalizeIsSkipped() {
        let peer = FakeFixtures.discoveryPeer("bad", url: "ftp://bad")
        let action = PeerAction.resolve(peer, known: noneKnown, knownInstance: noInstanceKnown)
        #expect(action == .skip)
    }
}
