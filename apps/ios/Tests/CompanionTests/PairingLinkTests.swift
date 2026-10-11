import Foundation
import MoldClient
import Testing

@testable import MoldCompanion

/// A pairing code opened from outside -- the Camera app reading the Mac's QR,
/// or a tapped link -- is shown for confirmation on Machines, never claimed
/// on its own: a link anyone can send must not pair the phone by itself.
@MainActor
struct PairingLinkTests {
    var link: URL { try! SecurePairingFixture.payload().url! }

    @Test func aUniversalLinkAsksBeforePairing() {
        let router = AppRouter()
        router.open(url: link)
        #expect(router.selection == .go(.machines))
        #expect(router.pairingLink?.payload?.name == "Studio Mac")
        #expect(router.pairingLink?.payload?.baseURL == "http://studio.local:7680")
    }

    /// UIKit presents nothing over a sheet that is already up: a link that
    /// arrives with Add a Machine (or Settings, or a print) open closes it.
    @Test func aLinkClosesWhateverSheetIsUp() {
        let router = AppRouter()
        router.addMachine()
        router.showsSettings = true
        router.openedPrint = AppRouter.OpenedPrint(id: PrintID(host: UUID(), filename: "a.png"))
        router.open(url: link)
        #expect(!router.showsAddMachine)
        #expect(!router.showsSettings)
        #expect(router.openedPrint == nil)
        #expect(router.pairingLink != nil)
    }

    @Test func aBrokenCodeSaysWhy() {
        let router = AppRouter()
        router.open(url: URL(string: "https://utensils.io/mold/pair#version=9")!)
        #expect(router.pairingLink?.payload == nil)
        #expect(router.pairingLink?.failure == MobilePairingPayload.ParseError.unsupported.errorDescription)
    }

    @Test func otherLinksGoWhereTheyGo() {
        let router = AppRouter()
        router.open(url: DeepLink.queue(job: nil).url)
        #expect(router.selection == .go(.queue))
        router.open(url: URL(string: "https://utensils.io/mold/guide/companion")!)
        #expect(router.pairingLink == nil)
        #expect(router.selection == .go(.queue))
    }
}
