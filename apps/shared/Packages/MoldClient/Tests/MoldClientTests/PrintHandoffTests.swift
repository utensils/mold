import Foundation
import Testing

@testable import MoldClient

/// A print continued on another device finds the same machine by the
/// server's run first -- the Mac names its own server 127.0.0.1 -- then by
/// address, and never guesses.
struct PrintHandoffTests {
    private let mac = MoldHost(name: "workstation", baseURL: URL(string: "http://10.0.0.4:7680")!, apiKey: nil)
    private let other = MoldHost(name: "laptop", baseURL: URL(string: "http://10.0.0.9:7680")!, apiKey: nil)

    @Test func theServersRunWinsOverAnAddressThisDeviceCannotUse() {
        let info = PrintHandoff.userInfo(filename: "a.png", address: URL(string: "http://127.0.0.1:7680")!,
                                         instanceId: "inst-9")
        let found = PrintHandoff.resolve(info, hosts: [other, mac]) { $0 == mac.id ? "inst-9" : "inst-1" }
        #expect(found == PrintID(host: mac.id, filename: "a.png"))
    }

    @Test func withoutARunTheAddressFindsIt() {
        let info = PrintHandoff.userInfo(filename: "b.mp4", address: URL(string: "http://10.0.0.9:7680")!, instanceId: nil)
        #expect(PrintHandoff.resolve(info, hosts: [mac, other]) { _ in nil } == PrintID(host: other.id, filename: "b.mp4"))
    }

    @Test func anUnknownMachineIsNotGuessed() {
        let info = PrintHandoff.userInfo(filename: "c.png", address: URL(string: "http://10.9.9.9:7680")!, instanceId: "x")
        #expect(PrintHandoff.resolve(info, hosts: [mac, other]) { _ in "y" } == nil)
        #expect(PrintHandoff.resolve([:], hosts: [mac]) { _ in nil } == nil)
    }
}
