import Foundation
import Testing

@testable import MoldClient

/// "4× NVIDIA L40S", not four lines: how every machine card on both apps says
/// what a machine renders with.
struct ServerStatusHardwareTests {
    private func status(gpus: String) throws -> ServerStatus {
        let json = #"{"version":"0.32.0","busy":false,"uptime_secs":1,"gpus":\#(gpus)}"#
        return try MoldJSON.decoder.decode(ServerStatus.self, from: Data(json.utf8))
    }

    @Test func oneCardIsItsName() throws {
        let s = try status(gpus: #"[{"ordinal":0,"name":"NVIDIA RTX 4090"}]"#)
        #expect(s.hardware == "NVIDIA RTX 4090")
    }

    @Test func identicalCardsAreCounted() throws {
        let card = #"{"ordinal":0,"name":"NVIDIA L40S"}"#
        let s = try status(gpus: "[\(card),\(card),\(card),\(card)]")
        #expect(s.hardware == "4× NVIDIA L40S")
    }

    @Test func mixedCardsAreANumber() throws {
        let s = try status(gpus: #"[{"ordinal":0,"name":"A"},{"ordinal":1,"name":"B"}]"#)
        #expect(s.hardware == "2 GPUs")
    }

    @Test func noCardsSaysNothing() throws {
        #expect(try status(gpus: "[]").hardware == nil)
        #expect(try status(gpus: "null").hardware == nil)
    }
}
