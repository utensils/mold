import Foundation
import Testing

@testable import MoldClient

// The machine list has to survive a relaunch. It did not: the list was written
// with `MoldJSON.encoder` and read with `MoldJSON.decoder`, and those two are
// NOT inverses -- so every launch came up with no machines at all.

@Test func aSavedMachineComesBackAfterARelaunch() throws {
    let host = MoldHost(name: "plato", baseURL: URL(string: "http://plato:7680")!,
                        apiKey: "secret")
    let data = try MoldJSON.localEncoder.encode([StoredHost(host)])
    let read = try MoldJSON.localDecoder.decode([StoredHost].self, from: data)
    #expect(read.count == 1)
    #expect(read.first?.id == host.id)
    #expect(read.first?.name == "plato")
    #expect(read.first?.baseURL == host.baseURL)
}

/// The shape already written to preferences by every build so far. Reading it
/// is the difference between the fix restoring someone's machines and the fix
/// quietly losing them one last time.
@Test func readsTheShapeAlreadyOnDisk() throws {
    let json = Data("""
    [{"name":"plato","id":"8A4F7E42-2D3F-4749-8BBA-413A39902C2F",\
    "base_url":"http://plato:7680"}]
    """.utf8)
    let read = try MoldJSON.localDecoder.decode([StoredHost].self, from: json)
    #expect(read.first?.name == "plato")
    #expect(read.first?.baseURL.absoluteString == "http://plato:7680")
}

@Test func writesTheShapeAlreadyOnDisk() throws {
    let host = MoldHost(name: "plato", baseURL: URL(string: "http://plato:7680")!)
    let text = String(decoding: try MoldJSON.localEncoder.encode(StoredHost(host)), as: UTF8.self)
    #expect(text.contains("\"base_url\""))
}

/// The key is a credential and goes to `SecretStore`. Preferences are copied
/// into backups and into any sync that takes the domain.
@Test func aSavedMachineNeverCarriesItsKey() throws {
    let host = MoldHost(name: "plato", baseURL: URL(string: "http://plato:7680")!,
                        apiKey: "secret")
    let text = String(decoding: try MoldJSON.localEncoder.encode(StoredHost(host)), as: UTF8.self)
    #expect(!text.contains("secret"))
    #expect(StoredHost(host).host(apiKey: "secret").apiKey == "secret")
    #expect(StoredHost(host).host(apiKey: nil).apiKey == nil)
}

/// Why `StoredHost` spells its keys out and why the local coders exist at all.
/// Foundation's `convertToSnakeCase` knows `baseURL` is an acronym and writes
/// `base_url`; `convertFromSnakeCase` reads `base_url` back as `baseUrl`, which
/// is not the name of the property. The wire is unaffected -- a server sends
/// `base_url` and nothing round-trips through both strategies -- but anything
/// this app writes and reads itself must not go near them.
@Test func theWireCodersAreNotInversesOfEachOther() throws {
    struct Naive: Codable { var baseURL: URL }
    let data = try MoldJSON.encoder.encode(Naive(baseURL: URL(string: "http://plato:7680")!))
    #expect(String(decoding: data, as: UTF8.self).contains("base_url"))
    #expect(throws: DecodingError.self) {
        try MoldJSON.decoder.decode(Naive.self, from: data)
    }
}
