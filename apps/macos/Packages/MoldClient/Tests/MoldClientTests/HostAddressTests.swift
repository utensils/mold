import Foundation
import Testing

@testable import MoldClient

// The table is lifted from `web/src/lib/hostRegistry.test.ts`, deliberately.
// Three apps normalize addresses independently; if they disagree, one box
// typed into two of them becomes two machines in a merged library.
@Test(arguments: [
    ("192.168.1.42:51789", "http://192.168.1.42:51789"),
    ("https://box.tail1234.ts.net", "https://box.tail1234.ts.net"),
    ("http://studio.local:7680/api/status?x=1", "http://studio.local:7680"),
    ("studio.local", "http://studio.local:7680"),
    ("100.105.134.43", "http://100.105.134.43:7680"),
    // A complete URL keeps the scheme's own default port, like a browser.
    ("http://100.105.134.43", "http://100.105.134.43"),
    ("https://box.tail1234.ts.net:443", "https://box.tail1234.ts.net"),
    ("::1", "http://[::1]:7680"),
    ("plato", "http://plato:7680"),
    ("  plato  ", "http://plato:7680"),
    ("http://plato:7680/", "http://plato:7680"),
    ("HTTP://Plato:7680", "http://plato:7680"),
])
func normalizesAnAddressTheWayTheOtherAppsDo(input: String, expected: String) {
    #expect(HostAddress.normalize(input)?.absoluteString == expected)
}

@Test(arguments: [
    "100.123.198.98", "100.123.198.98:9000", "http://100.123.198.98",
    "https://box.tail1234.ts.net", "https://box.tail1234.ts.net:443", "::1", "plato",
])
func normalizingIsIdempotent(input: String) throws {
    let once = try #require(HostAddress.normalize(input))
    #expect(HostAddress.normalize(once.absoluteString) == once)
}

@Test func refusesInputThatIsNotAnAddressAndSaysWhy() {
    #expect(throws: HostAddress.Problem.empty) { try HostAddress.resolve("   ") }
    #expect(throws: HostAddress.Problem.unparseable) { try HostAddress.resolve("http://") }
    // Prefixing this would parse as the host `ftp` and point somewhere else.
    #expect(throws: HostAddress.Problem.unsupportedScheme) { try HostAddress.resolve("ftp://plato") }
}

@Test func suggestsAHostnameToNameTheMachineAfter() throws {
    let local = try #require(HostAddress.normalize("studio.local"))
    #expect(HostAddress.suggestedName(for: local) == "studio")
    let ip = try #require(HostAddress.normalize("100.105.134.43:7680"))
    #expect(HostAddress.suggestedName(for: ip) == "100.105.134.43")
    let v6 = try #require(HostAddress.normalize("::1"))
    #expect(HostAddress.suggestedName(for: v6) == "::1")
}

@Test func hidesTheOrdinarySchemeWhenShowingAnAddress() throws {
    let plain = try #require(HostAddress.normalize("plato"))
    #expect(HostAddress.displayString(for: plain) == "plato:7680")
    let secure = try #require(HostAddress.normalize("https://box.ts.net"))
    #expect(HostAddress.displayString(for: secure) == "https://box.ts.net")
}

@Test func recognizesTwoSpellingsOfOneMachine() throws {
    let typed = try #require(HostAddress.normalize("PLATO"))
    let saved = try #require(HostAddress.normalize("http://plato:7680/"))
    #expect(HostAddress.sameOrigin(typed, saved))
    let other = try #require(HostAddress.normalize("plato:7681"))
    #expect(!HostAddress.sameOrigin(typed, other))
}
