import Foundation
import Testing
@testable import MoldClient

@Test func userFacingErrorsMatchTheCrossSurfaceContract() throws {
    struct Fixture: Decodable { let raw: String; let message: String }
    let root = try #require(RepoFixtures.repoRoot)
    let data = try Data(contentsOf: root.appendingPathComponent("docs/contracts/user-errors.json"))
    for fixture in try MoldJSON.decoder.decode([Fixture].self, from: data) {
        #expect(UserFacingError.message(fixture.raw) == fixture.message)
        #expect(UserFacingError.message(fixture.message) == fixture.message)
    }
}

@Test func memoryNumbersRemainReadableAtTheByteBoundary() {
    #expect(UserFacingError.bytes(0) == "0 B")
    #expect(UserFacingError.bytes(999) == "999 B")
    #expect(UserFacingError.bytes(1000) == "1 KB")
    #expect(UserFacingError.bytes(1_000_000) == "1 MB")
}

@Test func localFailuresReferToThisDeviceAndAppLogs() {
    #expect(UserFacingError.localMessage("No such file or directory (os error 2)") == "That file is no longer available. Choose it again.")
    #expect(UserFacingError.localMessage("No space left on device") == "This device is out of storage. Free some disk space and try again.")
    #expect(UserFacingError.localMessage(String(repeating: "trace", count: 60)) == "That action failed. Check the app’s logs for details.")
}
