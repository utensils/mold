import Testing
@testable import MoldClient

struct SourceMediaFilterTests {
    @Test func unknownKindsStayInOtherAndAllIncludesEveryRole() {
        let roles = [("source_image", "image"), ("audio_file", "audio"), ("source_video", "video"), ("future", "unknown")]
        for (role, kind) in roles {
            let member = RetainedSourceMedia.Member(memberId: role, role: role, displayName: "same", sizeBytes: 1)
            #expect(SourceMediaFilter.all.includes(member))
            #expect(SourceMediaFilter.allCases.filter { $0 != .all && $0.includes(member) }.map(\.rawValue) == [kind])
        }
    }
}
