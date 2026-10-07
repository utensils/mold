import Foundation
import Testing
@testable import MoldClient

@Test func queueInputsKeepOrderedRoleLabelsAndPreviewAvailability() throws {
    let inputs = try MoldJSON.decoder.decode([QueueInput].self, from: Data(#"[{"index":2,"label":"Reference image 1","preview":true},{"index":4,"label":"Reference 2 · audio","preview":false}]"#.utf8))
    #expect(inputs.map(\.index) == [2, 4])
    #expect(inputs[0].label == "Reference image 1")
    #expect(!inputs[1].preview)
}

@Test func queueMemberThumbnailKeepsHostPrefixAndAuthentication() {
    let backend = HTTPBackend(host: MoldHost(name: "box", baseURL: URL(string: "http://box/mold")!, apiKey: "secret"))
    let request = backend.request(backend.queueInputThumbnailPath("job #1", index: 4))
    #expect(request.url?.absoluteString == "http://box/mold/api/queue/job%20%231/input-thumbnail?index=4")
    #expect(request.value(forHTTPHeaderField: "X-Api-Key") == "secret")
}
