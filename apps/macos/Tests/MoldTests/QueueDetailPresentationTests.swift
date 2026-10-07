import Foundation
import MoldClient
import Testing
@testable import Mold

struct QueueDetailPresentationTests {
    private func progress(step: Int? = nil, total: Int? = nil, stage: String? = nil) throws -> JobProgress {
        var json: [String: Any] = [:]
        json["step"] = step
        json["total"] = total
        json["stage"] = stage
        return try MoldJSON.decoder.decode(JobProgress.self, from: JSONSerialization.data(withJSONObject: json))
    }

    @Test func preparingDoesNotInventStepProgress() {
        let status = QueueDetailPresentation.status(current: FakeFixtures.queueEntry("job", state: "running"), progress: nil)
        #expect(status.title == "Rendering")
        #expect(status.message == "Getting ready…")
        #expect(status.fraction == nil)
        #expect(status.stepText == nil)
    }

    @Test func finishedPausedAndDepartedJobsCannotShowStaleProgress() throws {
        let last = try progress(step: 3, total: 10, stage: "Sampling")
        for state in ["queued", "paused", "held", "complete", "failed", "cancelled", "cancelling", "unknown"] {
            let status = QueueDetailPresentation.status(current: FakeFixtures.queueEntry("job", state: state), progress: last)
            #expect(status.fraction == nil)
            #expect(status.stepText == nil)
            #expect(status.message != "Sampling")
        }
        let absent = QueueDetailPresentation.status(current: nil, progress: last)
        #expect(absent.title == "No longer queued")
        #expect(absent.fraction == nil)
    }

    @Test func progressIsBoundedAndUnknownTotalsStayIndeterminate() throws {
        let running = FakeFixtures.queueEntry("job", state: "running")
        let complete = QueueDetailPresentation.status(current: running, progress: try progress(step: 12, total: 10))
        #expect(complete.fraction == 1)
        #expect(complete.stepText == "Step 10 of 10")
        #expect(QueueDetailPresentation.status(current: running, progress: try progress(step: 2, total: 0)).fraction == nil)
        #expect(QueueDetailPresentation.status(current: running, progress: try progress(step: -1, total: 10)).fraction == nil)
    }

    @Test func promptHeadingIsNotRepeatedAndPromptTextIsNeverRewritten() throws {
        let metadata = try MoldJSON.decoder.decode(OutputMetadata.self, from: Data(#"{"prompt":"Prompt","negative_prompt":"blur"}"#.utf8))
        let group = try #require(PrintDetails.groups(for: metadata).first { $0.title == "Prompt" })
        let prompt = try #require(group.rows.first { $0.label == "Prompt" })
        let negative = try #require(group.rows.first { $0.label == "Negative prompt" })
        #expect(!QueueDetailPresentation.showsLabel(prompt, in: group))
        #expect(QueueDetailPresentation.showsLabel(negative, in: group))
        #expect(prompt.value == "Prompt")
    }
}
