import Foundation
import Testing
import MoldClientTesting
@testable import MoldClient

extension VideoJobCoordinatorTests {
    @Test func finishedParentsLeaveTheQueueLikeEveryOtherRow() async throws {
        let (_, owner, defaults) = try setup()
        let coordinator = VideoJobCoordinator(defaults: defaults)
        coordinator.publish(try saved("done"), owner: owner)
        coordinator.publish(try detail("stopped", state: "cancelled", cancel: false), owner: owner)
        #expect(coordinator.jobs.allSatisfy { !$0.belongsInQueue })
        #expect(!coordinator.hasQueueContent, "a finished video must not keep an otherwise-empty Queue open")
        for state in ["queued", "running", "paused", "failed", "interrupted"] {
            coordinator.publish(try detail(state, state: state), owner: owner)
        }
        #expect(coordinator.jobs.filter(\.belongsInQueue).map(\.operationId)
            == ["queued", "running", "paused", "failed", "interrupted"])
        #expect(coordinator.hasQueueContent)
        var unconfirmed = ObservedVideoJob(operationId: "checking", owner: owner)
        #expect(unconfirmed.belongsInQueue, "an admission still being confirmed is live work")
        unconfirmed.cancelledBeforeAdmission = true
        #expect(!unconfirmed.belongsInQueue)
    }

    func saved(_ id: String) throws -> VideoJob {
        try MoldJSON.decoder.decode(VideoJob.self, from: Data("""
        {"id":"\(id)","state":"completed","plan":\(VideoJobContractTests.plan()),"completed_clips":"1",
         "active_child_id":null,"output_filename":"video.mp4","error":null,"error_detail":null,
         "created_at_ms":1,"updated_at_ms":2,"can_cancel":false,"can_resume":false,"can_delete":false}
        """.utf8))
    }

    @Test func aCompletionWithNoSavedVideoStaysToSaySo() async throws {
        let (_, owner, defaults) = try setup()
        let coordinator = VideoJobCoordinator(defaults: defaults)
        coordinator.publish(try detail("lost", state: "completed", cancel: false), owner: owner)
        let job = try #require(coordinator.jobs.first)
        #expect(job.message != nil)
        #expect(job.belongsInQueue, "the only place the missing-video warning appears must stay visible")
        #expect(coordinator.hasQueueContent)
    }

    @Test func authoredSequencesLeaveTheQueueByTheSameRule() throws {
        func chain(_ state: String, gallery: String? = nil) throws -> ChainJobDetail {
            let file = gallery.map { #","finalizes":[{"gallery_filename":"\#($0)"}]"# } ?? ""
            return try MoldJSON.decoder.decode(ChainJobDetail.self, from: Data(
                #"{"id":"c","state":"\#(state)","model":"model","stage_count":2,"current_stage":2\#(file)}"#.utf8))
        }
        #expect(!(try chain("completed", gallery: "video.mp4")).belongsInQueue)
        #expect(try chain("completed").belongsInQueue, "finished without a Library file keeps its warning row")
        #expect(!(try chain("cancelled")).belongsInQueue)
        for state in ["queued", "running", "paused", "failed", "interrupted"] {
            #expect(try chain(state).belongsInQueue, "\(state)")
        }
    }
}
