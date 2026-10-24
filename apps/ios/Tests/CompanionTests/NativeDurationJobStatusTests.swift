import Foundation
import MoldClient
import Testing
@testable import MoldCompanion

@MainActor struct NativeDurationJobStatusTests {
    @Test func unavailableAuthorityQualifiesCachedProgressWithoutClaimingTheMachineIsOffline() throws {
        let detail = try MoldJSON.decoder.decode(VideoJob.self, from: Data(#"{"id":"saved","state":"running","plan":{"validation_revision":"proof","model":"fixture","width":768,"height":768,"fps":24,"single_clip_max_frames":124,"overlap_frames":0,"join_mode":"continuous","clip_count":"12","part_count":"1","added_frames":"1440","source_frames":"0","total_frames":"1440","added_duration_ms":"60000","total_duration_ms":"60000","warnings":[]},"completed_clips":"0","active_child_id":null,"output_filename":null,"error":null,"error_detail":null,"created_at_ms":1,"updated_at_ms":2,"can_cancel":true,"can_resume":false,"can_delete":false}"#.utf8))
        #expect(VideoJobRow(detail, current: false).caption == "Last known: Rendering clip 1 of 12")
        #expect(VideoJobRow(detail, current: true).caption == "Rendering clip 1 of 12")
        #expect(VideoJobRow(detail, current: true).detail == "60 s video · 12 clips")
    }
}
