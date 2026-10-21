import Foundation

extension FixtureMachine {
    func durationQueueResponse(_ path: String) -> Data? {
        if path == "/api/status" { return Data(#"{"version":"fixture","busy":false,"uptime_secs":1,"instance_id":"duration-fixture"}"#.utf8) }
        if path == "/api/capabilities" { return Data(#"{"video_jobs":{"supported":true,"literal_prompts":true,"library_extend":false}}"#.utf8) }
        if path == "/api/chain-jobs" { return Data(#"{"jobs":[]}"#.utf8) }
        if path == "/api/video-jobs" {
            return Data(("{\"jobs\":[" + durationParent(completed: durationCompleted.withLock { $0 }) + "],\"has_more\":false,\"next_cursor\":null}").utf8)
        }
        if path == "/api/video-jobs/fixture-completed" {
            durationCompleted.withLock { $0 = true }
            return Data(durationParent(completed: true).utf8)
        }
        return nil
    }
    private func durationParent(completed: Bool) -> String {
        """
        {"id":"fixture-completed","state":"\(completed ? "completed" : "running")","plan":{"validation_revision":"v","model":"Completed video fixture","width":768,"height":768,"fps":24,"single_clip_max_frames":124,"overlap_frames":9,"join_mode":"continuous","clip_count":"12","part_count":"1","added_frames":"1440","source_frames":"0","total_frames":"1440","added_duration_ms":"60000","total_duration_ms":"60000","warnings":[]},"completed_clips":"\(completed ? "12" : "0")","active_child_id":null,"output_filename":"movie.mp4","error":null,"error_detail":null,"created_at_ms":1,"updated_at_ms":2,"can_cancel":false,"can_resume":false,"can_delete":true}
        """
    }
}
