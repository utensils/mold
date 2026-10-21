import Foundation
import MoldClient
import MoldClientTesting
import SwiftUI
import Testing
import UIKit

@testable import MoldCompanion

@MainActor
struct QueueConnectionViewTests {
    @Test func openQueueRebindsPausedParentAfterLateCapabilitiesAndConnectionMetadata() async throws {
        let caps = try MoldJSON.decoder.decode(Capabilities.self, from: Data(
            #"{"video_jobs":{"supported":true,"literal_prompts":true,"library_extend":true}}"#.utf8))
        let status = try MoldJSON.decoder.decode(ServerStatus.self, from: Data(
            #"{"version":"test","busy":false,"uptime_secs":1,"instance_id":"fixture-installation"}"#.utf8))
        let parent = try MoldJSON.decoder.decode(VideoJob.self, from: Data(#"{"id":"fixture-paused","state":"paused","plan":{"validation_revision":"v","model":"Fixture video","width":768,"height":768,"fps":24,"single_clip_max_frames":124,"overlap_frames":9,"join_mode":"continuous","clip_count":"12","part_count":"1","added_frames":"1440","source_frames":"0","total_frames":"1440","added_duration_ms":"60000","total_duration_ms":"60000","warnings":[]},"completed_clips":"0","active_child_id":null,"output_filename":null,"error":null,"error_detail":null,"created_at_ms":1,"updated_at_ms":2,"can_cancel":false,"can_resume":true,"can_delete":true}"#.utf8))
        let emptyQueue = try MoldJSON.decoder.decode(QueueListing.self, from: Data(#"{"entries":[]}"#.utf8))
        let emptyChains = try MoldJSON.decoder.decode(ChainJobListing.self, from: Data(#"{"jobs":[]}"#.utf8))
        let listing = try MoldJSON.decoder.decode(VideoJobListing.self, from: Data("{\"jobs\":[".utf8)
            + MoldJSON.encoder.encode(parent) + Data("],\"has_more\":false,\"next_cursor\":null}".utf8))
        let directory = FileManager.default.temporaryDirectory.appending(path: "queue-view-\(UUID())")
        var backends: [FakeBackend] = []
        let hosts = HostStore(list: HostListFile(url: directory.appending(path: "hosts.json")),
                              credentials: HostStoreTests.MemoryCredentials(), makeBackend: { host in
            let fake = FakeBackend(host: host)
            fake.stub("status()", returning: status)
            fake.stub("capabilities()", returning: caps)
            fake.stub("queue()", returning: emptyQueue)
            fake.stub("listChainJobs()", returning: emptyChains)
            fake.stub("listVideoJobs(limit:cursor:)", returning: listing)
            backends.append(fake)
            return fake
        })
        let host = MoldHost(name: "Queue reconnect fixture", baseURL: URL(string: "http://127.0.0.1:17699")!, connectionMode: .regular)
        hosts.setHosts([host]); hosts.setReachability(.up(status), for: host.id)
        let defaults = UserDefaults(suiteName: "queue-view-\(UUID())")!
        let videos = VideoJobCoordinator(defaults: defaults)
        let generate = GenerateController(hosts: hosts, ledger: PendingLedger(url: directory.appending(path: "ledger.json")),
            drafts: DraftStore(directory: directory), sequence: NativeSequenceSession(defaults: defaults),
            videoJobs: videos, videoRestarts: VideoRestartController(defaults: defaults))
        let queue = QueueStore(hosts: hosts)
        let view = NavigationStack { QueueView() }.environment(hosts).environment(queue).environment(generate).environment(AppRouter())
        let scene = try #require(UIApplication.shared.connectedScenes.compactMap { $0 as? UIWindowScene }.first)
        let window = UIWindow(windowScene: scene)
        window.rootViewController = UIHostingController(rootView: view)
        window.makeKeyAndVisible()
        defer { window.isHidden = true; window.rootViewController = nil; videos.suspend() }
        try await Task.sleep(for: .milliseconds(150))
        #expect(videos.jobs.isEmpty)
        hosts.setCapabilities(caps, for: host.id)
        for _ in 0..<100 where videos.jobs.isEmpty { try await Task.sleep(for: .milliseconds(20)) }
        #expect(videos.jobs.first?.detail?.canResume == true)
        var refreshed = host
        refreshed.connectionInstanceID = status.instanceId
        refreshed.connectionOriginalURL = host.baseURL
        hosts.setHosts([refreshed])
        for _ in 0..<100 where videos.jobs.first?.owner.host != refreshed { try await Task.sleep(for: .milliseconds(20)) }
        #expect(videos.jobs.first?.owner.host == refreshed, "the already-visible Queue must replace stale owner rows without navigation or pull-to-refresh")
        #expect(videos.jobs.first?.detail?.state == .paused)
        #expect(backends.flatMap(\.calls).allSatisfy { !$0.route.contains("cancelVideoJob") && !$0.route.contains("resumeVideoJob") && !$0.route.contains("createVideoJob") })
        try await Task.sleep(for: .milliseconds(100))
        let image = UIGraphicsImageRenderer(bounds: window.bounds).image { _ in
            window.drawHierarchy(in: window.bounds, afterScreenUpdates: true)
        }
        let output = directory.appending(path: "queue-reconnected.png")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try image.pngData()?.write(to: output)
        print("QUEUE_RECONNECT_SCREENSHOT \(output.path)")
    }
}
