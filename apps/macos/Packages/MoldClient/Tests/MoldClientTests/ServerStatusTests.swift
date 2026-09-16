import Foundation
import Testing

@testable import MoldClient

// Captured verbatim from hal9000 (`GET /api/status`, mold 0.28.0). Decoding a
// real payload is the point -- a fixture we wrote ourselves would only prove
// we can round-trip our own assumptions.
private let hal9000Status = """
{"version":"0.28.0","git_sha":"b2dbb45d","build_date":"2026-09-12",
 "models_loaded":[],"busy":false,
 "gpu_info":{"name":"NVIDIA GeForce RTX 4090","vram_total_mb":24564,
             "vram_used_mb":2161,"backend":"cuda"},
 "uptime_secs":232723,"hostname":"hal9000","memory_status":"VRAM: 23.5 GB free",
 "gpus":[{"ordinal":0,"name":"NVIDIA GeForce RTX 4090",
          "vram_total_bytes":25757220864,"vram_used_bytes":2266431488,
          "state":"idle"}],
 "queue_depth":0,"queue_capacity":200,"queue_paused":false,
 "instance_id":"ff00bea2-a8fc-4ffa-80c6-f5f80cfa5580"}
""".data(using: .utf8)!

@Test func decodesAServerStatusAndIgnoresFieldsItDoesNotModel() throws {
    let status = try JSONDecoder().decode(ServerStatus.self, from: hal9000Status)

    #expect(status.version == "0.28.0")
    #expect(status.hostname == "hal9000")
    #expect(status.busy == false)
    #expect(status.queueDepth == 0)
    #expect(status.gpus?.count == 1)
    #expect(status.gpus?.first?.name == "NVIDIA GeForce RTX 4090")
    #expect(status.gpus?.first?.vramTotalBytes == 25_757_220_864)
}

@Test func aKeylessHostCarriesNoKey() {
    let host = MoldHost(name: "hal9000", baseURL: URL(string: "http://100.123.198.98:7680")!)
    #expect(host.apiKey == nil)
}
