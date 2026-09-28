import Foundation
import Testing

@testable import MoldClient

/// The device sentences both apps print. Moved from the Mac app into
/// MoldClient so the iPhone's machine page cannot word a card differently.
struct DeviceWordsTests {
    private func device(admin: String = "enabled", activity: String = "idle",
                        health: String = "healthy") throws -> DeviceInfo {
        let json = """
        {"id":"gpu0","name":"NVIDIA L40S","ordinal":0,"device_kind":"full_gpu",
         "memory":{"total_bytes":48000000000,"used_bytes":16000000000},
         "telemetry":{},"desired_enabled":true,"admin_state":"\(admin)",
         "health":"\(health)","activity":"\(activity)","schedulable":true,"loaded_models":[]}
        """
        return try MoldJSON.decoder.decode(DeviceInfo.self, from: Data(json.utf8))
    }

    @Test func anAdminStateOutranksTheActivity() throws {
        #expect(DeviceWords.state(try device(admin: "draining", activity: "idle")) == "Finishing what it has")
        #expect(DeviceWords.state(try device(activity: "generating")) == "Rendering")
    }

    @Test func healthIsSaidOnlyWhenItIsNotHealthy() {
        #expect(DeviceWords.health(.healthy) == nil)
        #expect(DeviceWords.health(.poisoned) == "Faulted")
    }

    @Test func aMissingFigureIsNeverZero() {
        #expect(DeviceWords.utilization(nil) == nil)
        #expect(DeviceWords.memory(used: 5, total: nil, mold: nil) == "Memory not reported")
    }

    @Test func holdingNamesWhatIsLoadedOrNothing() {
        #expect(DeviceWords.holding([]) == nil)
        #expect(DeviceWords.holding(["flux-dev:q4"]) == "Holding flux-dev:q4")
    }
}
