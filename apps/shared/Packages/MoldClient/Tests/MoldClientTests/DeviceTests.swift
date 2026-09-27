import Foundation
import Testing

@testable import MoldClient

/// `GET /api/devices`, decoded from real payloads -- workstation's four L40S cards,
/// hal9000's single 4090, and this Mac's own embedded Metal engine.
@Suite struct DeviceSuite {

    @Test func decodesWorkstationsFourCardsWithTheirOpaqueIdentitiesIntact() throws {
        let state = try MoldJSON.decoder.decode(
            DeviceState.self, from: RepoFixtures.fixture("devices-workstation.json"))

        #expect(state.devices.count == 4)
        for device in state.devices { #expect(device.id.hasPrefix("cuda:")) }
        #expect(state.devices.map(\.ordinal) == [0, 1, 2, 3])
        #expect(state.devices[0].memory.totalBytes == 48_305_799_168)
        #expect(state.devices[0].loadedModels.first == "qwen-image-2512:q8")
    }

    @Test func decodesHal9000sSingleCardWithNothingLoaded() throws {
        let state = try MoldJSON.decoder.decode(
            DeviceState.self, from: RepoFixtures.fixture("devices-hal9000.json"))

        #expect(state.devices.count == 1)
        #expect(state.devices[0].loadedModels.isEmpty)
        #expect(state.devices[0].deviceKind == .fullGpu)
    }

    /// The one that catches the silent bug: `keyDecodingStrategy` converts
    /// *keys*, not *values* -- `DeviceKind(rawValue:)` must match the WIRE
    /// spelling exactly, and every one of these degrades silently to
    /// `.unknown` rather than throwing if it does not.
    @Test func everySnakeCaseDeviceEnumMatchesItsWireSpelling() throws {
        func decode<T: OpenWireEnum>(_ type: T.Type, _ raw: String) throws -> T {
            try MoldJSON.decoder.decode(T.self, from: Data("\"\(raw)\"".utf8))
        }

        #expect(try decode(DeviceKind.self, "full_gpu") == .fullGpu)
        #expect(try decode(DeviceKind.self, "unknown_cuda") == .unknownCuda)
        #expect(try decode(DeviceAdminState.self, "startup_excluded") == .startupExcluded)
        #expect(try decode(DeviceActivity.self, "admin_loading") == .adminLoading)
        // A value this build has never heard of degrades rather than throws.
        #expect(try decode(DeviceKind.self, "quantum_gpu_9000") == .unknown)
    }

    @Test func aDeviceThatDoesNotReportItsRestartFlagIsNotWaitingForOne() throws {
        let json = deviceJSON(restartRequired: nil)
        let device = try MoldJSON.decoder.decode(DeviceInfo.self, from: json)
        #expect(device.restartRequired == nil)
        #expect(device.needsRestart == false)
    }

    /// Fact 3, resolved by a live capture against this Mac's own embedded
    /// engine: the Metal device reports BOTH an ordinal and a total, which
    /// the design did not know going in. The `nil`-safe branches on
    /// `DeviceMemory` exist for a backend that omits them, but nothing in
    /// this fleet exercises that path.
    @Test func aMetalDeviceDecodesWithTheOrdinalAndTotalThisMacReports() throws {
        let state = try MoldJSON.decoder.decode(
            DeviceState.self, from: RepoFixtures.fixture("devices-metal.json"))
        let metal = try #require(state.devices.first)

        #expect(metal.id == "metal:default")
        #expect(metal.deviceKind == .metal)
        #expect(metal.ordinal == 0)
        #expect(metal.memory.totalBytes == 51_539_607_552)
        // CUDA-only attribution, absent on Metal.
        #expect(metal.memory.moldUsedBytes == nil)
        #expect(metal.memory.otherUsedBytes == nil)
    }

    @Test func nullTelemetryIsAbsenceNotZero() throws {
        let state = try MoldJSON.decoder.decode(
            DeviceState.self, from: RepoFixtures.fixture("devices-metal.json"))
        let metal = try #require(state.devices.first)
        #expect(metal.telemetry.utilizationPercent == nil)
    }

    /// `restartRequired == nil` OMITS the key -- an older host's own
    /// `#[serde(default)]` payload never sends it at all, and that must not
    /// be confused with an explicit `false`.
    private func deviceJSON(restartRequired: Bool?) -> Data {
        let restartField = restartRequired.map { #""restart_required":\#($0),"# } ?? ""
        let json = """
        {"id":"cuda:abc","name":"NVIDIA L40S","ordinal":0,"device_kind":"full_gpu",
         "memory":{"total_bytes":null,"used_bytes":null,"mold_used_bytes":null,"other_used_bytes":null},
         "telemetry":{"utilization_percent":null},
         "desired_enabled":true,\(restartField)
         "admin_state":"enabled","health":"healthy","activity":"idle",
         "schedulable":true,"unschedulable_reason":null,"loaded_models":[]}
        """
        return Data(json.utf8)
    }
}
