import Foundation
import Testing

@testable import MoldClient

/// `GET /api/resources`, the 1 Hz sample behind the GPU and RAM bars.
@Suite struct ResourceSuite {

    @Test func decodesPlatosFourGpusAndTheZfsCredit() throws {
        let snapshot = try MoldJSON.decoder.decode(
            ResourceSnapshot.self, from: RepoFixtures.fixture("resources-plato.json"))

        #expect(snapshot.hostname == "plato")
        #expect(snapshot.gpus.count == 4)
        #expect(snapshot.gpus[0].vramTotal == 48_305_799_168)
        #expect(snapshot.systemRam.total == 1_623_070_584_832)
    }

    @Test func decodesHal9000sSingleGpuWithCpuPresent() throws {
        let snapshot = try MoldJSON.decoder.decode(
            ResourceSnapshot.self, from: RepoFixtures.fixture("resources-hal9000.json"))

        #expect(snapshot.gpus.count == 1)
        #expect(snapshot.cpu?.cores == 32)
    }

    /// `cpu` is absent until the aggregator has taken two samples -- CPU use
    /// is a delta, and the first snapshot would always read zero. Written
    /// against a hand-built frame rather than a captured fixture: both
    /// captures on file happened to land after the aggregator's second tick.
    @Test func aSnapshotWithNoCpuSectionStillDecodes() throws {
        let json = """
        {"hostname":"h","timestamp":1,"gpus":[],
         "system_ram":{"total":100,"used":50,"used_by_mold":10}}
        """
        let snapshot = try MoldJSON.decoder.decode(ResourceSnapshot.self, from: Data(json.utf8))
        #expect(snapshot.cpu == nil)
    }

    /// `available` (`MemAvailable`) is additive and absent on an older host.
    @Test func availableFallsBackToTotalMinusUsedWhenTheHostDoesNotReportIt() throws {
        let json = """
        {"hostname":"h","timestamp":1,"gpus":[],
         "system_ram":{"total":100,"used":40,"used_by_mold":10}}
        """
        let snapshot = try MoldJSON.decoder.decode(ResourceSnapshot.self, from: Data(json.utf8))
        #expect(snapshot.systemRam.available == nil)
        #expect(snapshot.systemRam.availableOrComputed == 60)
    }
}
