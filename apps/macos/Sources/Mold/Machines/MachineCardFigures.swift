import Foundation
import MoldClient

/// The whole fleet's worth of a machine, collapsed into the four figures a
/// card has room for.
///
/// Pure, and deliberately conservative: every one of these answers `nil`
/// rather than zero when nothing reported, because a card showing "0%" for a
/// machine that does not measure utilization is a card telling you it is idle.
enum MachineCardFigures {
    /// "4× NVIDIA L40S", "2 GPUs" for a mixed machine, the one name for a
    /// single card -- the way a person says it, which is also how the host
    /// editor's own sentence says it (`HostStatus.swift`'s `hardware`).
    ///
    /// `/api/devices` is the fuller answer and is preferred; `reported` is
    /// `GET /api/status`'s own list, which is all there is on a machine that
    /// does not let this app see its devices.
    static func gpus(devices: [DeviceInfo], reported: [ServerStatus.GPU]?) -> String? {
        let names = devices.isEmpty ? (reported ?? []).map(\.name) : devices.map(\.name)
        return collapse(names)
    }

    static func collapse(_ names: [String]) -> String? {
        guard let first = names.first else { return nil }
        guard names.count > 1 else { return first }
        return Set(names).count == 1 ? "\(names.count)× \(first)" : "\(names.count) GPUs"
    }

    /// One load figure for the machine: the mean over the cards that report
    /// one. Absent when none does -- only NVML reports utilization, so on
    /// Metal and on the `nvidia-smi` fallback there is no such number.
    static func load(devices: [DeviceInfo], snapshot: ResourceSnapshot?) -> String? {
        let percentages = devices.compactMap { device in
            sample(for: device, in: snapshot)?.gpuUtilization ?? device.telemetry.utilizationPercent
        }
        guard !percentages.isEmpty else { return nil }
        let mean = Double(percentages.reduce(0, +)) / Double(percentages.count)
        return "\(mean.rounded().formatted(.number.precision(.fractionLength(0))))%"
    }

    /// Video memory across every card, as one bar and one sentence. The live
    /// 1 Hz sample where there is one, the card's own last answer where there
    /// is not -- `DeviceRow` makes the same choice per card.
    static func videoMemory(devices: [DeviceInfo], snapshot: ResourceSnapshot?)
        -> MachineCard.MemoryFigure? {
        var used: UInt64 = 0
        var total: UInt64 = 0
        var mold: UInt64 = 0
        var moldReported = false
        for device in devices {
            let live = sample(for: device, in: snapshot)
            total += live.map(\.vramTotal) ?? device.memory.totalBytes ?? 0
            used += live?.vramUsed ?? device.memory.usedBytes ?? 0
            if let share = live?.vramUsedByMold ?? device.memory.moldUsedBytes {
                mold += share
                moldReported = true
            }
        }
        // No total, no bar: a 0-of-0 bar reads as a full one, and a machine
        // whose cards report no size has nothing to say here at all.
        guard let reading = MemoryReading(used: used, total: total) else { return nil }
        return MachineCard.MemoryFigure(
            reading: reading,
            text: DeviceWords.memory(used: used, total: total,
                                     mold: moldReported ? mold : nil))
    }

    /// The machine's own memory, which is a different thing from a card's.
    static func systemMemory(_ snapshot: ResourceSnapshot?) -> MachineCard.MemoryFigure? {
        guard let snapshot,
              let reading = MemoryReading(used: snapshot.systemRam.used,
                                          total: snapshot.systemRam.total)
        else { return nil }
        return MachineCard.MemoryFigure(
            reading: reading,
            text: DeviceWords.capacity(used: snapshot.systemRam.used,
                                       total: snapshot.systemRam.total))
    }

    /// Matched by ordinal within one host and one second, the only scope in
    /// which an ordinal means anything (`MachineStore.sample(for:on:)`).
    private static func sample(for device: DeviceInfo, in snapshot: ResourceSnapshot?) -> GpuSample? {
        guard let ordinal = device.ordinal else { return nil }
        return snapshot?.gpus.first { $0.ordinal == ordinal }
    }
}
