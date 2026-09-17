import SwiftUI

/// A used-of-total figure that is safe to draw.
///
/// Failable ON PURPOSE. `total_bytes` is null wherever a backend does not
/// report it, and a 0-of-0 bar reads as a FULL one -- the worst possible lie
/// to tell about memory. No reading means no bar, and the row says the size
/// is not reported instead.
struct MemoryReading: Equatable {
    let used: UInt64
    let total: UInt64

    init?(used: UInt64?, total: UInt64?) {
        guard let total, total > 0 else { return nil }
        self.total = total
        // A machine reporting more used than it has is a machine with a
        // rounding bug, not a bar that should run off its end.
        self.used = min(used ?? 0, total)
    }

    var fraction: Double { Double(used) / Double(total) }

    var percentage: String { fraction.formatted(.percent.precision(.fractionLength(0))) }
}

/// The bar a GPU row and the system row share.
///
/// `Gauge` with `.accessoryLinearCapacity` rather than a `ProgressView`: a
/// progress bar means "this is happening and will finish", which is the wrong
/// thing to say about memory. A capacity gauge is the control macOS itself
/// uses for how full something is, and it has an intrinsic compact size, so it
/// sits at the trailing edge of a Form row instead of stretching across it.
struct MemoryBar: View {
    let reading: MemoryReading

    var body: some View {
        Gauge(value: reading.fraction) {
            Text("In use")
        }
        .gaugeStyle(.accessoryLinearCapacity)
        .labelsHidden()
        .frame(width: 92)
        .accessibilityLabel("In use")
        .accessibilityValue(reading.percentage)
    }
}
