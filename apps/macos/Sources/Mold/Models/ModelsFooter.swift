import MoldClient
import SwiftUI

/// "82 installed on workstation · 1.71 TB of 2.5 TB used".
///
/// The second clause is the MACHINE's own `status.modelsDisk` figure and
/// NEVER a sum of the Size column: a shared VAE or encoder is counted once
/// per model that references it, so the column can legitimately add up to
/// more than the disk actually holds (design fact 3, M5). Absent on a host
/// that hasn't reported one -- an older server -- the sentence simply stops
/// after the count, never showing a fabricated zero.
struct ModelsFooter: View {
    let count: Int
    let host: MoldHost?
    let status: ServerStatus?

    var body: some View {
        HStack {
            Text(Self.sentence(count: count, hostName: host?.name, disk: status?.modelsDisk))
                .font(.caption)
                .foregroundStyle(.secondary)
            Spacer()
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
    }

    /// The sentence alone, pure -- so a test can ask the exact question the
    /// footer asks without a view or a host (design fact 3, M5, pinned).
    static func sentence(count: Int, hostName: String?, disk: ServerStatus.ModelsDisk?) -> String {
        guard let hostName else { return "No machine" }
        let base = "\(count) installed on \(hostName)"
        guard let disk else { return base }
        let used = disk.totalBytes - disk.freeBytes
        return "\(base) · \(FileBytes.text(used)) of \(FileBytes.text(disk.totalBytes)) used"
    }
}
