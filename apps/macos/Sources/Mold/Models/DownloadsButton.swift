import MoldClient
import MoldStyle
import SwiftUI

/// The toolbar's own download tray, Safari's idiom: present only while this
/// machine has something to say, gone the moment it is idle. The popover is
/// the only surface for any of this -- no persistent tray, no row anywhere
/// else in the window (decision 16, M5).
struct DownloadsButton: View {
    let host: MoldHost
    @Environment(DownloadStore.self) private var downloads
    @State private var showsPopover = false

    private var active: [String: DownloadStore.Progress] { downloads.active[host.id] ?? [:] }
    private var finished: [DownloadJob] { downloads.finished[host.id] ?? [] }

    var body: some View {
        if Self.isShown(active: active, finished: finished) {
            Button { showsPopover.toggle() } label: { icon }
                .buttonStyle(.plain)
                .help("Downloads")
                .accessibilityLabel("Downloads")
                .popover(isPresented: $showsPopover) {
                    DownloadsPopover(host: host)
                }
        }
    }

    @ViewBuilder private var icon: some View {
        ZStack(alignment: .topTrailing) {
            if active.isEmpty {
                // Nothing in flight, but something finished this launch --
                // the popover still has a row and a Clear button to offer.
                Image(systemName: "arrow.down.circle")
            } else {
                ProgressView(value: Self.combinedFraction(active: active) ?? 0)
                    .progressViewStyle(.circular)
                    .controlSize(.small)
            }
            if active.count > 1 {
                Text("\(active.count)")
                    .font(.system(size: 9, weight: .bold))
                    .monospacedDigit()
                    .foregroundStyle(.white)
                    .padding(3)
                    .background(Chrome.badgeBackdrop, in: Circle())
                    .offset(x: 8, y: -8)
            }
        }
        .frame(width: 20, height: 20)
    }

    /// Present whenever there is anything this machine would want to say:
    /// a job running, one queued, or one this app watched finish since
    /// launch. Never on a machine with none of the three.
    static func isShown(active: [String: DownloadStore.Progress], finished: [DownloadJob]) -> Bool {
        !active.isEmpty || !finished.isEmpty
    }

    /// Bytes done over bytes total, across every job that HAS reported a
    /// total -- a job still resolving its manifest contributes nothing to
    /// either side rather than dragging the ring back to zero.
    static func combinedFraction(active: [String: DownloadStore.Progress]) -> Double? {
        let known = active.values.filter { ($0.bytesTotal ?? 0) > 0 }
        guard !known.isEmpty else { return nil }
        let done = known.reduce(Int64(0)) { $0 + ($1.bytesDone ?? 0) }
        let total = known.reduce(Int64(0)) { $0 + ($1.bytesTotal ?? 0) }
        guard total > 0 else { return nil }
        return Double(done) / Double(total)
    }
}
