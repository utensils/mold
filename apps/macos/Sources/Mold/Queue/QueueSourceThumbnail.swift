import AppKit
import MoldClient
import SwiftUI

/// The owning server's retained input, including work submitted by another client.
struct QueueSourceThumbnail: View {
    @Environment(HostStore.self) private var hosts
    let entry: QueueEntry
    let host: MoldHost
    @State private var image: NSImage?

    private var identity: String {
        "\(host.id)|\(hosts.instanceID(of: host.id) ?? "unknown")|\(entry.id)|\(hosts.isUp(host))"
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            if let image {
                VStack(alignment: .leading, spacing: 3) {
                    Image(nsImage: image).resizable().scaledToFill()
                        .frame(width: 48, height: 48)
                        .clipShape(.rect(cornerRadius: 6))
                        .accessibilityLabel("Source image for this render")
                        .accessibilityIdentifier("queue-source-" + entry.id)
                    Text("Source").font(.caption).foregroundStyle(.secondary)
                }
            }
        }
        .task(id: identity) {
            image = nil
            guard hosts.isUp(host) else { return }
            let requestIdentity = identity
            guard let bytes = try? await hosts.backend(for: host).queueInputThumbnail(id: entry.id),
                  !Task.isCancelled, identity == requestIdentity,
                  bytes.count <= 2 * 1024 * 1024 else { return }
            image = NSImage(data: bytes)
        }
    }
}
