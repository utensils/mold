import AppKit
import MoldClient
import SwiftUI

/// The owning server's retained input, including work submitted by another client.
struct QueueSourceThumbnail: View {
    @Environment(HostStore.self) private var hosts
    let entry: QueueEntry
    let host: MoldHost
    @State private var image: NSImage?
    @State private var loading = true

    private var identity: String {
        "\(host.id)|\(hosts.instanceID(of: host.id) ?? "unknown")|\(entry.id)|\(hosts.isUp(host))"
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            if loading || image != nil {
                VStack(alignment: .leading, spacing: 3) {
                    Group {
                        if let image { Image(nsImage: image).resizable().scaledToFill() }
                        else { Color.clear }
                    }
                    .frame(width: 48, height: 48)
                    .clipShape(.rect(cornerRadius: 6))
                    .accessibilityLabel("Source image for this render")
                    .accessibilityIdentifier("queue-source-" + entry.id)
                    Text("Source").font(.caption).foregroundStyle(.secondary)
                }
                // Native List measures before the authenticated thumbnail arrives.
                // Reserve the same image and intrinsic caption geometry while pending.
                .opacity(image == nil ? 0 : 1)
                .accessibilityHidden(image == nil)
            }
        }
        .task(id: identity) {
            image = nil
            loading = true
            guard hosts.isUp(host) else { loading = false; return }
            let requestIdentity = identity
            defer { if !Task.isCancelled, identity == requestIdentity { loading = false } }
            guard let bytes = try? await hosts.backend(for: host).queueInputThumbnail(id: entry.id),
                  !Task.isCancelled, identity == requestIdentity,
                  bytes.count <= 2 * 1024 * 1024 else { return }
            image = NSImage(data: bytes)
        }
    }
}
