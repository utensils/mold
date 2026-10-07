import AppKit
import MoldClient
import SwiftUI

/// The owning server's retained input, including work submitted by another client.
struct QueueSourceThumbnail: View {
    @Environment(HostStore.self) private var hosts
    let entry: QueueEntry
    let host: MoldHost
    var size: CGFloat = 48
    var detailed = false
    @State private var previews: [QueueInputPreview] = []
    @State private var image: NSImage?
    @State private var loading = true
    @State private var retry = 0
    @State private var failed = false

    private var identity: String {
        "\(host.id)|\(hosts.instanceID(of: host.id) ?? "unknown")|\(entry.id)|\(hosts.isUp(host))|\(retry)"
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if detailed {
                if failed { Text("Input previews unavailable").foregroundStyle(.secondary) }
                ForEach(previews) { preview in
                    VStack(alignment: .leading, spacing: 4) {
                        Text(preview.input.label).font(.headline)
                        if let bytes = preview.bytes, let image = NSImage(data: bytes) {
                            Image(nsImage: image).resizable().scaledToFit()
                                .frame(maxWidth: 320, maxHeight: 320)
                                .accessibilityLabel(preview.input.label)
                        } else {
                            Text(preview.input.preview ? "Preview unavailable" : "No still preview")
                                .foregroundStyle(.secondary)
                        }
                    }
                }
                if failed || previews.contains(where: { $0.input.preview && $0.bytes == nil }) {
                    Button("Retry input previews") { retry += 1 }
                }
            } else if loading || image != nil {
                VStack(alignment: .leading, spacing: 3) {
                    Group {
                        if let image { Image(nsImage: image).resizable().scaledToFit() }
                        else { Color.clear }
                    }
                    .frame(width: size, height: size)
                    .clipShape(.rect(cornerRadius: 6))
                    .accessibilityLabel(previews.first(where: { $0.bytes != nil })?.input.label ?? "Source")
                    .accessibilityIdentifier("queue-source-" + entry.id)
                    Text(previews.first(where: { $0.bytes != nil })?.input.label ?? "Source")
                        .font(.caption).foregroundStyle(.secondary)
                    if previews.count > 1 { Text("+\(previews.count - 1) inputs").font(.caption).foregroundStyle(.secondary) }
                }
                .opacity(image == nil ? 0 : 1)
                .accessibilityHidden(image == nil)
            }
        }
        .task(id: identity) {
            failed = false
            image = nil
            previews = []
            loading = true
            guard hosts.isUp(host) else { loading = false; return }
            let requestIdentity = identity
            defer { if !Task.isCancelled, identity == requestIdentity { loading = false } }
            do {
                let result = try await hosts.backend(for: host).queueInputPreviews(id: entry.id, firstOnly: !detailed)
                guard !Task.isCancelled, identity == requestIdentity else { return }
                previews = result
                image = result.first(where: { $0.bytes != nil })?.bytes.flatMap(NSImage.init(data:))
            } catch {
                if !Task.isCancelled, identity == requestIdentity { failed = true }
            }
        }
    }
}
