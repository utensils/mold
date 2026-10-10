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
    @State private var compactLabel = "Source"
    @State private var compactCount = 0
    @State private var retry = 0
    @State private var failed = false

    private struct RequestIdentity: Equatable {
        let host: MoldHost
        let instance: String
        let job: String
        let online: Bool
        let detailed: Bool
        let retry: Int
    }
    private var identity: RequestIdentity {
        RequestIdentity(host: host, instance: hosts.instanceID(of: host.id) ?? "unknown",
                        job: entry.id, online: hosts.isUp(host), detailed: detailed, retry: retry)
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
            } else {
                VStack(alignment: .leading, spacing: 3) {
                    Group {
                        if let image { Image(nsImage: image).resizable().scaledToFit() }
                        else { Color.clear }
                    }
                    .frame(width: size, height: size)
                    .clipShape(.rect(cornerRadius: 6))
                    .accessibilityLabel(compactLabel)
                    .accessibilityIdentifier("queue-source-" + entry.id)
                    Text(compactLabel)
                        .font(.caption).foregroundStyle(.secondary).lineLimit(1)
                    Text("+\(max(0, compactCount - 1)) inputs")
                        .font(.caption).foregroundStyle(.secondary).lineLimit(1)
                        .opacity(compactCount > 1 ? 1 : 0)
                        .accessibilityHidden(compactCount <= 1)
                }
                .frame(width: max(size, 80), alignment: .leading)
                .opacity(image == nil ? 0 : 1)
                .accessibilityHidden(image == nil)
            }
        }
        .task(id: identity) {
            failed = false
            image = nil
            previews = []
            compactCount = 0
            compactLabel = "Source"
            guard hosts.isUp(host) else { return }
            let requestIdentity = identity
            do {
                if detailed {
                    let result = try await hosts.backend(for: host).queueInputPreviews(id: entry.id)
                    guard !Task.isCancelled, identity == requestIdentity else { return }
                    previews = result
                } else {
                    let key = QueuePreviewCache.Key(host: host, instance: requestIdentity.instance, job: entry.id)
                    let result = try await hosts.queuePreviews.preview(for: key, backend: hosts.backend(for: host))
                    guard !Task.isCancelled, identity == requestIdentity else { return }
                    image = result.image
                    compactLabel = result.label
                    compactCount = result.count
                }
            } catch {
                if !Task.isCancelled, identity == requestIdentity { failed = true }
            }
        }
    }
}
