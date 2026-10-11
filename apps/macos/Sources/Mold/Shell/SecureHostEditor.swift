import MoldClient
import SwiftUI

/// Pairing pins the host independently of discovery and the cloud service.
struct SecureHostEditor: View {
    let existing: MoldHost?
    @Environment(HostStore.self) private var hosts
    @Environment(\.dismiss) private var dismiss
    @State private var code = ""
    @State private var name = ""
    @State private var working = false
    @State private var failure: String?
    @State private var attempt = 0

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text(existing == nil ? "Pair a Machine" : "Edit Machine").font(.headline)
            if let existing, existing.secure != nil {
                TextField("Name", text: $name)
                Text("This machine uses an encrypted connection.").font(.callout).foregroundStyle(.secondary)
            } else {
                Text("On the other machine, open Pair a Device and copy its pairing link. Paste the link here to connect securely.")
                    .fixedSize(horizontal: false, vertical: true)
                TextField("Pairing link", text: $code, axis: .vertical)
                    .lineLimit(3...5).textFieldStyle(.roundedBorder).disabled(working)
                if existing != nil {
                    Text("This saved connection needs a new pairing code after the encryption upgrade.").font(.callout)
                }
            }
            if let failure { Text(failure).foregroundStyle(.red).fixedSize(horizontal: false, vertical: true) }
            HStack {
                if working { ProgressView().controlSize(.small) }
                Spacer()
                Button("Cancel") { dismiss() }.keyboardShortcut(.cancelAction)
                Button(existing?.secure != nil ? "Save" : "Pair") {
                    if var host = existing, host.secure != nil {
                        host.name = name; hosts.update(host); dismiss()
                    } else { working = true; failure = nil; attempt += 1 }
                }
                .keyboardShortcut(.defaultAction)
                .disabled(working || (existing?.secure == nil && code.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty))
            }
        }
        .padding(20).frame(width: 460)
        .onAppear { name = existing?.name ?? "" }
        .task(id: attempt) {
            guard attempt > 0 else { return }
            defer { working = false }
            do {
                let payload = try MobilePairingPayload.parse(code)
                _ = try await hosts.pair(payload, replacing: existing?.id)
                guard !Task.isCancelled else { return }
                code = ""; dismiss()
            } catch {
                guard !Task.isCancelled else { return }
                failure = error.failureSentence
            }
        }
    }
}
