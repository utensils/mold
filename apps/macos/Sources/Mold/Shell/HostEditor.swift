import MoldClient
import SwiftUI

/// Add or edit one machine.
struct HostEditor: View {
    let existing: MoldHost?
    let save: (String, URL, String?) -> Void

    @Environment(\.dismiss) private var dismiss
    @State private var name: String
    @State private var address: String
    @State private var apiKey: String

    init(host: MoldHost? = nil, save: @escaping (String, URL, String?) -> Void) {
        self.existing = host
        self.save = save
        _name = State(initialValue: host?.name ?? "")
        _address = State(initialValue: host?.baseURL.absoluteString ?? "http://")
        _apiKey = State(initialValue: host?.apiKey ?? "")
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Form {
                TextField("Name", text: $name, prompt: Text("plato"))
                TextField("Address", text: $address, prompt: Text("http://10.0.0.5:7680"))
                    .autocorrectionDisabled()
                SecureField("API key", text: $apiKey, prompt: Text("Leave empty if not required"))
            }
            .formStyle(.grouped)

            Text(keyGuidance)
                .font(.caption)
                .foregroundStyle(.secondary)
                .padding(.horizontal, 20)
                .fixedSize(horizontal: false, vertical: true)

            Divider().padding(.top, 12)
            HStack {
                Spacer()
                Button("Cancel") { dismiss() }.keyboardShortcut(.cancelAction)
                Button(existing == nil ? "Add" : "Save") { commit() }
                    .keyboardShortcut(.defaultAction)
                    .disabled(url == nil || name.trimmingCharacters(in: .whitespaces).isEmpty)
            }
            .padding(12)
        }
        .frame(width: 420)
    }

    /// A keyless host is a normal configuration, not a mistake -- a mold
    /// server with no MOLD_API_KEY leaves every route open by policy.
    private var keyGuidance: String {
        "Only needed if the server sets MOLD_API_KEY. Keys are stored in your keychain."
    }

    private var url: URL? {
        let trimmed = address.trimmingCharacters(in: .whitespaces)
        guard let url = URL(string: trimmed), url.scheme != nil, url.host() != nil else {
            return nil
        }
        return url
    }

    private func commit() {
        guard let url else { return }
        save(name.trimmingCharacters(in: .whitespaces), url,
             apiKey.isEmpty ? nil : apiKey)
        dismiss()
    }
}

/// One machine in Settings, with what it last said.
struct HostSettingsRow: View {
    let host: MoldHost
    let reachability: HostStore.Reachability

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: "circle.fill").font(.system(size: 7)).foregroundStyle(tint)
            VStack(alignment: .leading, spacing: 1) {
                Text(host.name)
                Text(host.baseURL.absoluteString)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            if host.apiKey != nil {
                Image(systemName: "key.fill")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
                    .help("Using a stored API key")
            }
            Text(detail).font(.caption).foregroundStyle(.secondary).lineLimit(1)
        }
    }

    private var tint: Color {
        switch reachability {
        case .unknown, .checking: .secondary
        case .up: .green
        case .needsKey: .orange
        case .down: .red
        }
    }

    private var detail: String {
        switch reachability {
        case .unknown: ""
        case .checking: "Checking…"
        case let .up(status): status.version
        case .needsKey: "Needs an API key"
        case .down: "Unreachable"
        }
    }
}
