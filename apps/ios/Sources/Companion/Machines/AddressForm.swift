import MoldClient
import SwiftUI

/// Enter an Address: checked live while it is typed (a half-typed name never
/// touches the list), with an optional key that goes only to the Keychain.
struct AddressForm: View {
    @Environment(HostStore.self) private var hosts
    let initialName: String
    let initialAddress: String
    let done: () -> Void

    @State private var name = ""
    @State private var address = ""
    @State private var apiKey = ""
    @State private var makeDefault = false
    @State private var check: HostStore.Reachability = .unknown
    @State private var problem: String?

    var body: some View {
        Form {
            Section {
                TextField("Name", text: $name, prompt: Text(suggestedName))
                    .accessibilityIdentifier("machine-name")
                TextField("Address", text: $address, prompt: Text(verbatim: "workstation.local"))
                    .accessibilityIdentifier("machine-address")
                    .keyboardType(.URL)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
                    .font(.body.monospaced())
                    .textContentType(.URL)
                SecureField("API key (optional)", text: $apiKey)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
                Toggle("Make this the Default machine", isOn: $makeDefault)
            } footer: {
                VStack(alignment: .leading, spacing: 6) {
                    Text("A name, an IP address, or a Tailscale address. Port 7680 is assumed.")
                    if let sentence = check.sentence {
                        Label(sentence, systemImage: "network")
                    }
                }
                .foregroundStyle(.secondaryText)
            }
            if let problem {
                Section { Label(problem, systemImage: "exclamationmark.triangle").foregroundStyle(.red) }
            }
        }
        .navigationTitle("Enter an Address")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .confirmationAction) {
                Button("Add") { add() }.disabled(HostAddress.normalize(address) == nil)
            }
        }
        .onAppear {
            if name.isEmpty { name = initialName }
            if address.isEmpty { address = initialAddress }
            makeDefault = hosts.hosts.isEmpty
        }
        .task(id: address + "\u{0}" + apiKey) { await recheck() }
    }

    private var suggestedName: String {
        HostAddress.normalize(address).map(HostAddress.suggestedName(for:)) ?? String(localized: "workstation")
    }

    /// Debounced: a keystroke cancels the previous check.
    private func recheck() async {
        problem = nil
        guard let url = HostAddress.normalize(address) else { check = .unknown; return }
        try? await Task.sleep(for: .milliseconds(450))
        guard !Task.isCancelled else { return }
        check = .checking
        let answer = await hosts.probe(url: url, apiKey: apiKey.isEmpty ? nil : apiKey)
        if !Task.isCancelled { check = answer }
    }

    private func add() {
        do {
            try hosts.add(name: name, address: address, apiKey: apiKey, makeDefault: makeDefault)
            done()
        } catch {
            problem = error.errorDescription
        }
    }
}
