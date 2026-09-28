import MoldClient
import SwiftUI

/// Edit…: rename, move, or change the key. A key left untouched stays as it
/// is; cleared, it is removed from the Keychain.
struct EditMachineSheet: View {
    @Environment(HostStore.self) private var hosts
    @Environment(\.dismiss) private var dismiss
    let host: MoldHost

    @State private var name = ""
    @State private var address = ""
    @State private var apiKey = ""
    @State private var keyTouched = false
    @State private var problem: String?

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    TextField("Name", text: $name)
                    TextField("Address", text: $address)
                        .keyboardType(.URL)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                        .font(.body.monospaced())
                }
                Section {
                    SecureField(host.apiKey == nil ? "API key (optional)" : "API key (saved)", text: $apiKey)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                        .onChange(of: apiKey) { keyTouched = true }
                    if host.apiKey != nil {
                        Button("Remove Key", role: .destructive) { apiKey = ""; keyTouched = true; save() }
                    }
                } footer: {
                    Text("Keys are kept only in this iPhone's Keychain.").foregroundStyle(.secondaryText)
                }
                if let problem {
                    Section { Label(problem, systemImage: "exclamationmark.triangle").foregroundStyle(.red) }
                }
            }
            .navigationTitle("Edit \(host.name)")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Cancel") { dismiss() } }
                ToolbarItem(placement: .confirmationAction) { Button("Save") { save() } }
            }
            .onAppear {
                name = host.name
                address = HostAddress.displayString(for: host.baseURL)
            }
        }
    }

    private func save() {
        do {
            try hosts.update(host.id, name: name, address: address, apiKey: keyTouched ? apiKey : nil)
            dismiss()
        } catch {
            problem = error.errorDescription
        }
    }
}
