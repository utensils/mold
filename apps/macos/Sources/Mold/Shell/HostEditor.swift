import MoldClient
import SwiftUI

/// Add or edit one machine.
///
/// The address is the only thing anyone actually knows, so it comes first and
/// everything else follows from it: the name is filled in from the address and
/// then replaced by the hostname the machine reports, and the machine is
/// checked while the sheet is open so nobody clicks Add and then wonders.
struct HostEditor: View {
    let existing: MoldHost?
    let save: (String, URL, String?) -> Void

    // Internal, not private: `HostEditor+Address` reads these.
    @Environment(HostStore.self) var hosts
    @Environment(\.dismiss) private var dismiss

    @State var address: String
    @State private var name: String
    @State private var apiKey: String
    /// The name we filled in ourselves. A name the person typed is never
    /// overwritten, and comparing against this is how we tell the two apart
    /// without tracking which field had focus.
    @State private var autoName: String?
    @State private var probe: HostStore.Reachability = .unknown
    @State private var attempt = 0

    init(host: MoldHost? = nil, save: @escaping (String, URL, String?) -> Void) {
        self.existing = host
        self.save = save
        _address = State(initialValue: host.map { HostAddress.displayString(for: $0.baseURL) } ?? "")
        _name = State(initialValue: host?.name ?? "")
        _apiKey = State(initialValue: host?.apiKey ?? "")
    }

    /// A machine we found but have never spoken to. It is an ADD -- `existing`
    /// stays nil so the sheet says so -- with the address and name already
    /// filled.
    init(adding name: String, at address: String, save: @escaping (String, URL, String?) -> Void) {
        self.existing = nil
        self.save = save
        _address = State(initialValue: address)
        _name = State(initialValue: name)
        _apiKey = State(initialValue: "")
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Text(existing == nil ? "Add a Machine" : "Edit Machine")
                .font(.headline)
                .padding(.horizontal, 20)
                .padding(.top, 20)

            // Columns, not `.grouped`: this is a dialog, so the labels belong
            // in a right-aligned column beside their fields the way every
            // other Mac dialog does it.
            Form {
                LabeledContent("Address:") {
                    VStack(alignment: .leading, spacing: 4) {
                        TextField("Address", text: $address, prompt: Text(verbatim: "plato"))
                            .labelsHidden()
                            .autocorrectionDisabled()
                            .onSubmit(commit)
                        if let hint { hint.font(.caption) }
                    }
                }
                LabeledContent("Name:") {
                    TextField("Name", text: $name, prompt: Text(verbatim: suggestedName))
                        .labelsHidden()
                        .onSubmit(commit)
                }
                LabeledContent("API key:") {
                    VStack(alignment: .leading, spacing: 4) {
                        SecureField("API key", text: $apiKey,
                                    prompt: Text("Leave empty if not required"))
                            .labelsHidden()
                            .onSubmit(commit)
                        Text(verbatim: "A name or IP is enough — Mold fills in http:// and "
                             + "port 7680 unless you say otherwise. A key is only needed if "
                             + "the server sets MOLD_API_KEY; it is kept in your keychain.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
            }
            .formStyle(.columns)
            .textFieldStyle(.roundedBorder)
            .padding(20)

            Divider()
            HStack(spacing: 10) {
                HostProbeSummary(reachability: probe,
                                 recheck: resolved == nil ? nil : { attempt += 1 })
                Spacer(minLength: 12)
                Button("Cancel") { dismiss() }.keyboardShortcut(.cancelAction)
                Button(existing == nil ? "Add" : "Save") { commit() }
                    .keyboardShortcut(.defaultAction)
                    .disabled(resolved == nil || duplicate != nil)
            }
            .padding(12)
        }
        .frame(width: 480)
        .onChange(of: address) { _, _ in adoptSuggestedName() }
        .task(id: probeKey) { await runProbe() }
    }

    // MARK: - Naming and checking

    private func adoptSuggestedName() {
        guard name.isEmpty || name == autoName, let resolved else { return }
        let candidate = HostAddress.suggestedName(for: resolved)
        name = candidate
        autoName = candidate
    }

    /// Changing any of these means the last answer is about a different
    /// machine, so `.task(id:)` cancels the old check and starts a new one.
    private var probeKey: String {
        "\(attempt)\u{1}\(resolved?.absoluteString ?? "")\u{1}\(apiKey)"
    }

    private func runProbe() async {
        guard let resolved else { probe = .unknown; return }
        // Typing an address goes through a dozen invalid prefixes on the way
        // to a real one. Waiting for a pause means one request, not twelve.
        try? await Task.sleep(for: .milliseconds(500))
        guard !Task.isCancelled else { return }
        probe = .checking
        let outcome = await hosts.probe(url: resolved, apiKey: apiKey.isEmpty ? nil : apiKey)
        guard !Task.isCancelled else { return }
        probe = outcome
        // The machine's own hostname beats whatever we guessed from the
        // address -- it is the same answer however you reached the box.
        if case let .up(status) = outcome, let hostname = status.hostname,
           name.isEmpty || name == autoName {
            name = hostname
            autoName = hostname
        }
    }

    private func commit() {
        guard let resolved, duplicate == nil else { return }
        save(name.trimmingCharacters(in: .whitespacesAndNewlines), resolved,
             apiKey.isEmpty ? nil : apiKey)
        dismiss()
    }
}
