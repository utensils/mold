import MoldClient
import SwiftUI

/// Settings ▸ Storage.
///
/// One knob, because there is one thing on this disk worth deciding about: the
/// copies Mold keeps of prints that live on other machines.
struct StorageSettings: View {
    @AppStorage(PrintMaterializer.capKey, store: AppStorageSuite.defaults)
    private var capMegabytes = PrintMaterializer.defaultCapMegabytes
    @Environment(PrintMaterializer.self) private var materializer
    @State private var used = 0

    private static let choices = [256, 512, 1_024, 2_048, 4_096, 8_192]

    var body: some View {
        Form {
            Section {
                Picker("Media cache", selection: $capMegabytes) {
                    ForEach(Self.choices, id: \.self) { megabytes in
                        Text(size(megabytes)).tag(megabytes)
                    }
                    Text("Off").tag(0)
                }
                LabeledContent("Using", value: used > 0 ? bytes(used) : "Nothing")
                Button("Empty Now") {
                    materializer.purge()
                    used = 0
                }
                .disabled(used == 0)
            } footer: {
                Text("""
                     Prints live on the machines that made them. Quick Look, \
                     sharing, saving and dragging one to the Finder all need a \
                     copy on this Mac, and this is how much of that Mold keeps \
                     so a second look costs nothing. It is emptied when you \
                     quit, and nothing in it is the only copy of anything.
                     """)
                .font(.caption)
                .foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
        .task { used = materializer.usedBytes }
        .onChange(of: capMegabytes) { _, _ in
            materializer.enforceBudget()
            used = materializer.usedBytes
        }
    }

    private func size(_ megabytes: Int) -> String {
        bytes(megabytes * 1_024 * 1_024)
    }

    /// Binary, not decimal. The cap is applied in mebibytes, so showing the
    /// 1,024 MB choice as Apple's decimal "1.07 GB" would be reporting a
    /// number nothing in the app uses.
    private func bytes(_ count: Int) -> String {
        count.formatted(.byteCount(style: .memory))
    }
}
