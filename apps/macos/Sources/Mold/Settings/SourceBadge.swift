import MoldClient
import SwiftUI

/// `source` as a bordered caption -- `file` / `db` / `env` / `default`, the
/// SURFACE a row's write would land on or refuse against, never where the
/// value historically came from (`ConfigEntry+Editing.swift`). Semantic
/// styles only: `make lint-color` fails on a literal.
struct SourceBadge: View {
    let entry: ConfigEntry

    var body: some View {
        HStack(spacing: 6) {
            Text(label)
                .font(.caption2.weight(.medium))
                .foregroundStyle(.secondary)
                .padding(.horizontal, 6)
                .padding(.vertical, 2)
                .overlay(Capsule().strokeBorder(.secondary.opacity(0.3)))
            if entry.needsRestart {
                Text("Needs a restart")
                    .font(.caption2)
                    .foregroundStyle(.orange)
            }
        }
        .help(sourceExplanation)
        .accessibilityElement(children: .combine)
        .accessibilityValue(entry.source)
    }

    private var sourceExplanation: String {
        switch entry.source {
        case "db": "Changes to this setting are saved in this machine’s settings."
        case "file": "Changes to this setting are saved in this machine’s configuration file."
        case "env": "This value is set when the machine starts and cannot be changed here."
        default: "This machine is using the default value."
        }
    }

    private var label: String {
        switch entry.source {
        case "db": "Database"
        case "file": "File"
        case "env": "Environment"
        default: "Default"
        }
    }
}
