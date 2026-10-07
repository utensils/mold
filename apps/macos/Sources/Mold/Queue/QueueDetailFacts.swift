import MoldClient
import SwiftUI

/// Prose gets the full column; short settings share a label/value row. The
/// Prompt section already names its first paragraph, so never repeat it.
struct QueueDetailFacts: View {
    let group: PrintDetailGroup

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Divider()
            Text(group.title).font(.headline)
            ForEach(group.rows, id: \.label) { row in
                if row.isProse {
                    VStack(alignment: .leading, spacing: 4) {
                        if QueueDetailPresentation.showsLabel(row, in: group) {
                            Text(row.label).font(.caption).foregroundStyle(.secondary)
                        }
                        Text(row.value).textSelection(.enabled)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                } else {
                    HStack(alignment: .firstTextBaseline, spacing: 12) {
                        Text(row.label).foregroundStyle(.secondary)
                            .frame(width: 110, alignment: .leading)
                        Text(row.value).textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    .font(.callout)
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}
