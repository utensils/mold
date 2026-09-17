import MoldClient
import MoldStyle
import SwiftUI

// The Installed table itself: five columns, one `Section` per family, and
// the translation between the pane's persisted `ModelSort` and the native
// `sortOrder` a `Table`'s headers drive. Split from the view for size.
extension ModelsPane {
    var table: some View {
        Table(of: Model.self, selection: $selection, sortOrder: sortOrder) {
            // The bare name: `baseTitle` drops the trailing variant words
            // (`headline` would repeat "Q4" beside the Variant column's own
            // chip -- M5 S3 UAT).
            TableColumn("Model", value: \.sortHeadline) { model in
                Text(model.baseTitle).help(model.name)
            }
            TableColumn("Variant", value: \.sortVariant) { model in
                variantCell(model)
            }
            TableColumn("Trade-off", value: \.sortTradeOff) { model in
                Text(model.tradeOff ?? "—")
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
            TableColumn("Size", value: \.sortSize) { model in
                Text(sizeText(model))
                    .monospacedDigit()
                    .foregroundStyle(.secondary)
            }
            TableColumn("State", value: \.sortState) { model in
                ModelStateCell(model: model, progress: progress(model), install: install, cancel: cancel(model))
                    .frame(maxWidth: .infinity, alignment: .trailing)
            }
        } rows: {
            ForEach(sections, id: \.family) { section in
                Section(section.family) {
                    ForEach(section.rows) { model in
                        TableRow(model).contextMenu { contextMenu(for: model) }
                    }
                }
            }
        }
    }

    /// The same items the Model menu offers for this row -- one door, one
    /// list, both surfaces (design S5).
    @ViewBuilder private func contextMenu(for model: Model) -> some View {
        if let host {
            ForEach(menuItems(for: model)) { item in
                Button(item.title, role: item.role) { actions.perform(item.kind, on: model, host: host) }
            }
        }
    }

    /// The quantization tag as a chip, or an em dash for an untagged model --
    /// what `ModelVariantRow`'s own chip drew, now the Variant column's cell.
    private func variantCell(_ model: Model) -> some View {
        Group {
            if let tag = model.tag {
                Text(tag.uppercased())
                    .font(.caption.weight(.medium))
                    .monospaced()
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(Chrome.wellFill, in: RoundedRectangle(cornerRadius: Chrome.tileRadius))
            } else {
                Text("—").foregroundStyle(.secondary)
            }
        }
    }

    /// What is actually on disk, or failing that what it would cost to
    /// fetch -- the same honest figure `sortSize` orders by, formatted for
    /// reading rather than coerced to zero for comparison.
    private func sizeText(_ model: Model) -> String {
        guard let bytes = model.diskUsageBytes ?? model.remainingDownloadBytes else { return "—" }
        return Int64(bytes).formatted(.byteCount(style: .file))
    }

    /// `Table`'s own header-tap mechanism wants `[KeyPathComparator<Model>]`;
    /// this is the one place that shape exists, translated to and from the
    /// persisted `ModelSort` on every read and every tap. The actual row
    /// order is never asked of a `KeyPathComparator` -- `ModelSort.sorted`
    /// is the one pure authority `sections` reads, and this binding only
    /// tells the header which column and direction is showing.
    private var sortOrder: Binding<[KeyPathComparator<Model>]> {
        Binding(
            get: { [Self.comparator(for: sort.wrappedValue)] },
            set: { comparators in
                guard let first = comparators.first, let column = Self.column(for: first.keyPath) else { return }
                sort.wrappedValue = ModelSort(column: column, ascending: first.order == .forward)
            }
        )
    }

    nonisolated private static func comparator(for sort: ModelSort) -> KeyPathComparator<Model> {
        let order: SortOrder = sort.ascending ? .forward : .reverse
        return switch sort.column {
        case .model: KeyPathComparator(\Model.sortHeadline, order: order)
        case .variant: KeyPathComparator(\Model.sortVariant, order: order)
        case .tradeoff: KeyPathComparator(\Model.sortTradeOff, order: order)
        case .size: KeyPathComparator(\Model.sortSize, order: order)
        case .state: KeyPathComparator(\Model.sortState, order: order)
        }
    }

    private static func column(for keyPath: PartialKeyPath<Model>) -> ModelSort.Column? {
        if keyPath == \Model.sortHeadline { return .model }
        if keyPath == \Model.sortVariant { return .variant }
        if keyPath == \Model.sortTradeOff { return .tradeoff }
        if keyPath == \Model.sortSize { return .size }
        if keyPath == \Model.sortState { return .state }
        return nil
    }
}
