import MoldClient
import SwiftUI

/// Aspect + size, drawn as two menus under the prompt instead of one buried
/// "1024 × 1024" menu (M8 design, decision 3).
///
/// The pure resolution behind this -- `Shape`, `Presentation`, `resolve`,
/// `size(in:)` -- lives in `ShapeControl+Resolve.swift`, which extends this
/// struct rather than declaring its own type.
struct ShapeControl: View {
    let resolution: ResolutionProfile
    @Binding var draft: RenderDraft

    var body: some View {
        switch ShapeControl.resolve(resolution: resolution, width: draft.width, height: draft.height) {
        case let .menus(shape):
            HStack(spacing: 6) {
                aspectMenu(shape)
                sizeMenu(shape)
            }
            .help("The picture's aspect and size")
        case let .fixed(text):
            // Verbatim: pixel dimensions take no thousands separator.
            Text(verbatim: text).monospacedDigit()
        case .fromSource:
            Text("From the source").foregroundStyle(.secondary)
        case .hidden:
            // `ControlsRow` only places this view when `resolve` is NOT
            // `.hidden` -- reaching here would be that gate's own bug.
            EmptyView() // a11y: no glyph, nothing to opt out of
        }
    }

    private func aspectMenu(_ shape: Shape) -> some View {
        Menu {
            ForEach(shape.aspects) { group in
                Button {
                    choose(group)
                } label: {
                    if group.id == shape.aspect {
                        Label(group.label, systemImage: "checkmark")
                    } else {
                        Text(group.label)
                    }
                }
            }
        } label: {
            Text(shape.aspect)
        }
        .menuStyle(.button)
        .buttonStyle(.accessoryBar)
        .fixedSize()
    }

    private func sizeMenu(_ shape: Shape) -> some View {
        Menu {
            ForEach(shape.sizes) { preset in
                Button {
                    chooseSize(preset)
                } label: {
                    if preset.width == draft.width, preset.height == draft.height {
                        Label(preset.label, systemImage: "checkmark")
                    } else {
                        Text(preset.label)
                    }
                }
            }
        } label: {
            Text(verbatim: "\(draft.width) × \(draft.height)").monospacedDigit()
        }
        .menuStyle(.button)
        .buttonStyle(.accessoryBar)
        .fixedSize()
    }

    /// Choosing a canvas RECORDS that somebody chose it (#1166): the intent
    /// is written when the act happens, never inferred from the size
    /// afterwards, so an attached source stops moving this canvas from here on.
    private func chooseSize(_ preset: SizePreset) {
        draft.width = preset.width
        draft.height = preset.height
        draft.canvasIntent = .manual
    }

    /// Moves to the preset in `group` nearest the current pixel count --
    /// picking 16:9 from a 1024x1024 canvas lands on 1024x576, not on the
    /// group's smallest.
    private func choose(_ group: AspectGroup) {
        guard let preset = ShapeControl.size(in: group, nearWidth: draft.width, height: draft.height) else { return }
        chooseSize(preset)
    }
}
