import SwiftUI

/// One collapsible group of the Generate inspector, and the one rule its
/// contents line up by.
///
/// **A section's content starts where its TITLE does**, never where its
/// chevron does: the chevron keeps a gutter of its own and everything under
/// the title -- a row, a slider, a button, the sentence a section shows when
/// it has nothing to offer -- leads from that one edge.
///
/// Both halves of that used to be wrong. A bare `DisclosureGroup` puts its
/// content back at the chevron's edge, a step left of the title it belongs to;
/// and the leading `VStack` each group draws never said it was the flexible
/// side, so a group whose widest child was narrow shrank to that child and
/// SwiftUI centred the lot -- "No adapters installed for this model." floated
/// in the middle of the column while every other line led (the owner's
/// screenshot, 2026-09-17). `InspectorSectionLayoutTests` measures both by
/// drawing the section and reading the pixels, so neither can drift back.
///
/// The accessory is for a section whose title carries a control of its own,
/// the way Recent carries its Refresh -- the title stays a title, and the
/// control stays out of the content's leading edge.
struct InspectorSection<Content: View, Accessory: View>: View {
    private let title: String
    @Binding private var isExpanded: Bool
    private let accessory: Accessory
    private let content: Content

    init(_ title: String, isExpanded: Binding<Bool>,
         @ViewBuilder accessory: () -> Accessory,
         @ViewBuilder content: () -> Content) {
        self.title = title
        _isExpanded = isExpanded
        self.accessory = accessory()
        self.content = content()
    }

    var body: some View {
        DisclosureGroup(isExpanded: $isExpanded) {
            content
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.leading, Self.titleInset)
                .padding(.top, 6)
        } label: {
            Text(title)
        }
        .disclosureGroupStyle(InspectorDisclosureStyle(title: title, accessory: accessory))
        .font(.callout)
    }

    /// Keep the heading and content aligned after the chevron gutter.
    static var titleInset: CGFloat { 14 }
}

extension InspectorSection where Accessory == EmptyView {
    init(_ title: String, isExpanded: Binding<Bool>, @ViewBuilder content: () -> Content) {
        self.init(title, isExpanded: isExpanded, accessory: { EmptyView() }, content: content)
    }
}

/// Give the whole heading one explicit action. The default macOS disclosure
/// label is not a reliable pointer target, and nesting Refresh inside it
/// also makes the accessory part of the disclosure's accessibility label.
private struct InspectorDisclosureStyle<Accessory: View>: DisclosureGroupStyle {
    let title: String
    let accessory: Accessory

    func makeBody(configuration: Configuration) -> some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 8) {
                Button {
                    configuration.isExpanded.toggle()
                } label: {
                    HStack(spacing: 0) {
                        Image(systemName: configuration.isExpanded ? "chevron.down" : "chevron.right")
                            .font(.caption.weight(.semibold))
                            .frame(width: InspectorSection<EmptyView, EmptyView>.titleInset, alignment: .leading)
                            .accessibilityHidden(true)
                        configuration.label
                            .fontWeight(.semibold)
                            .fixedSize(horizontal: false, vertical: true)
                        Spacer(minLength: 0)
                    }
                    .frame(minHeight: 30)
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .accessibilityLabel(title)
                .help(configuration.isExpanded ? "Hide \(title) settings" : "Show \(title) settings")
                .accessibilityValue(configuration.isExpanded ? "Expanded" : "Collapsed")
                accessory
            }
            if configuration.isExpanded { configuration.content.padding(.bottom, 10) }
            Divider().padding(.top, 6)
        }
    }
}
