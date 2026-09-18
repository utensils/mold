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
                // Both halves of the rule, in the one place that owns it.
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.leading, Self.titleInset)
                .padding(.top, 6)
        } label: {
            HStack(spacing: 0) {
                Text(title)
                Spacer(minLength: 8)
                accessory
            }
        }
        .font(.callout)
    }

    /// What a native macOS `DisclosureGroup` leaves between its own leading
    /// edge and the first glyph of its title at `.callout`.
    ///
    /// Measured off the real control rather than guessed --
    /// `uat/lane-n-02-generate-inspector-top-before.png` is the photograph,
    /// where the chevron's leading edge and the title's first glyph are 14
    /// points apart. It has to be a photograph: the disclosure row is drawn
    /// by AppKit and `ImageRenderer` does not capture it, so
    /// `InspectorSectionLayoutTests` can only pin what this file does with
    /// the number -- which is the half that actually regressed.
    static var titleInset: CGFloat { 14 }
}

extension InspectorSection where Accessory == EmptyView {
    init(_ title: String, isExpanded: Binding<Bool>, @ViewBuilder content: () -> Content) {
        self.init(title, isExpanded: isExpanded, accessory: { EmptyView() }, content: content)
    }
}
