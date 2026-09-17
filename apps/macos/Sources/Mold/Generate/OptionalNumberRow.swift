import MoldStyle
import SwiftUI

/// A number the recipe already has an answer for, which you may override.
///
/// EMPTY is the whole point, and it is not zero: every sampler control is
/// absent from the request until it is moved, because the engine keeps its
/// own constant only while the field is absent. So the field's placeholder is
/// the recipe's own value and clearing the text puts the control back to it,
/// rather than sending a `0` nobody asked for.
struct OptionalNumberRow: View {
    let title: String
    /// What the recipe does when nothing is typed -- shown as the
    /// placeholder, never as a value.
    let placeholder: String
    let value: Binding<Double?>
    /// The sentence this value would be refused with, shown under the field
    /// rather than at submit. `nil` while the value is usable.
    let refusal: String?

    var body: some View {
        LabeledSection(title) {
            VStack(alignment: .leading, spacing: 3) {
                TextField(placeholder, text: text)
                    .textFieldStyle(.roundedBorder)
                    .frame(maxWidth: 140)
                    .accessibilityLabel(title)
                if let refusal {
                    Text(refusal).font(.caption).foregroundStyle(.secondary)
                }
            }
        }
    }

    /// Held as text, not as a formatted number: a formatter rewrites "0." and
    /// "1.2e" under the cursor, and a half-typed value has to survive being
    /// half-typed. The refusal below says what an unusable one means.
    private var text: Binding<String> {
        Binding(
            get: { value.wrappedValue.map { $0.formatted(.number.grouping(.never)) } ?? "" },
            set: { typed in
                let trimmed = typed.trimmingCharacters(in: .whitespaces)
                value.wrappedValue = trimmed.isEmpty ? nil : Double(trimmed)
            })
    }
}

/// The whole-number twin, for `skip_step` -- a `u32` on the wire, where a
/// fractional value fails deserialization outright rather than coming back as
/// a field-named 422 (`guidanceOverrides.ts:108-121`).
struct OptionalStepRow: View {
    let title: String
    let placeholder: String
    let value: Binding<Int?>
    let refusal: String?

    var body: some View {
        OptionalNumberRow(
            title: title, placeholder: placeholder,
            value: Binding(
                get: { value.wrappedValue.map(Double.init) },
                set: { value.wrappedValue = $0.map { Int($0.rounded()) } }),
            refusal: refusal)
    }
}
