import SwiftUI

extension LibraryPane {
    func bulkStatusRow<Controls: View>(_ message: String,
                                      @ViewBuilder controls: () -> Controls) -> some View {
        HStack {
            ProgressView().controlSize(.small)
            Text(message).accessibilityAddTraits(.updatesFrequently)
            Spacer()
            controls()
        }
        .padding(12)
        .background(.bar)
        .accessibilityElement(children: .contain)
    }
}
