import SwiftUI
import UIKit

/// Apply the preference at the window so root content and presented sheets
/// share one override. System explicitly clears it: a SwiftUI sheet can retain
/// its previous explicit scheme after preferredColorScheme becomes nil.
struct AppearanceWindow: UIViewRepresentable {
    let appearance: AppAppearance

    func makeUIView(context: Context) -> Probe {
        let view = Probe(frame: .zero)
        view.isUserInteractionEnabled = false
        view.accessibilityElementsHidden = true
        view.appearance = appearance
        return view
    }

    func updateUIView(_ view: Probe, context: Context) { view.appearance = appearance }

    final class Probe: UIView {
        var appearance = AppAppearance.system { didSet { apply() } }

        override func didMoveToWindow() {
            super.didMoveToWindow()
            apply()
        }

        private func apply() {
            guard let window, window.overrideUserInterfaceStyle != appearance.interfaceStyle else { return }
            window.overrideUserInterfaceStyle = appearance.interfaceStyle
        }
    }
}
