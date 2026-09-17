import MoldClient
import SwiftUI

// What an adapter row offers on a right-click, rendered from `GenerateMenus`
// so it and the inline controls beside it cannot drift. Split from the
// group's own shape purely for size.
extension AdaptersGroup {
    /// The adapter row's shared list. Reset Strength is only offered where
    /// there is something to reset (`GenerateMenus.adapterRow`).
    @ViewBuilder func adapterMenu(_ choice: LoraChoice) -> some View {
        let items = GenerateMenus.adapterRow(
            isAtDefaultStrength: choice.scale == Lora.defaultScale)
        ForEach(items.ordinary, id: \.self) { action in
            Button(action.title) { perform(action, on: choice) }
        }
        if !items.destructive.isEmpty {
            Divider()
            ForEach(items.destructive, id: \.self) { action in
                Button(action.title, role: .destructive) { perform(action, on: choice) }
            }
        }
    }

    func perform(_ action: GenerateAction, on choice: LoraChoice) {
        switch action {
        case .resetStrength:
            scaleBinding(for: choice).wrappedValue = Lora.defaultScale
        case .removeAdapter:
            draft.media.loras.removeAll { $0.path == choice.path }
        default:
            break
        }
    }
}
