import MoldClient
import SwiftUI

// What an adapter row offers on a right-click, rendered from `GenerateMenus`
// so it and the inline controls beside it cannot drift. Split from the
// group's own shape purely for size.
extension AdaptersGroup {
    /// An adapter row's menu holds one thing no other Generate row does: the
    /// adapter's own trained words, which are its VOCABULARY rather than
    /// actions on it -- so the row's kind is either of the two.
    enum Item: Hashable {
        case insert(String)
        case act(GenerateAction)
    }

    /// The row's own words first, then the shared list
    /// (`GenerateMenus.adapterRow`), whose Reset Strength is offered only
    /// where there is something to reset. Grouped explicitly, because a list
    /// that declares one separator declares them all.
    func adapterMenu(_ choice: LoraChoice, words: [String]) -> [RowAction<Item>] {
        let vocabulary = words.map { RowAction(kind: Item.insert($0), title: "Insert \"\($0)\"") }
        let actions = GenerateMenus.adapterRow(isAtDefaultStrength: choice.scale == Lora.defaultScale)
        return vocabulary + [.separator] + RowAction.grouped(actions.map { $0.mapKind(Item.act) })
    }

    func perform(_ item: Item, on choice: LoraChoice) {
        switch item {
        case let .insert(word): insert(word)
        case let .act(action): perform(action, on: choice)
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
