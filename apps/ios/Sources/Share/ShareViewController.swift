import SwiftUI
import UIKit

/// Share ▸ Mold Studio. From M12 it stages the shared photo in the App Group
/// inbox (DESIGN.md §5.9) and never touches the network; until then it says
/// plainly that it cannot take a picture yet.
final class ShareViewController: UIViewController {
    override func viewDidLoad() {
        super.viewDidLoad()
        let host = UIHostingController(rootView: ShareSheet { [weak self] in
            self?.extensionContext?.completeRequest(returningItems: nil)
        })
        addChild(host)
        host.view.frame = view.bounds
        host.view.autoresizingMask = [.flexibleWidth, .flexibleHeight]
        view.addSubview(host.view)
        host.didMove(toParent: self)
    }
}

struct ShareSheet: View {
    let done: () -> Void

    var body: some View {
        NavigationStack {
            ContentUnavailableView(
                "Not yet", systemImage: "square.and.arrow.down",
                description: Text("Mold Studio can't take pictures from Share yet."))
                .navigationTitle("Mold Studio")
                .navigationBarTitleDisplayMode(.inline)
                .toolbar {
                    ToolbarItem(placement: .confirmationAction) {
                        Button("Done", action: done)
                    }
                }
        }
    }
}
