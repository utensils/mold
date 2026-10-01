import SwiftUI

/// The app's logo also identifies Live Activities, whose extension has its own resources.
struct ActivityBrandIcon: View {
    var body: some View {
        Image("MoldLogo")
            .resizable()
            .scaledToFit()
            .clipShape(.rect(cornerRadius: 4))
    }
}
