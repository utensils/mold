import MoldClient
import SwiftUI

/// The same explicit connection choice from Settings and a machine's menu.
struct PhonePairingSheet: View {
    let host: MoldHost
    var body: some View { PairingSheet(host: host) }
}
