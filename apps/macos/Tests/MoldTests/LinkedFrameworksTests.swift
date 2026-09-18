import Foundation
import Testing

/// `VideoPlayer` is SwiftUI's wrapper over AVKit's `AVPlayerView`, and
/// `import AVKit` autolinks only the `_AVKit_SwiftUI` overlay and
/// AVFoundation -- AVKit itself was never in the binary's load commands. So
/// the first clip opened anywhere, in the Library viewer or on the Generate
/// canvas, aborted the app in the Swift runtime ("failed to demangle
/// superclass of VideoPlayerView from mangled name So12AVPlayerViewC"). The
/// test host IS the app, so what dyld loaded for it is what a person gets.
///
/// **Fails today**: `AVPlayerView` is not a class this process knows.
struct LinkedFrameworksTests {
    @Test func avKitIsLinkedSoVideoPlayerCanResolveItsSuperclass() {
        #expect(NSClassFromString("AVPlayerView") != nil)
    }
}
