// swift-tools-version: 6.0
import PackageDescription

// mold's wire contract and transport. Deliberately has NO dependency on
// SwiftUI or AppKit: it must stay usable from tests, and the layer lint in the
// Makefile fails the build if a UI import appears here.
let package = Package(
    name: "MoldClient",
    platforms: [.macOS("26.0"), .iOS("26.0")],
    products: [
        .library(name: "MoldClient", targets: ["MoldClient"]),
        // Test doubles for both apps' store tests. Test targets only: nothing
        // in it belongs in a shipping binary.
        .library(name: "MoldClientTesting", targets: ["MoldClientTesting"]),
    ],
    targets: [
        .binaryTarget(name: "MoldSecure", path: "build/MoldSecure.xcframework"),
        .target(
            name: "MoldClient",
            dependencies: ["MoldSecure"],
            swiftSettings: [.swiftLanguageMode(.v6)],
            linkerSettings: [.linkedLibrary("iconv")]
        ),
        .target(
            name: "MoldClientTesting",
            dependencies: ["MoldClient"],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .testTarget(
            name: "MoldClientTests",
            dependencies: ["MoldClient", "MoldClientTesting"],
            // Read by path off #filePath, not from a bundle, so SwiftPM should
            // leave them alone rather than treat them as unhandled resources.
            exclude: ["Fixtures"],
            swiftSettings: [.swiftLanguageMode(.v6)]
        )
    ]
)
