// swift-tools-version: 6.0
import PackageDescription

// mold's wire contract and transport. Deliberately has NO dependency on
// SwiftUI or AppKit: it must stay usable from tests, and the layer lint in the
// Makefile fails the build if a UI import appears here.
let package = Package(
    name: "MoldClient",
    platforms: [.macOS("26.0")],
    products: [
        .library(name: "MoldClient", targets: ["MoldClient"])
    ],
    targets: [
        .target(
            name: "MoldClient",
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .testTarget(
            name: "MoldClientTests",
            dependencies: ["MoldClient"],
            swiftSettings: [.swiftLanguageMode(.v6)]
        )
    ]
)
