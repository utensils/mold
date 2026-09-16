// swift-tools-version: 6.0
import PackageDescription

// Visual tokens and low-level chrome. Knows nothing about mold's wire format,
// hosts, or generation -- it is the bottom of the stack and depends on nothing.
let package = Package(
    name: "MoldStyle",
    platforms: [.macOS("26.0")],
    products: [
        .library(name: "MoldStyle", targets: ["MoldStyle"])
    ],
    targets: [
        .target(
            name: "MoldStyle",
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .testTarget(
            name: "MoldStyleTests",
            dependencies: ["MoldStyle"],
            swiftSettings: [.swiftLanguageMode(.v6)]
        )
    ]
)
