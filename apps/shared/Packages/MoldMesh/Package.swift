// swift-tools-version: 6.0
import PackageDescription

// The mesh viewer's Metal half: parsing a GLB into GPU buffers, the camera,
// the draw, and the shaders -- shared by the Mac app's NSView and the iPhone
// app's UIView, so a mesh turns and shades the same on both. The views that
// take the events stay in each app; nothing here imports AppKit or UIKit.
let package = Package(
    name: "MoldMesh",
    platforms: [.macOS("26.0"), .iOS("26.0")],
    products: [
        .library(name: "MoldMesh", targets: ["MoldMesh"])
    ],
    dependencies: [
        .package(path: "../MoldClient")
    ],
    targets: [
        .target(
            name: "MoldMesh",
            dependencies: [.product(name: "MoldClient", package: "MoldClient")],
            // Copied as SOURCE and compiled at runtime: `swift build` never
            // compiles .metal into a default library (only Xcode does), and the
            // same package must draw under both.
            resources: [.copy("MeshShaders.metal")],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .testTarget(
            name: "MoldMeshTests",
            dependencies: ["MoldMesh"],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
    ]
)
