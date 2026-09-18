import AppKit
import Foundation
import MoldClient
import Testing

@testable import Mold

/// The inspector and the Library menu read the ONE derivation.
///
/// **Fails today**: `LibraryPane.swift:119-120` and `LibraryPane+Menu.swift:14`
/// and `:51` each read `showing.selected` directly, so with a print open in the
/// viewer the right-hand column says "Nothing selected" and File ▸ Export…,
/// Favourite and Trash act on whatever the grid still holds rather than on the
/// picture filling the window.
@MainActor
struct LibraryInspectedTests {
    /// A source scan for the same reason `MenuSurfaceTests` is one: the rule
    /// is a single expression, and a fourth surface that reads the selection
    /// straight is a regression nobody sees until a print is open.
    ///
    /// The subtitle is the one honest exception and is named: "3 selected"
    /// describes the GRID, not the print you are looking at.
    @Test func everySurfaceThatFollowsTheOpenPrintReadsTheOneDerivation() throws {
        let library = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent() // Tests/MoldTests
            .deletingLastPathComponent() // Tests
            .deletingLastPathComponent() // apps/macos
            .appending(path: "Sources/Mold/Library")
        let files = try FileManager.default
            .contentsOfDirectory(at: library, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "swift" }
        // A scan that finds no files passes for the wrong reason.
        #expect(files.count > 30, "the Library source directory was not found")

        var offences: [String] = []
        for file in files where file.lastPathComponent != "LibraryPane+Empty.swift" {
            let text = try String(contentsOf: file, encoding: .utf8)
            for (number, line) in text.components(separatedBy: "\n").enumerated() {
                let code = line.trimmingCharacters(in: .whitespaces)
                guard code.contains("showing.selected"), !code.hasPrefix("//") else { continue }
                offences.append("\(file.lastPathComponent):\(number + 1)")
            }
        }
        #expect(offences == [], "a surface reading the grid's selection past the open print")
    }

    /// The label column is a constant, so it is measured rather than trusted:
    /// every label `PrintDetails` can emit has to FIT it at the size it is
    /// drawn, or one row somewhere quietly reads "Identity from ste…".
    @Test func everyDetailLabelFitsTheColumnItIsDrawnIn() throws {
        let font = NSFont.preferredFont(forTextStyle: .caption1)
        let labels = PrintDetails.groups(for: try everything()).flatMap { $0.rows.map(\.label) }
        // A print carrying every field reaches every group; a fixture with
        // the gates switched off would measure three labels and pass.
        #expect(labels.count > 30, "the all-fields print did not reach every group")

        let tooWide = labels.filter {
            NSAttributedString(string: $0, attributes: [.font: font]).size().width
                > InspectorDetails.labelWidth
        }
        #expect(tooWide == [], "a detail label wider than the column it is drawn in")
    }

    /// One print carrying everything the host can record, so the measurement
    /// above sees every label there is.
    ///
    /// Decoded whole, the way `FakeFixtures` builds one: this bundle sees only
    /// MoldClient's public surface and `GalleryPrint`'s memberwise init is not
    /// part of it.
    private func everything() throws -> LibraryEntry {
        let metadata: [String: Any] = [
            "prompt": "a cat", "negative_prompt": "blurry", "original_prompt": "cat",
            "model": "ltx-2.5-22b-distilled:q8", "family": "ltx2", "seed": 1, "steps": 8,
            "guidance": 1.0, "width": 768, "height": 768,
            "generation_width": 384, "generation_height": 384,
            "frames": 145, "fps": 24, "generation_time_ms": 238_374, "job_id": "job",
            "output_format": "mp4", "output_mode": "sequence", "chain_job_id": "chain",
            "chain": ["stage_count": 4, "motion_tail_frames": 8],
            "scheduler": "euler-ancestral", "strength": 0.75, "cfg_plus": true,
            "sample_shift": 5.0, "distill_strength_high": 0.9,
            "distill_strength_low": 0.8, "pipeline": "distilled",
            "extend_overlap_frames": 8,
            "keyframes": [["frame": 0, "sha256": "a"]],
            "source_image_name": "source.png", "source_image_sha256": "abc",
            "id_image_names": ["face.png"], "id_weight": 0.9, "id_start_step": 3,
            "edit_image_sha256s": ["a"],
            "loras": [["path": "/srv/film.safetensors", "scale": 0.8]],
            "control_model": "canny", "control_scale": 0.6,
            "upscale_model": "realesrgan-x4", "enable_audio": true,
            "mesh": ["octree_resolution": 256, "threshold": 0.6,
                     "target_faces": 40_000, "texture": true],
            "mesh_workflow": ["job_id": "w", "mode": "text_to_mesh",
                              "role": "final_glb", "stage_index": 0],
            "version": "0.30.0",
        ]
        let row: [String: Any] = [
            "filename": "everything.mp4", "metadata": metadata, "timestamp": 1_789_529_980,
            "format": "mp4", "size_bytes": 6_947_151, "media_version": "v1",
            "title": "Everything",
        ]
        let print = try MoldJSON.decoder.decode(
            GalleryPrint.self, from: JSONSerialization.data(withJSONObject: row))
        return LibraryEntry(
            host: MoldHost(id: UUID(), name: "plato", baseURL: URL(string: "http://h")!),
            print: print)
    }
}
