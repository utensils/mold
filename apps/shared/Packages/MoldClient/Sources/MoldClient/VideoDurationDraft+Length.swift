import Foundation

/// What the native Length slider, its Longer video toggle and the scene
/// sliders do to a Duration draft. Lengths stay exact decimal strings.
public extension VideoDurationDraft {
    /// Longer video: the scenes editor, each scene at most one generation.
    var isLonger: Bool { scenesEnabled && !scenes.isEmpty }

    /// The exact requested length, shown as it is: a restored "14" reads 14.0 s.
    var lengthMilliseconds: UInt64? { try? Self.milliseconds(seconds) }

    /// The single-prompt slider writes whole tenths of a second.
    mutating func setLength(tenths: Int, on grid: VideoLengthGrid) {
        seconds = VideoLengthGrid.seconds(tenths: min(max(tenths, grid.singleMinTenths), grid.singleMaxTenths))
    }

    mutating func setLonger(_ on: Bool, prompt: String, on grid: VideoLengthGrid) {
        if on {
            if scenes.isEmpty {
                let tenths = grid.clampSceneTenths(VideoLengthGrid.tenths(seconds: seconds) ?? grid.sceneMinTenths)
                scenes = [.init(prompt: prompt, seconds: VideoLengthGrid.seconds(tenths: tenths))]
            }
            scenesEnabled = true
            updateDurationFromScenes()
        } else {
            scenesEnabled = false
            if let total = lengthMilliseconds, total > UInt64(grid.singleMaxTenths) * 100 {
                setLength(tenths: grid.singleMaxTenths, on: grid)
            }
        }
    }

    mutating func addScene(on grid: VideoLengthGrid) {
        scenes.append(.init(prompt: "", seconds: VideoLengthGrid.seconds(tenths: grid.clampSceneTenths(50))))
        updateDurationFromScenes()
    }

    func sceneTenths(_ id: Scene.ID, on grid: VideoLengthGrid) -> Int {
        grid.clampSceneTenths(scenes.first { $0.id == id }.flatMap { VideoLengthGrid.tenths(seconds: $0.seconds) } ?? grid.sceneMinTenths)
    }

    mutating func setScene(_ id: Scene.ID, tenths: Int, on grid: VideoLengthGrid) {
        guard let index = scenes.firstIndex(where: { $0.id == id }) else { return }
        scenes[index].seconds = VideoLengthGrid.seconds(tenths: grid.clampSceneTenths(tenths))
        updateDurationFromScenes()
    }

    /// A saved length the sliders cannot show is never silently reinterpreted:
    /// a single prompt longer than one generation, or a scene longer than the
    /// always-one-clip limit, becomes equal scenes of the same prompt that add
    /// up to the same length. Returns whether anything changed.
    @discardableResult
    mutating func fit(to grid: VideoLengthGrid, prompt: String) -> Bool {
        let limit = UInt64(grid.sceneMaxTenths) * 100
        if !isLonger {
            guard let total = try? Self.milliseconds(seconds),
                  total > UInt64(grid.maxFrames) * 1_000 / UInt64(grid.fps) else { return false }
            scenes = Self.split(prompt: prompt, milliseconds: total, limit: limit)
            scenesEnabled = true
            updateDurationFromScenes()
            return true
        }
        var changed = false
        var fitted: [Scene] = []
        for scene in scenes {
            if let value = try? Self.milliseconds(scene.seconds), value > limit {
                fitted += Self.split(prompt: scene.prompt, milliseconds: value, limit: limit)
                changed = true
            } else {
                fitted.append(scene)
            }
        }
        guard changed else { return false }
        scenes = fitted
        updateDurationFromScenes()
        return true
    }

    private static func split(prompt: String, milliseconds total: UInt64, limit: UInt64) -> [Scene] {
        let count = max((total + limit - 1) / max(limit, 1), 1)
        let part = total / count
        return (0..<count).map { index in
            let value = index == count - 1 ? total - part * (count - 1) : part
            return Scene(prompt: prompt, seconds: seconds(milliseconds: value))
        }
    }
}
