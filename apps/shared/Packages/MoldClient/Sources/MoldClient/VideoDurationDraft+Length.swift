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
                  total > UInt64(grid.maxFrames) * 1_000 / UInt64(grid.fps),
                  let count = Self.sceneCount(milliseconds: total, limit: limit),
                  count <= UInt64(Self.maxScenes) else { return false }
            scenes = Self.split(prompt: prompt, milliseconds: total, count: count)
            scenesEnabled = true
            updateDurationFromScenes()
            return true
        }
        // Count first, allocate second: past the request's scene limit the
        // saved scenes are kept exactly as they were.
        var counts: [UInt64] = []
        var total: UInt64 = 0
        for scene in scenes {
            var count: UInt64 = 1
            if let value = try? Self.milliseconds(scene.seconds), value > limit {
                guard let split = Self.sceneCount(milliseconds: value, limit: limit) else { return false }
                count = split
            }
            let (next, overflow) = total.addingReportingOverflow(count)
            guard !overflow, next <= UInt64(Self.maxScenes) else { return false }
            total = next
            counts.append(count)
        }
        guard total > UInt64(scenes.count) else { return false }
        var fitted: [Scene] = []
        for (scene, count) in zip(scenes, counts) {
            if count > 1, let value = try? Self.milliseconds(scene.seconds) {
                fitted += Self.split(prompt: scene.prompt, milliseconds: value, count: count)
            } else {
                fitted.append(scene)
            }
        }
        scenes = fitted
        updateDurationFromScenes()
        return true
    }

    /// The most scenes one request may carry (`request` refuses more).
    static let maxScenes = 4_096

    /// A single prompt longer than one generation that restoring could not
    /// turn into scenes: shown read-only, still submitted as it is.
    func exceedsOneGeneration(on grid: VideoLengthGrid) -> Bool {
        guard !isLonger, let total = lengthMilliseconds else { return false }
        return total > UInt64(grid.maxFrames) * 1_000 / UInt64(grid.fps)
    }

    /// `ceil(total / limit)` without the `total + limit - 1` overflow; nil for
    /// a zero limit.
    static func sceneCount(milliseconds total: UInt64, limit: UInt64) -> UInt64? {
        guard limit > 0 else { return nil }
        return max(total / limit + (total % limit == 0 ? 0 : 1), 1)
    }

    private static func split(prompt: String, milliseconds total: UInt64, count: UInt64) -> [Scene] {
        let part = total / count
        return (0..<count).map { index in
            let value = index == count - 1 ? total - part * (count - 1) : part
            return Scene(prompt: prompt, seconds: seconds(milliseconds: value))
        }
    }
}
