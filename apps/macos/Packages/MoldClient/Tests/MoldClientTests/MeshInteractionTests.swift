import Foundation
import Testing

@testable import MoldClient

/// The gestures, pinned against `studio/components/MeshViewer.vue:869-973`.
///
/// **Fails today**: there was no mesh view on this Mac at all, so nothing said
/// a drag orbits at 0.008 rad/px, that the pitch stops short of the pole, that
/// zoom is bounded, or that there is no pan.
@Suite struct MeshInteractionSuite {
    private let home = MeshViewerCamera.homeCamera()

    @Test func dragsAtTheViewersOwnGain() {
        let dragged = MeshInteraction.orbit(
            home, dx: 50 * MeshInteraction.dragRadiansPerPixel,
            dy: -20 * MeshInteraction.dragRadiansPerPixel)
        #expect(abs(dragged.yaw - (home.yaw + 0.4)) < 1e-12)
        #expect(abs(dragged.pitch - (home.pitch - 0.16)) < 1e-12)
        #expect(dragged.zoom == home.zoom)
    }

    /// A rightward drag lowers the server-frame azimuth, which is the
    /// direction the turntable steps and auto-rotate tours.
    @Test func aRightwardDragTurnsTheObjectTheWayTheTurntableDoes() {
        let dragged = MeshInteraction.orbit(home, dx: 0.4, dy: 0)
        #expect(MeshViewerCamera.azimuthDegOfYaw(dragged.yaw)
            < MeshViewerCamera.azimuthDegOfYaw(home.yaw))
    }

    @Test func clampsThePitchJustShortOfEitherPole() {
        let up = MeshInteraction.orbit(home, dx: 0, dy: -100)
        let down = MeshInteraction.orbit(home, dx: 0, dy: 100)
        #expect(up.pitch == -MeshInteraction.pitchLimit)
        #expect(down.pitch == MeshInteraction.pitchLimit)
        #expect(MeshInteraction.pitchLimit < .pi / 2)
        // The yaw is NOT clamped: it wraps freely all the way round.
        #expect(MeshInteraction.orbit(home, dx: 100, dy: 0).yaw == home.yaw + 100)
    }

    @Test func boundsTheZoomAtBothEnds() {
        var camera = home
        for _ in 0..<100 { camera = MeshInteraction.zoom(camera, by: 1.15) }
        #expect(camera.zoom == MeshInteraction.maximumZoom)
        for _ in 0..<200 { camera = MeshInteraction.zoom(camera, by: 1 / 1.15) }
        #expect(camera.zoom == MeshInteraction.minimumZoom)
    }

    @Test func ignoresANonFiniteOrNonPositiveZoomFactor() {
        #expect(MeshInteraction.zoom(home, by: .nan) == home)
        #expect(MeshInteraction.zoom(home, by: 0) == home)
        #expect(MeshInteraction.zoom(home, by: -2) == home)
    }

    /// A wheel notch is a RATIO, so scrolling down and back up returns to
    /// exactly where it started rather than drifting.
    @Test func readsTheWheelAsARatio() {
        #expect(MeshInteraction.wheelFactor(deltaY: 0) == 1)
        #expect(MeshInteraction.wheelFactor(deltaY: 100) > 1)
        #expect(MeshInteraction.wheelFactor(deltaY: -100) < 1)
        let out = MeshInteraction.zoom(home, by: MeshInteraction.wheelFactor(deltaY: 120))
        let back = MeshInteraction.zoom(out, by: MeshInteraction.wheelFactor(deltaY: -120))
        #expect(abs(back.zoom - home.zoom) < 1e-12)
        #expect(MeshInteraction.wheelFactor(deltaY: .nan) == 1)
    }

    @Test func spreadingTwoFingersZoomsIn() {
        #expect(MeshInteraction.pinchFactor(previous: 100, current: 200) < 1)
        #expect(MeshInteraction.pinchFactor(previous: 200, current: 100) > 1)
        #expect(MeshInteraction.pinchFactor(previous: 0, current: 100) == 1)
        #expect(MeshInteraction.pinchFactor(previous: 100, current: 0) == 1)
    }

    @Test func stepsTheArrowsAndHoldsShiftForABiggerStep() {
        #expect(MeshInteraction.apply(.orbitRight, to: home, shift: false).yaw
            == home.yaw + MeshInteraction.keyRadians)
        #expect(MeshInteraction.apply(.orbitRight, to: home, shift: true).yaw
            == home.yaw + MeshInteraction.shiftKeyRadians)
        #expect(MeshInteraction.apply(.orbitUp, to: home, shift: false).pitch
            == home.pitch - MeshInteraction.keyRadians)
        #expect(MeshInteraction.apply(.orbitDown, to: home, shift: false).pitch
            == home.pitch + MeshInteraction.keyRadians)
    }

    /// `+` comes closer and `-` pulls back, which is the opposite sense to the
    /// `zoom` field they write.
    @Test func plusComesCloserAndMinusPullsBack() {
        #expect(MeshInteraction.apply(.zoomIn, to: home, shift: false).zoom < home.zoom)
        #expect(MeshInteraction.apply(.zoomOut, to: home, shift: false).zoom > home.zoom)
    }

    /// `0` is the poster's camera again, from wherever the view has been
    /// dragged and zoomed to.
    @Test func homeIsThePostersCameraFromAnywhere() {
        let wandered = MeshInteraction.zoom(
            MeshInteraction.orbit(home, dx: 3, dy: 1.2), by: 4)
        #expect(wandered != home)
        #expect(MeshInteraction.apply(.home, to: wandered, shift: false) == home)
    }

    @Test func readsTheZoomAndResetCharacters() {
        #expect(MeshInteraction.key(forCharacters: "+") == .zoomIn)
        #expect(MeshInteraction.key(forCharacters: "=") == .zoomIn)
        #expect(MeshInteraction.key(forCharacters: "-") == .zoomOut)
        #expect(MeshInteraction.key(forCharacters: "_") == .zoomOut)
        #expect(MeshInteraction.key(forCharacters: "0") == .home)
        #expect(MeshInteraction.key(forCharacters: "k") == nil)
        #expect(MeshInteraction.key(forCharacters: "") == nil)
    }

    /// There is NO pan: nothing here moves the centre the camera orbits, so
    /// the framing stays the poster's at every angle.
    @Test func nothingMovesTheCentreTheCameraOrbits() {
        let moved = MeshInteraction.apply(
            .orbitLeft, to: MeshInteraction.zoom(home, by: 2), shift: true)
        // A `ViewerCamera` carries only yaw, pitch and zoom -- there is no
        // field a pan could write, which is the structural half of the rule.
        #expect(moved.zoom == MeshInteraction.zoom(home, by: 2).zoom)
    }
}
