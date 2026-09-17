import Foundation
import Testing

@testable import MoldClient

/// Ported from `studio/lib/meshViewerCamera.test.ts`.
///
/// **Fails today**: this app had no mesh camera, so nothing said that its home
/// view is the poster's — which is the whole parity claim: the gallery
/// thumbnail, the viewer's first frame and turntable frame 0 are one picture.
@Suite struct MeshViewerCameraSuite {
    private typealias Camera = MeshViewerCamera

    /// Column-major `m · v`, the multiplication a vertex shader performs.
    private func apply(_ m: Mat4, _ v: [Double]) -> [Double] {
        (0..<4).map { row in
            (0..<4).reduce(0.0) { $0 + Double(m[$1 * 4 + row]) * v[$1] }
        }
    }

    /// The drag-to-yaw gain in `MeshViewer.vue`'s `onPointerMove`.
    private let dragRadiansPerPixel = 0.008

    // MARK: - homeCamera

    @Test func opensOnTheServersPosterCameraExactly() {
        let home = Camera.homeCamera()
        #expect(abs(home.yaw - -0.5235987755982988) < 1e-12)
        #expect(abs(home.pitch - 0.3490658503988659) < 1e-12)
        #expect(home.zoom == 1)
        // The poster's own numbers, not a second set that happens to agree.
        #expect(abs(home.yaw - (-Camera.POSTER_AZIMUTH_DEG * .pi / 180)) < 1e-15)
        #expect(abs(home.pitch - (Camera.POSTER_ELEVATION_DEG * .pi / 180)) < 1e-15)
    }

    @Test func handsBackAFreshValueSoAViewerCannotMutateTheHomeView() {
        var first = Camera.homeCamera()
        first.yaw = 1.5
        #expect(abs(Camera.homeCamera().yaw - -0.5235987755982988) < 1e-12)
    }

    // MARK: - azimuthDegOfYaw

    @Test func readsTheHomeYawBackAsThePostersAzimuth() {
        #expect(abs(Camera.azimuthDegOfYaw(Camera.homeCamera().yaw)
            - Camera.POSTER_AZIMUTH_DEG) < 1e-10)
    }

    @Test func isTheExactInverseOfTheYawConversion() {
        for azimuth in [-180.0, -34.4, 0, 30, 97.5, 180] {
            #expect(abs(Camera.azimuthDegOfYaw(-azimuth * .pi / 180) - azimuth) < 1e-10)
        }
    }

    // MARK: - The turntable turns the way a rightward drag does

    @Test func lowersTheAzimuthForARightwardDragAndStepsTheSweepTheSameWay() {
        let home = Camera.homeCamera()
        // +50 px to the right: `orbit(dx * 0.008, …)`.
        let dragged = home.yaw + 50 * dragRadiansPerPixel
        #expect(Camera.azimuthDegOfYaw(dragged) < Camera.azimuthDegOfYaw(home.yaw))
        // The server's sweep must step the azimuth the same direction, or the
        // GIF would spin opposite to the drag and to the auto-rotate tour.
        #expect(Camera.TURNTABLE_AZIMUTH_STEP_SIGN < 0)
    }

    @Test func agreesWithAdvanceAutoRotateWhichAlsoRaisesTheYaw() {
        let home = Camera.homeCamera()
        let toured = MeshViewerMath.advanceAutoRotate(yaw: home.yaw, elapsedMs: 1000)
        #expect(toured > home.yaw)
        #expect(Camera.azimuthDegOfYaw(toured) < Camera.azimuthDegOfYaw(home.yaw))
    }

    // MARK: - sweepExtent

    private var elevation: Double { Camera.POSTER_ELEVATION_DEG * .pi / 180 }

    @Test func framesAUnitBoxFromEveryAzimuthAtThePosterElevation() {
        // Every corner of [-1, 1]³ is radial √2 out and 1 above or below the
        // centre, so the bound is `cos e · 1 + sin e · √2` — larger than the
        // radial term, which is what makes the elevated view the binding one.
        var positions: [Float] = []
        for x in [Float(-1), 1] {
            for y in [Float(-1), 1] {
                for z in [Float(-1), 1] { positions.append(contentsOf: [x, y, z]) }
            }
        }
        let extent = Camera.sweepExtent(positions, center: .zero, elevationRad: elevation)
        #expect(abs(extent - (cos(elevation) + sin(elevation) * 2.0.squareRoot())) < 1e-6)
        #expect(abs(extent - 1.4233823) < 1e-6)
        #expect(extent > 2.0.squareRoot())
    }

    @Test func isTheRadialDistanceWhenTheMeshIsFlatInTheXZPlane() {
        // No height at all: `cos e · 0 + sin e · r` is smaller than `r`, so
        // the silhouette width wins and the extent is exactly the radius.
        let positions: [Float] = [3, 0, 0, 0, 0, -3, -3, 0, 0]
        #expect(abs(Camera.sweepExtent(positions, center: .zero,
                                       elevationRad: elevation) - 3) < 1e-6)
    }

    @Test func framesTheParsedOneTriangleFixtureAboutItsBoundingBoxCentre() throws {
        let mesh = try GLB.parse(GLBFixture.triangleGLB())
        let center = mesh.bounds.center
        #expect(center == SIMD3(1, 2, -0.5))
        // Every vertex is radial √1.25 from the centre and 2 above or below it.
        let radial = 1.25.squareRoot()
        let extent = Camera.sweepExtent(mesh.positions, center: center,
                                        elevationRad: elevation)
        #expect(abs(extent - (cos(elevation) * 2 + sin(elevation) * radial)) < 1e-6)
        #expect(abs(extent - 2.2617754) < 1e-6)
    }

    @Test func skipsANonFiniteVertexRatherThanLettingItPoisonTheMaximum() {
        let positions: [Float] = [1, 0, 0, .nan, 0, 0, 0, 0, .infinity, 2, 0, 0]
        #expect(abs(Camera.sweepExtent(positions, center: .zero,
                                       elevationRad: elevation) - 2) < 1e-6)
    }

    @Test func isZeroWhenThereIsNothingFiniteToFrame() {
        #expect(Camera.sweepExtent([], center: .zero, elevationRad: elevation) == 0)
        #expect(Camera.sweepExtent([.nan, 0, 0], center: .zero,
                                   elevationRad: elevation) == 0)
        // A trailing partial vertex is dropped, not read as zeros.
        #expect(Camera.sweepExtent([0, 0], center: .zero, elevationRad: elevation) == 0)
    }

    @Test func framesACameraLookingUpExactlyAsOneLookingDown() {
        // `sweep_fit_for` takes the magnitude of both trig terms; a negative
        // elevation is the same view from below and must frame identically.
        let positions: [Float] = [1, 2, 3, -4, 5, -6, 0.5, -7, 2]
        for degrees in [0.0, 20, 45, 89] {
            let radians = degrees * .pi / 180
            #expect(Camera.sweepExtent(positions, center: .zero, elevationRad: -radians)
                == Camera.sweepExtent(positions, center: .zero, elevationRad: radians))
        }
    }

    // MARK: - sweepProfile

    @Test func isTheRadialAndHeightPairOfEveryFiniteVertex() {
        let positions: [Float] = [1, 2, 3, -4, 5, -6, 0.5, -7, 2, 0, 0, 0]
        let profile = Camera.sweepProfile(positions, center: SIMD3(0.25, -0.5, 1))
        #expect(profile.count == 8)
        #expect(abs(Double(profile[0])
            - (0.75 * 0.75 + 2.0 * 2.0).squareRoot()) < 1e-5)
        #expect(abs(Double(profile[1]) - 2.5) < 1e-5)
    }

    @Test func dropsANonFiniteVertexAndATrailingPartialOne() {
        #expect(Camera.sweepProfile([1, 0, 0, .nan, 0, 0, 2, 0, 0], center: .zero).count == 4)
        #expect(Camera.sweepProfile([1, 0, 0, 5, 5], center: .zero).count == 2)
        #expect(Camera.sweepProfile([], center: .zero).isEmpty)
    }

    // MARK: - sweepExtentOfProfile

    /// The bound written out longhand, straight from `sweep_fit_for`, with no
    /// profile in between: the oracle both production paths must reproduce.
    private func reference(_ positions: [Float], _ center: SIMD3<Double>,
                           _ elevationRad: Double) -> Double {
        let sinE = abs(sin(elevationRad))
        let cosE = abs(cos(elevationRad))
        var extent = 0.0
        var index = 0
        while index + 2 < positions.count {
            let dx = Double(positions[index]) - center.x
            let dy = Double(positions[index + 1]) - center.y
            let dz = Double(positions[index + 2]) - center.z
            let radial = (dx * dx + dz * dz).squareRoot()
            extent = Swift.max(extent, radial, cosE * abs(dy) + sinE * radial)
            index += 3
        }
        return extent
    }

    /// The profile exists only so a tilting viewer can re-frame without the
    /// mesh; it must never become a second definition of the framing.
    @Test func agreesWithSweepExtentAndWithTheLonghandBoundAtEveryElevation() {
        let positions: [Float] = [1, 2, 3, -4, 5, -6, 0.5, -7, 2, 0, 0, 0, 9, -1, -2]
        let center = SIMD3<Double>(0.25, -0.5, 1)
        let profile = Camera.sweepProfile(positions, center: center)
        for degrees in [-89.0, -20, 0, 20, 45, 89] {
            let radians = degrees * .pi / 180
            let value = Camera.sweepExtentOfProfile(profile, elevationRad: radians)
            #expect(value == Camera.sweepExtent(positions, center: center,
                                                elevationRad: radians))
            // The profile stores its pairs as f32, exactly as the Rust bound
            // computes them, so the agreement is to single precision.
            #expect(abs(value - reference(positions, center, radians)) < 1e-5)
        }
    }

    @Test func growsAsTheCameraTiltsAwayFromThePostersElevation() {
        // A half-unit cube: 0.7117 at 20°, 0.8536 at 45°. Framing the 45° view
        // with the 20° number would clip a tenth of the silhouette.
        var positions: [Float] = []
        for x in [Float(-0.5), 0.5] {
            for y in [Float(-0.5), 0.5] {
                for z in [Float(-0.5), 0.5] { positions.append(contentsOf: [x, y, z]) }
            }
        }
        let profile = Camera.sweepProfile(positions, center: .zero)
        let poster = Camera.sweepExtentOfProfile(profile, elevationRad: elevation)
        #expect(abs(poster - 0.7117) < 1e-4)
        let tilted = Camera.sweepExtentOfProfile(profile, elevationRad: .pi / 4)
        #expect(abs(tilted - 0.8536) < 1e-4)
        #expect(tilted > poster)
    }

    @Test func isZeroForAnEmptyProfile() {
        #expect(Camera.sweepExtentOfProfile([], elevationRad: 0.3) == 0)
    }

    // MARK: - orthographicScale

    @Test func matchesTheRustFitScaleArithmeticOnA64by48Frame() {
        // `min(half_w, half_h) / extent * (1 - margin)` — the short axis binds.
        for extent in [0.5, 1, 2.2617754, 17] {
            #expect(abs(Camera.orthographicScale(extent: extent, width: 64, height: 48,
                                                 margin: Camera.POSTER_MARGIN)
                - (24 / extent) * 0.92) < 1e-10)
        }
        #expect(abs(Camera.orthographicScale(extent: 2, width: 48, height: 64,
                                             margin: Camera.POSTER_MARGIN)
            - (24 / 2) * 0.92) < 1e-10)
    }

    @Test func leavesTheMarginsShareOfTheShortAxisEmpty() {
        // The mesh's extent lands at `1 - margin` of the half-frame, which is
        // what makes the viewer's home frame the poster's frame.
        let scale = Camera.orthographicScale(extent: 3, width: 200, height: 200,
                                             margin: Camera.POSTER_MARGIN)
        #expect(abs(3 * scale - 100 * (1 - Camera.POSTER_MARGIN)) < 1e-10)
    }

    @Test func clampsTheMarginTheWayTheRasterizerDoes() {
        #expect(abs(Camera.orthographicScale(extent: 1, width: 100, height: 100,
                                             margin: -1) - 50) < 1e-10)
        #expect(abs(Camera.orthographicScale(extent: 1, width: 100, height: 100,
                                             margin: 0.95) - 5) < 1e-10)
        #expect(abs(Camera.orthographicScale(extent: 1, width: 100, height: 100,
                                             margin: .nan) - 50) < 1e-10)
    }

    @Test func isZeroRatherThanInfinityForAMeshWithNoExtent() {
        let margin = Camera.POSTER_MARGIN
        #expect(Camera.orthographicScale(extent: 0, width: 64, height: 48, margin: margin) == 0)
        #expect(Camera.orthographicScale(extent: -1, width: 64, height: 48, margin: margin) == 0)
        #expect(Camera.orthographicScale(extent: .nan, width: 64, height: 48,
                                         margin: margin) == 0)
        #expect(Camera.orthographicScale(extent: .infinity, width: 64, height: 48,
                                         margin: margin) == 0)
        #expect(Camera.orthographicScale(extent: 2, width: 0, height: 0, margin: margin) == 0)
    }

    // MARK: - Matrices

    @Test func mapsTheHalfExtentsToTheEdgesOfTheNDCCube() {
        let m = MeshMatrix.orthographic(halfWidth: 4, halfHeight: 3, near: 1, far: 9)
        let corner = apply(m, [4, 3, -5, 1])
        #expect(abs(corner[0] - 1) < 1e-6)
        #expect(abs(corner[1] - 1) < 1e-6)
        let opposite = apply(m, [-4, -3, -5, 1])
        #expect(abs(opposite[0] + 1) < 1e-6)
        #expect(abs(opposite[1] + 1) < 1e-6)
    }

    @Test func mapsTheNearAndFarPlanesWithNoPerspectiveDivide() {
        let m = MeshMatrix.orthographic(halfWidth: 4, halfHeight: 3, near: 1, far: 9)
        #expect(abs(apply(m, [0, 0, -1, 1])[2] + 1) < 1e-6)
        #expect(abs(apply(m, [0, 0, -9, 1])[2] - 1) < 1e-6)
        // `w` stays 1: the whole point of an orthographic frame.
        #expect(apply(m, [0, 0, -1, 1])[3] == 1)
        #expect(m[11] == 0)
        #expect(m[15] == 1)
    }

    @Test func multipliesColumnMajorSoTranslationComposesOnTheRight() {
        let m = MeshMatrix.multiply(MeshMatrix.translation(1, 2, 3),
                                    MeshMatrix.translation(10, 20, 30))
        #expect(apply(m, [0, 0, 0, 1]) == [11, 22, 33, 1])
    }

    @Test func leavesAVectorAloneUnderTheIdentity() {
        #expect(apply(MeshMatrix.identity(), [1, -2, 3, 1]) == [1, -2, 3, 1])
    }

    @Test func turnsPlusZTowardPlusXUnderAPositiveRotationY() {
        let turned = apply(MeshMatrix.rotationY(.pi / 2), [0, 0, 1, 1])
        #expect(abs(turned[0] - 1) < 1e-6)
        #expect(abs(turned[2]) < 1e-6)
    }

    @Test func tipsPlusYTowardPlusZUnderAPositiveRotationX() {
        let tipped = apply(MeshMatrix.rotationX(.pi / 2), [0, 1, 0, 1])
        #expect(abs(tipped[1]) < 1e-6)
        #expect(abs(tipped[2] - 1) < 1e-6)
    }

    @Test func takesTheUpperLeft3x3AsTheNormalMatrix() {
        let m = MeshMatrix.multiply(MeshMatrix.translation(5, 6, 7),
                                    MeshMatrix.rotationY(0.4))
        #expect(MeshMatrix.upper3x3(m) == MeshMatrix.upper3x3(MeshMatrix.rotationY(0.4)))
    }

    /// The whole draw, composed the way `MeshViewer.vue:307-313` composes it:
    /// the bounding-box CENTRE ends up on the eye axis at the orbit distance,
    /// whatever the camera angle — a mesh centred on the query grid rather
    /// than on the model would otherwise swing out of frame as it turned.
    @Test func putsTheBoundingBoxCentreOnTheEyeAxisAtEveryAngle() {
        let center = SIMD3<Double>(1, 2, -0.5)
        for camera in [Camera.homeCamera(),
                       ViewerCamera(yaw: 2.1, pitch: -0.9, zoom: 1),
                       ViewerCamera(yaw: -3, pitch: 1.4, zoom: 4)] {
            let m = MeshMatrix.modelView(camera: camera, center: center, distance: 3)
            let eye = apply(m, [center.x, center.y, center.z, 1])
            #expect(abs(eye[0]) < 1e-6)
            #expect(abs(eye[1]) < 1e-6)
            #expect(abs(eye[2] + 3) < 1e-6)
        }
    }
}
