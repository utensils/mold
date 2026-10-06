import Testing
@testable import MoldClient

struct AspectRatioGeometryTests {
    @Test func outlinesPreserveActualOfferedRatios() {
        let landscape = AspectRatioGeometry.size(width: 704, height: 1280, bound: 22)
        #expect(landscape.height == 22)
        #expect(abs(landscape.width / landscape.height - 11.0 / 20.0) < 0.00001)
        let square = AspectRatioGeometry.size(width: 960, height: 960, bound: 22)
        #expect(square.width == square.height)
        let wide = AspectRatioGeometry.size(width: 1344, height: 768, bound: 22)
        #expect(wide.width == 22)
        #expect(abs(wide.width / wide.height - 7.0 / 4.0) < 0.00001)
    }
}
