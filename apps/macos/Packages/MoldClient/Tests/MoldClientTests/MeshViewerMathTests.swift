import Foundation
import Testing

@testable import MoldClient

/// Ported from `studio/lib/meshViewerMath.test.ts`.
///
/// **Fails today**: neither the auto-rotate step nor the edge list existed on
/// this Mac — there was no mesh view to need them.
@Suite struct MeshViewerMathSuite {
    private typealias Math = MeshViewerMath

    // MARK: - advanceAutoRotate

    @Test func advancesTheYawByTheRateTimesTheElapsedSeconds() {
        #expect(abs(Math.advanceAutoRotate(yaw: 0, elapsedMs: 1000) - 0.25) < 1e-10)
        #expect(abs(Math.advanceAutoRotate(yaw: 0, elapsedMs: 500) - 0.125) < 1e-10)
        #expect(abs(Math.advanceAutoRotate(yaw: 0.1, elapsedMs: 2000,
                                           radiansPerSecond: 0.5) - 1.1) < 1e-10)
    }

    @Test func wrapsIntoMinusPiToPiSoALongRotationNeverGrowsUnbounded() {
        let wrapped = Math.advanceAutoRotate(yaw: .pi - 0.1, elapsedMs: 1000)
        #expect(abs(wrapped - (-Double.pi + 0.15)) < 1e-10)

        var yaw = 0.0
        for _ in 0..<10_000 { yaw = Math.advanceAutoRotate(yaw: yaw, elapsedMs: 100) }
        #expect(yaw >= -.pi)
        #expect(yaw < .pi)
    }

    @Test func wrapsAYawTheCallerAlreadyPushedOutOfRange() {
        #expect(abs(Math.advanceAutoRotate(yaw: .pi * 4, elapsedMs: 0)) < 1e-10)
        #expect(abs(Math.advanceAutoRotate(yaw: -.pi * 3, elapsedMs: 0) + .pi) < 1e-10)
    }

    @Test func standsStillForANonPositiveOrNonFiniteElapsedTime() {
        #expect(abs(Math.advanceAutoRotate(yaw: 0.4, elapsedMs: 0) - 0.4) < 1e-10)
        #expect(abs(Math.advanceAutoRotate(yaw: 0.4, elapsedMs: -16) - 0.4) < 1e-10)
        #expect(abs(Math.advanceAutoRotate(yaw: 0.4, elapsedMs: .nan) - 0.4) < 1e-10)
    }

    // MARK: - edgeIndices

    @Test func givesOneTriangleItsThreeEdges() {
        #expect(Math.edgeIndices([0, 1, 2]) == [0, 1, 1, 2, 0, 2])
    }

    @Test func emitsASharedEdgeOnceSoTwoTrianglesGiveFiveEdges() {
        let edges = Math.edgeIndices([0, 1, 2, 2, 1, 3])
        #expect(edges.count == 10)
        #expect(edges == [0, 1, 1, 2, 0, 2, 1, 3, 2, 3])
    }

    @Test func ordersEveryEdgeLowToHighAndKeepsFirstSeenOrder() {
        #expect(Math.edgeIndices([7, 3, 5]) == [3, 7, 3, 5, 5, 7])
    }

    @Test func skipsDegenerateEdgesAndATrailingPartialTriangle() {
        #expect(Math.edgeIndices([0, 0, 1]) == [0, 1])
        #expect(Math.edgeIndices([4, 4, 4]).isEmpty)
        #expect(Math.edgeIndices([0, 1, 2, 3, 4]) == [0, 1, 1, 2, 0, 2])
    }

    /// The readable set-of-partners version this replaced, kept as the ORACLE:
    /// the production path packs each edge into one number so a large mesh no
    /// longer allocates a set per vertex, but the edge list it emits must be
    /// identical, edge for edge, in first-seen order.
    private func referenceEdgeIndices(_ indices: [UInt32]) -> [UInt32] {
        var seen: [UInt32: Set<UInt32>] = [:]
        var out: [UInt32] = []
        let triangles = indices.count - (indices.count % 3)
        func add(_ a: UInt32, _ b: UInt32) {
            if a == b { return }
            let low = Swift.min(a, b)
            let high = Swift.max(a, b)
            if seen[low, default: []].contains(high) { return }
            seen[low, default: []].insert(high)
            out.append(low)
            out.append(high)
        }
        var position = 0
        while position < triangles {
            add(indices[position], indices[position + 1])
            add(indices[position + 1], indices[position + 2])
            add(indices[position], indices[position + 2])
            position += 3
        }
        return out
    }

    @Test func matchesTheReferenceEdgeListOnALargeSharedEdgeGrid() {
        // A 300×300 vertex grid, two triangles per cell: 179,400 triangles
        // whose interior edges are each shared by two of them.
        let side: UInt32 = 300
        let cells = side - 1
        var indices: [UInt32] = []
        indices.reserveCapacity(Int(cells * cells) * 6)
        for row in 0..<cells {
            for column in 0..<cells {
                let a = row * side + column
                indices.append(contentsOf: [a, a + 1, a + side, a + 1, a + side + 1, a + side])
            }
        }
        let edges = Math.edgeIndices(indices)
        #expect(edges == referenceEdgeIndices(indices))
        // Euler: a grid has 2·c·(c+1) axis edges plus one diagonal per cell.
        #expect(edges.count / 2 == Int(2 * cells * (cells + 1) + cells * cells))
    }

    @Test func matchesTheReferenceOnUnorderedIndicesThatShareEdgesBothWays() {
        var indices: [UInt32] = []
        var seed: UInt64 = 7
        for _ in 0..<60_000 {
            seed = (seed &* 1_103_515_245 &+ 12345) % 2_147_483_648
            indices.append(UInt32(seed % 5000))
        }
        #expect(Math.edgeIndices(indices) == referenceEdgeIndices(indices))
    }

    /// The packing is `low * vertexCount + high`; a 16-bit key would collide
    /// two different edges of a real mesh into one.
    @Test func keepsAVertexIndexAbove65536ExactWhenPackingAnEdge() {
        let indices: [UInt32] = [0, 70_000, 2_000_000, 2_000_000, 70_000, 1]
        #expect(Math.edgeIndices(indices) == [0, 70_000, 70_000, 2_000_000, 0, 2_000_000,
                                              1, 70_000, 1, 2_000_000])
    }

    @Test func handsBackAnEmptyBufferForAnEmptyIndexList() {
        #expect(Math.edgeIndices([]).isEmpty)
    }

    // MARK: - meshHasEdges

    @Test func isTrueAsSoonAsOneCompleteTriangleJoinsTwoDistinctVertices() {
        #expect(Math.meshHasEdges([0, 1, 2]))
        #expect(Math.meshHasEdges([4, 4, 4, 4, 4, 5]))
        #expect(Math.meshHasEdges([7, 7, 3]))
    }

    @Test func isFalseForNoTrianglesDegenerateOnesOrAPartialOne() {
        #expect(!Math.meshHasEdges([]))
        #expect(!Math.meshHasEdges([4, 4, 4]))
        #expect(!Math.meshHasEdges([0, 1]))
        #expect(!Math.meshHasEdges([2, 2, 2, 0, 1]))
    }

    /// The whole point of the cheap scan: the wireframe button is offered
    /// exactly when switching it on would draw something.
    @Test func agreesWithEdgeIndicesOnWhetherAnythingWouldBeDrawn() {
        for indices: [UInt32] in [[0, 1, 2], [3, 3, 3], [], [1, 1, 1, 2, 2, 2, 5]] {
            #expect(Math.meshHasEdges(indices) == !Math.edgeIndices(indices).isEmpty)
        }
    }
}
