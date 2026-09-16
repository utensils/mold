import Foundation
import Testing

@testable import MoldClient

/// Wan's real profile: frames step 4 from an offset of 1, so 4k+1.
private let wan = try! MoldJSON.decoder.decode(TemporalProfile.self, from: Data("""
{"frames":{"default":121,"min":1,"max":257,"step":4,"recommended":[121],"mode":"adjustable"},
 "frame_offset":1,"fps":{"mode":"adjustable","default":24,"min":1,"max":120,"step":1}}
""".utf8))

@Test func decodesAnAdjustableFpsControl() {
    #expect(wan.fps.isAdjustable)
    #expect(wan.fps.value == 24)
}

@Test func decodesAFixedFpsControlAsASingleValue() throws {
    let fixed = try MoldJSON.decoder.decode(FpsControl.self, from: Data("""
    {"mode":"fixed","value":16}
    """.utf8))
    #expect(!fixed.isAdjustable)
    #expect(fixed.value == 16)
}

@Test func anUnknownFpsModeDoesNotFailTheDecode() throws {
    let odd = try MoldJSON.decoder.decode(FpsControl.self, from: Data("""
    {"mode":"something_new"}
    """.utf8))
    #expect(odd == .unknown)
}

@Test func framesSnapToTheGridTheFamilyActuallyAccepts() {
    // Wan's grid is 4k+1: it refuses 120 and accepts 121. Sending the wrong
    // one is a 422, not a rounded render.
    #expect(wan.snap(120) == 121)
    #expect(wan.snap(121) == 121)
    #expect(wan.snap(122) == 121)
    #expect(wan.snap(123) == 125)
    // Every snapped value must be on the grid.
    for requested in stride(from: 1, through: 257, by: 7) {
        #expect((wan.snap(requested) - 1) % 4 == 0)
    }
}

@Test func snappingStaysInsideTheAdvertisedBounds() {
    #expect(wan.snap(-50) >= wan.frames.min)
    #expect(wan.snap(10_000) <= wan.frames.max)
}

@Test func durationUsesTheRecipesOwnRate() {
    // 121 frames at 24fps is a five second clip.
    #expect(abs(wan.duration(forFrames: 121) - 5.04) < 0.01)
}

@Test func sourceImageSupportReadsThroughFromTheWire() throws {
    #expect(try MoldJSON.decoder.decode(SourceImageCapability.self,
                                        from: Data("\"unsupported\"".utf8)).isSupported == false)
    #expect(try MoldJSON.decoder.decode(SourceImageCapability.self,
                                        from: Data("\"optional\"".utf8)).isSupported)
    // A value from a newer host must not fail the whole model list.
    #expect(try MoldJSON.decoder.decode(SourceImageCapability.self,
                                        from: Data("\"brand_new\"".utf8)) == .unknown)
}
