import Foundation
import Testing

@testable import MoldClient

// Ported from `studio/lib/videoOnly.test.ts`, which is the oracle: this file
// and that one describe the same policy, and mold's rule is that neither
// surface invents its own conflict table.
//
// The whole thing was untested here -- including the deliberate precedence
// and the `enabled && blocked` case, whose answer is an OUTPUT change if it
// is wrong.

private typealias Inputs = VideoOnlyPolicy.Inputs

@Test func aCleanFormBlocksNothing() {
    #expect(VideoOnlyPolicy.blockedReason(Inputs()) == nil)
}

@Test func eachConflictSaysWhatToDoAboutIt() {
    #expect(VideoOnlyPolicy.blockedReason(Inputs(audioEnabled: true))
            == "Turn off Generate audio first — video-only skips the branch that renders it.")
    #expect(VideoOnlyPolicy.blockedReason(Inputs(audioOnlyPipeline: true))
            == "Text-to-audio renders sound only; video-only does not apply.")
    #expect(VideoOnlyPolicy.blockedReason(Inputs(hasConditioningAudio: true))
            == "Remove the conditioning audio first — video-only skips the branch it drives.")
    #expect(VideoOnlyPolicy.blockedReason(Inputs(isExtend: true))
            == "A continuation keeps its source clip's rendering path.")
}

/// The precedence is deliberate: a text-to-audio pipeline is the one conflict
/// nothing else can fix, so it is named even when the form also has audio
/// turned on -- telling someone to turn off audio on a render that is ONLY
/// audio would send them in a circle.
@Test func theAudioOnlyPipelineExplanationWinsOverEveryOtherConflict() {
    let everything = Inputs(
        audioEnabled: true, audioOnlyPipeline: true,
        hasConditioningAudio: true, isExtend: true)
    #expect(VideoOnlyPolicy.blockedReason(everything)
            == "Text-to-audio renders sound only; video-only does not apply.")
}

/// And the rest of the order, each pair checked against the one behind it.
@Test func theRemainingConflictsKeepTheirOrder() {
    #expect(VideoOnlyPolicy.blockedReason(Inputs(audioEnabled: true, hasConditioningAudio: true))
            == VideoOnlyPolicy.blockedReason(Inputs(audioEnabled: true)))
    #expect(VideoOnlyPolicy.blockedReason(Inputs(hasConditioningAudio: true, isExtend: true))
            == VideoOnlyPolicy.blockedReason(Inputs(hasConditioningAudio: true)))
}

/// The wire value is `true` or ABSENT, never `false`.
///
/// Sending `false` would pin the server off its default multimodal path
/// rather than leaving it to decide -- an output change, from a field the
/// user never set.
@Test func theWireValueIsTrueOrAbsentAndNeverFalse() {
    #expect(VideoOnlyPolicy.requestValue(enabled: true, Inputs()) == true)
    #expect(VideoOnlyPolicy.requestValue(enabled: false, Inputs()) == nil)
    // Enabled but blocked: the toggle is on and the form cannot carry it.
    // `false` here is the trap -- the answer is to send nothing.
    #expect(VideoOnlyPolicy.requestValue(enabled: true, Inputs(audioEnabled: true)) == nil)
    #expect(VideoOnlyPolicy.requestValue(enabled: false, Inputs(audioEnabled: true)) == nil)
}
