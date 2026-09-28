import Foundation
import MoldClient
import Testing

@testable import MoldCompanion

/// Progress in words a first-timer can act on, with the engineer's figure in
/// mono beside it -- and the server's own stage names as the inputs.
struct ProgressWordsTests {
    private func progress(stage: String?, step: Int? = nil, total: Int? = nil) throws -> JobProgress {
        var fields: [String] = []
        if let stage { fields.append(#""stage":"\#(stage)""#) }
        if let step { fields.append(#""step":\#(step)"#) }
        if let total { fields.append(#""total":\#(total)"#) }
        return try MoldJSON.decoder.decode(JobProgress.self, from: Data("{\(fields.joined(separator: ","))}".utf8))
    }

    @Test func theServersStagesReadAsPlainWords() throws {
        #expect(ProgressWords.plain(try progress(stage: "Loading VAE (GPU)")) == "Getting the model ready")
        #expect(ProgressWords.plain(try progress(stage: "Encoding prompt")) == "Reading your prompt")
        #expect(ProgressWords.plain(try progress(stage: "Encoding source image (VAE)")) == "Studying your picture")
        #expect(ProgressWords.plain(try progress(stage: "Denoising", step: 3, total: 28)) == "Adding detail")
        #expect(ProgressWords.plain(try progress(stage: "VAE decode")) == "Finishing the picture")
        #expect(ProgressWords.plain(try progress(stage: "Removing background")) == "Removing the background")
    }

    @Test func anUnknownStageKeepsTheMachinesOwnWords() throws {
        #expect(ProgressWords.plain(try progress(stage: "reticulating splines")) == "Reticulating splines")
    }

    @Test func theFigureIsTheStageAndTheSteps() throws {
        #expect(ProgressWords.figure(try progress(stage: "Denoising", step: 18, total: 28)) == "denoise 18/28")
    }

    @Test func timeLeftJoinsTheSentenceOnlyWhenMeasured() throws {
        let p = try progress(stage: "Denoising", step: 18, total: 28)
        #expect(ProgressWords.sentence(p) == "Adding detail")
        #expect(ProgressWords.sentence(p, remaining: .seconds(12)).hasPrefix("Adding detail — about"))
        #expect(ProgressWords.spoken(p) == "18 of 28")
    }

    @Test func aQueuedRenderSaysWhereItStands() {
        #expect(ProgressWords.sentence(nil, position: 1) == "Waiting — next in line")
        #expect(ProgressWords.sentence(nil, position: 3) == "Waiting — 2 ahead")
    }

    @Test func theRateNeedsTwoStepsBeforeItSaysAnything() throws {
        var rate = StepRate()
        let start = Date(timeIntervalSince1970: 0)
        #expect(rate.remaining(try progress(stage: nil, step: 2, total: 12), now: start) == nil)
        let left = rate.remaining(try progress(stage: nil, step: 4, total: 12), now: start.addingTimeInterval(2))
        #expect(left == .seconds(8))
    }
}

extension ProgressWordsTests {
    @Test func theAnnouncementComesAtEachQuarter() throws {
        func progress(_ step: Int, _ total: Int) throws -> JobProgress {
            try MoldJSON.decoder.decode(JobProgress.self, from: Data(#"{"step":\#(step),"total":\#(total)}"#.utf8))
        }
        #expect(ProgressWords.quarter(nil) == nil)
        #expect(ProgressWords.quarter(try progress(6, 28)) == 0)
        #expect(ProgressWords.quarter(try progress(7, 28)) == 1)
        #expect(ProgressWords.quarter(try progress(14, 28)) == 2)
        #expect(ProgressWords.quarter(try progress(28, 28)) == 4)
    }
}
