import Foundation
import Testing

@testable import MoldClient

/// Parking a sampler control across a model switch, and what reaches the wire.
/// **Fails today**: none of these controls existed.
struct AdvancedControlsParkTests {
    private func offered(
        schedulers: [String] = [], cfgPlus: Bool = false, sampleShift: Bool = false,
        distillStrength: Bool = false, guidance: Bool = false, modalityScale: Bool = false
    ) -> AdvancedControlsOffered {
        AdvancedControlsOffered(
            schedulers: schedulers, cfgPlus: cfgPlus, sampleShift: sampleShift,
            distillStrength: distillStrength, guidance: guidance, modalityScale: modalityScale)
    }

    /// The solver is the one control whose support depends on its VALUE: a
    /// wan tier's `dpm-pp` is a spelling the server's `Scheduler` enum takes
    /// happily, so carrying it onto a recipe that does not list it would
    /// sample with something nobody chose.
    @Test func aSolverThisRecipeDoesNotListIsParkedAndComesBack() {
        var controls = AdvancedControls()
        controls.scheduler = "dpm-pp"

        controls.reconcile(with: offered(schedulers: ["ddim", "euler-ancestral", "uni-pc"]))
        #expect(controls.scheduler == nil)
        #expect(controls.parked.scheduler == "dpm-pp")

        controls.reconcile(with: offered(schedulers: ["uni-pc", "euler", "dpm-pp"]))
        #expect(controls.scheduler == "dpm-pp")
        #expect(controls.parked.scheduler == nil)
    }

    /// The SECOND switch is where a naive park loses things: a value parked by
    /// one recipe must survive a recipe that also cannot take it, rather than
    /// being overwritten by nothing.
    @Test func aValueSurvivesTwoRecipesThatCannotTakeIt() {
        var controls = AdvancedControls()
        controls.sampleShift = 5
        controls.stgBlocks = "3, 7"
        controls.cfgPlus = true

        controls.reconcile(with: offered())
        controls.reconcile(with: offered())
        #expect(controls.sampleShift == nil)
        #expect(controls.parked.sampleShift == 5)
        #expect(controls.parked.stgBlocks == "3, 7")
        #expect(controls.parked.cfgPlus == true)

        controls.reconcile(with: offered(cfgPlus: true, sampleShift: true, guidance: true))
        #expect(controls.sampleShift == 5)
        #expect(controls.stgBlocks == "3, 7")
        #expect(controls.cfgPlus)
    }

    /// A live value beats a parked one -- parking is a rescue, not a history.
    @Test func aLiveValueIsNotOverwrittenByAParkedOne() {
        var controls = AdvancedControls()
        controls.parked.sampleShift = 1
        controls.sampleShift = 9
        controls.reconcile(with: offered(sampleShift: true))
        #expect(controls.sampleShift == 9)
        #expect(controls.parked.sampleShift == 1)
    }

    /// Reset puts the live controls back to the recipe's own values and
    /// leaves the PARK alone: another model's rescued values are not this
    /// reset's to throw away.
    @Test func resetClearsTheLiveControlsAndKeepsThePark() {
        var controls = AdvancedControls()
        controls.parked.stgScale = 2
        controls.sampleShift = 5
        controls.cfgPlus = true
        #expect(controls.touchedCount == 2)

        controls.reset()
        #expect(controls.touchedCount == 0)
        #expect(controls.parked.stgScale == 2)
    }

    /// An empty override set is refused outright by admission
    /// ("must set at least one field; omit it to keep pipeline defaults"), so
    /// the absent case has to be ABSENCE and not `{}`.
    @Test func anUntouchedDraftPutsNoSamplerFieldOnTheWire() {
        let request = RenderRequest.one(RenderDraft(), model: "m")
        #expect(request.guidanceOverrides == nil)
        #expect(request.scheduler == nil)
        #expect(request.cfgPlus == nil)
        #expect(request.sampleShift == nil)
    }

    /// CFG++ is `true` or nothing. Absence IS false to the server, so an
    /// explicit `false` would record a choice nobody made.
    @Test func cfgPlusReachesTheWireOnlyWhenItIsOn() {
        var draft = RenderDraft()
        draft.advanced.cfgPlus = true
        #expect(RenderRequest.one(draft, model: "m").cfgPlus == true)
        draft.advanced.cfgPlus = false
        #expect(RenderRequest.one(draft, model: "m").cfgPlus == nil)
    }

    /// A value the wire cannot carry contributes nothing, and does not take
    /// the rest of the block with it.
    @Test func anUnusableBlockListDoesNotSinkTheOtherOverrides() {
        var draft = RenderDraft()
        draft.advanced.stgBlocks = "3, banana"
        draft.advanced.stgScale = 1.5
        let overrides = RenderRequest.one(draft, model: "m").guidanceOverrides
        #expect(overrides?.stgBlocks == nil)
        #expect(overrides?.stgScale == 1.5)
        #expect(draft.advanced.refusal == "STG blocks: \"banana\" is not a block index.")
    }
}
