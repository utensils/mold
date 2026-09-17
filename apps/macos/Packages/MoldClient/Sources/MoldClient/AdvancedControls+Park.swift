import Foundation

// Parking a sampler control the new recipe does not advertise, and handing it
// back when one that does returns. The rule and the helper are the ones every
// conditioning input already uses (`DraftMedia.reconcile`): a live value
// always beats a parked one, because parking is a rescue and not a history.
public extension AdvancedControls {
    mutating func reconcile(with offered: AdvancedControlsOffered) {
        reconcileScheduler(offered.schedulers)
        reconcileCfgPlus(supported: offered.cfgPlus)
        DraftMedia.reconcile(&sampleShift, &parked.sampleShift, supported: offered.sampleShift)
        DraftMedia.reconcile(&distillStrengthHigh, &parked.distillStrengthHigh,
                             supported: offered.distillStrength)
        DraftMedia.reconcile(&distillStrengthLow, &parked.distillStrengthLow,
                             supported: offered.distillStrength)
        DraftMedia.reconcile(&stgScale, &parked.stgScale, supported: offered.guidance)
        DraftMedia.reconcile(&rescaleScale, &parked.rescaleScale, supported: offered.guidance)
        DraftMedia.reconcile(&skipStep, &parked.skipStep, supported: offered.guidance)
        DraftMedia.reconcile(&modalityScale, &parked.modalityScale, supported: offered.modalityScale)
        reconcileStgBlocks(supported: offered.guidance)
    }

    /// The solver is the one control whose support depends on its own VALUE,
    /// not merely on the recipe: a wan tier's `dpm-pp` carried onto an SDXL
    /// recipe is not a solver SDXL lists, and the server's `Scheduler` enum
    /// would take it happily and sample with something nobody chose. So this
    /// cannot use the shared helper, which asks one yes/no for the field.
    private mutating func reconcileScheduler(_ advertised: [String]) {
        if let scheduler, !advertised.contains(scheduler) {
            parked.scheduler = scheduler
            self.scheduler = nil
        }
        if scheduler == nil, let restored = parked.scheduler, advertised.contains(restored) {
            scheduler = restored
            parked.scheduler = nil
        }
    }

    /// `cfgPlus` is a plain `Bool` on the live side -- a switch is on or off,
    /// there is no third state -- so `false` is the empty slot, and the parked
    /// twin is optional to keep "nothing parked" and "parked off" apart.
    private mutating func reconcileCfgPlus(supported: Bool) {
        if supported {
            if !cfgPlus, let restored = parked.cfgPlus {
                cfgPlus = restored
                parked.cfgPlus = nil
            }
        } else if cfgPlus {
            parked.cfgPlus = true
            cfgPlus = false
        }
    }

    /// The block list is free text, so "" is its empty slot.
    private mutating func reconcileStgBlocks(supported: Bool) {
        if supported {
            if stgBlocks.isEmpty, let restored = parked.stgBlocks {
                stgBlocks = restored
                parked.stgBlocks = nil
            }
        } else if !stgBlocks.isEmpty {
            parked.stgBlocks = stgBlocks
            stgBlocks = ""
        }
    }

    /// Every live control back to the recipe's own value. What the group's
    /// Reset verb does. The PARK is deliberately left alone: a reset is about
    /// this recipe's controls, and clearing another model's rescued values
    /// with it would lose them for a switch nobody made.
    mutating func reset() {
        let parked = parked
        self = AdvancedControls()
        self.parked = parked
    }

    /// How many controls have been moved off the recipe's own values -- the
    /// group's count, and what its Reset row is gated on. Port of
    /// `guidanceOverrideCount` (`:67-78`) and `wanRecipeCount`
    /// (`wanRecipe.ts:50-59`), joined because this app draws one group.
    var touchedCount: Int {
        var count = 0
        if scheduler != nil { count += 1 }
        if cfgPlus { count += 1 }
        for value in [sampleShift, distillStrengthHigh, distillStrengthLow,
                      stgScale, rescaleScale, modalityScale] where value != nil {
            count += 1
        }
        if skipStep != nil { count += 1 }
        if !stgBlocks.trimmingCharacters(in: .whitespaces).isEmpty { count += 1 }
        return count
    }
}
