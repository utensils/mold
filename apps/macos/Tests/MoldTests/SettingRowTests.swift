import Foundation
import MoldClient
import Testing

@testable import Mold

/// `SettingRow.resolve` (S4a): pure, so a curated pane's per-row rendering
/// decision is askable with no view at all -- the same `DiscoverRow.resolve`
/// idiom `ConfigValueField.Plan` already uses.
struct SettingRowTests {
    private var widthSetting: SettingKey {
        SettingKeys.generationRendering.first { $0.key == "default_width" }!
    }

    private var variantSetting: SettingKey {
        SettingKeys.generationRendering.first { $0.key == "t5_variant" }!
    }

    @Test func aKeyTheMachineNeverListedDrawsNothing() {
        #expect(SettingRow.resolve(widthSetting, entry: nil) == nil)
    }

    @Test func aRefusalRidesTheRow() {
        let entry = FakeFixtures.configEntry("default_width", value: .number(1024), source: "db")
        let plan = SettingRow.resolve(widthSetting, entry: entry, refusal: "Must be between 64 and 8192.")
        #expect(plan?.refusal == "Must be between 64 and 8192.")
    }

    @Test func aChoiceDrawsTheEnginesList() {
        let entry = FakeFixtures.configEntry("t5_variant", value: .string("q8"), source: "db")
        let plan = SettingRow.resolve(variantSetting, entry: entry)
        #expect(plan?.setting.editor == .choice(["auto", "fp16", "q8", "q6", "q5", "q4", "q3"]))
    }

    @Test func aNumberWithAStepDrawsAStepper() {
        let entry = FakeFixtures.configEntry("default_width", value: .number(1024), source: "db")
        let plan = SettingRow.resolve(widthSetting, entry: entry)
        #expect(plan?.showsStepper == true)
    }

    @Test func aNumberWithNoStepDrawsNoStepper() {
        // No curated key declares a nil step today -- built directly so the
        // negative case is real rather than assumed.
        let setting = SettingKey(
            key: "default_steps", label: "Steps", help: "",
            editor: .number(min: 1, max: 1000, step: nil))
        let entry = FakeFixtures.configEntry("default_steps", value: .number(20), source: "db")
        let plan = SettingRow.resolve(setting, entry: entry)
        #expect(plan?.showsStepper == false)
    }
}
