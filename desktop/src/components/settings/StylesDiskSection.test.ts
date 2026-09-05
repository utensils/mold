/**
 * Settings ▸ Styles & disk. The meter reads the primary's own `models_disk`
 * and its reading shares one unit with the machine card and the status bar,
 * so the same bytes cannot read two ways in two places.
 */
import { beforeEach, describe, expect, it, vi } from "vitest";
import { mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";

vi.mock("./ConfigSettingRow.vue", () => ({ default: { template: "<div />" } }));

import StylesDiskSection from "./StylesDiskSection.vue";
import { useHostStatusStore } from "../../stores/hostStatus";

beforeEach(() => setActivePinia(createPinia()));

describe("StylesDiskSection", () => {
  it("hides the meter until the primary reports a disk", () => {
    const wrapper = mount(StylesDiskSection);
    expect(wrapper.find("[data-test='settings-disk-meter']").exists()).toBe(false);
  });

  it("states the reading as one used/total pair, decimal, with the unit once", () => {
    const hostStatus = useHostStatusStore();
    hostStatus.status = {
      models_disk: { total_bytes: 36_000_000_000, free_bytes: 23_900_000_000 },
    } as never;

    const meter = mount(StylesDiskSection).get("[data-test='settings-disk-meter']");
    expect(meter.text()).toContain("Disk for styles");
    expect(meter.text()).toContain("12.1 / 36.0 GB");
    expect(meter.get("[role='meter']").attributes("aria-valuenow")).toBe("34");
  });
});
