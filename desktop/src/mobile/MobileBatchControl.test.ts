import { flushPromises, mount } from "@vue/test-utils";
import { reactive } from "vue";
import { describe, expect, it } from "vitest";
import { newGenerateForm, type GenerateForm } from "../lib/generateForm";
import type { ModelEntry } from "../lib/api/types";
import MobileBatchControl from "./MobileBatchControl.vue";

function mountControl(form: GenerateForm, selectedModel: ModelEntry | null = null) {
  return mount(MobileBatchControl, {
    props: { form: reactive(form), selectedModel },
  });
}

describe("MobileBatchControl", () => {
  it("owns the general-form batch stepper and accepts direct values", async () => {
    const form = newGenerateForm();
    form.family = "flux";
    form.model = "flux-dev:fp8";
    const wrapper = mountControl(form);

    await wrapper.get("[data-test='mobile-batch-increment']").trigger("click");
    expect(form.batchSize).toBe(2);

    await wrapper.get("[data-test='mobile-batch-value']").setValue("300");
    await wrapper.get("[data-test='mobile-batch-value']").trigger("change");
    expect(form.batchSize).toBe(300);

    await wrapper.get("[data-test='mobile-batch-decrement']").trigger("click");
    expect(form.batchSize).toBe(299);

    await wrapper.get("[data-test='mobile-batch-value']").setValue("999999999999");
    await wrapper.get("[data-test='mobile-batch-value']").trigger("change");
    expect(form.batchSize).toBe(10_000);
    expect(
      wrapper.get("[data-test='mobile-batch-increment']").attributes("disabled"),
    ).toBeDefined();
  });

  it("locks edit models to one", async () => {
    const form = newGenerateForm();
    form.family = "qwen-image-edit";
    form.model = "qwen-image-edit:q4";
    form.batchSize = 4;
    const wrapper = mountControl(form, {
      name: form.model,
      family: form.family,
    } as ModelEntry);
    await flushPromises();

    expect(form.batchSize).toBe(1);
    expect(wrapper.get("[data-test='mobile-batch-value']").attributes("disabled")).toBeDefined();
    expect(wrapper.get("[data-test='mobile-batch-locked']").text()).toContain("one at a time");
  });
});
