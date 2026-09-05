import { flushPromises, mount } from "@vue/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { newGenerateForm } from "../../lib/generateForm";
import { fetchLoras } from "../../lib/api/loras";
import LoraStack from "./LoraStack.vue";

vi.mock("../../lib/api/loras", () => ({ fetchLoras: vi.fn() }));

const route = {
  hostId: "plato-7680",
  label: "plato",
  kind: "remote" as const,
  target: { baseUrl: "http://plato:7680", apiKey: null },
  instanceId: "plato-instance",
};

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => (resolve = done));
  return { promise, resolve };
}

describe("LoraStack host routing", () => {
  beforeEach(() => vi.mocked(fetchLoras).mockReset());

  it("loads and binds LoRAs from the machine that will generate", async () => {
    const form = newGenerateForm();
    form.model = "z-image-turbo:q8";
    form.family = "z-image";
    vi.mocked(fetchLoras).mockResolvedValue([
      {
        id: "alien",
        name: "Alien Xenomorph",
        path: "/storage/mold/models/alien.safetensors",
        family: "z-image",
        trained_words: ["alien"],
        added_at: 1,
      },
    ]);
    const wrapper = mount(LoraStack, { props: { form, model: form.model, route } });

    const addButton = wrapper.findAll("button").find((button) => button.text().trim() === "Add");
    expect(addButton).toBeDefined();
    await addButton!.trigger("click");
    await flushPromises();

    expect(fetchLoras).toHaveBeenCalledWith(form.model, route.target);
    await wrapper.get("[data-test='lora-picker'] button").trigger("click");
    expect(form.loras).toEqual([
      {
        path: "/storage/mold/models/alien.safetensors",
        name: "Alien Xenomorph",
        scale: 1,
        trainedWords: ["alien"],
        hostId: "plato-7680",
        hostBaseUrl: "http://plato:7680",
        hostInstanceId: "plato-instance",
      },
    ]);
  });

  it("discards a stale LoRA listing when the route changes", async () => {
    const form = newGenerateForm();
    form.model = "z-image-turbo:q8";
    form.family = "z-image";
    const first = deferred<Awaited<ReturnType<typeof fetchLoras>>>();
    const second = deferred<Awaited<ReturnType<typeof fetchLoras>>>();
    vi.mocked(fetchLoras).mockReturnValueOnce(first.promise).mockReturnValueOnce(second.promise);
    const wrapper = mount(LoraStack, { props: { form, model: form.model, route } });

    await wrapper
      .findAll("button")
      .find((button) => button.text().trim() === "Add")!
      .trigger("click");
    const halRoute = {
      ...route,
      hostId: "hal-7680",
      label: "hal",
      target: { baseUrl: "http://hal:7680", apiKey: null },
      instanceId: "hal-instance",
    };
    await wrapper.setProps({ route: halRoute });
    await wrapper
      .findAll("button")
      .find((button) => button.text().trim() === "Add")!
      .trigger("click");
    second.resolve([
      {
        id: "hal-lora",
        name: "HAL LoRA",
        path: "/hal/lora.safetensors",
        family: "z-image",
        trained_words: [],
        added_at: 2,
      },
    ]);
    await flushPromises();
    first.resolve([
      {
        id: "plato-lora",
        name: "Plato LoRA",
        path: "/plato/lora.safetensors",
        family: "z-image",
        trained_words: [],
        added_at: 1,
      },
    ]);
    await flushPromises();

    expect(wrapper.get("[data-test='lora-picker']").text()).toContain("HAL LoRA");
    expect(wrapper.get("[data-test='lora-picker']").text()).not.toContain("Plato LoRA");
    await wrapper.get("[data-test='lora-picker'] button").trigger("click");
    expect(form.loras[0]).toMatchObject({
      path: "/hal/lora.safetensors",
      hostId: "hal-7680",
      hostBaseUrl: "http://hal:7680",
      hostInstanceId: "hal-instance",
    });
  });

  it("discards a stale listing when a host ID now points at another server instance", async () => {
    const form = newGenerateForm();
    form.model = "z-image-turbo:q8";
    form.family = "z-image";
    const first = deferred<Awaited<ReturnType<typeof fetchLoras>>>();
    const second = deferred<Awaited<ReturnType<typeof fetchLoras>>>();
    vi.mocked(fetchLoras).mockReturnValueOnce(first.promise).mockReturnValueOnce(second.promise);
    const wrapper = mount(LoraStack, { props: { form, model: form.model, route } });

    await wrapper
      .findAll("button")
      .find((button) => button.text().trim() === "Add")!
      .trigger("click");
    const replacementRoute = { ...route, instanceId: "replacement-instance" };
    await wrapper.setProps({ route: replacementRoute });
    await wrapper
      .findAll("button")
      .find((button) => button.text().trim() === "Add")!
      .trigger("click");
    second.resolve([
      {
        id: "new-lora",
        name: "Replacement LoRA",
        path: "/replacement/lora.safetensors",
        family: "z-image",
        trained_words: [],
        added_at: 2,
      },
    ]);
    await flushPromises();
    first.resolve([
      {
        id: "old-lora",
        name: "Old Instance LoRA",
        path: "/old/lora.safetensors",
        family: "z-image",
        trained_words: [],
        added_at: 1,
      },
    ]);
    await flushPromises();

    expect(wrapper.get("[data-test='lora-picker']").text()).toContain("Replacement LoRA");
    expect(wrapper.get("[data-test='lora-picker']").text()).not.toContain("Old Instance LoRA");
    await wrapper.get("[data-test='lora-picker'] button").trigger("click");
    expect(form.loras[0]).toMatchObject({
      path: "/replacement/lora.safetensors",
      hostId: "plato-7680",
      hostBaseUrl: "http://plato:7680",
      hostInstanceId: "replacement-instance",
    });
  });
});
