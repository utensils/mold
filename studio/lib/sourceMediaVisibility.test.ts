import { mount, flushPromises } from "@vue/test-utils";
import { defineComponent, ref } from "vue";
import { expect, it } from "vitest";
import { useSourceMediaVisibility } from "./sourceMediaVisibility";
import {
  cachedSourceInventory,
  cacheSourceInventory,
} from "./sourceMediaResources";
import {
  sourceOwnerIdentity,
  type SourceMediaOwner,
} from "./sourceMediaBrowser";
it("invalidates hidden and replaced owners while the browser is unmounted", async () => {
  const a: SourceMediaOwner = {
    key: "a",
    filename: "a.png",
    title: "a",
    hostLabel: "Host",
    target: { baseUrl: "http://a", apiKey: "old" },
  };
  const owners = ref([a]);
  const key = sourceOwnerIdentity(a);
  const view = mount(
    defineComponent({
      setup() {
        useSourceMediaVisibility(() => owners.value);
        return () => null;
      },
    }),
  );
  cacheSourceInventory(key, { availability: "available", members: [] });
  owners.value = [{ ...a, target: { ...a.target, apiKey: "new" } }];
  await flushPromises();
  expect(cachedSourceInventory(key)).toBeUndefined();
  const replacement = sourceOwnerIdentity(owners.value[0]!);
  cacheSourceInventory(replacement, { availability: "available", members: [] });
  owners.value = [];
  await flushPromises();
  expect(cachedSourceInventory(replacement)).toBeUndefined();
  view.unmount();
});
