import { mount } from "@vue/test-utils";
import { afterEach, describe, expect, it } from "vitest";
import DownloadTargetDialog from "./DownloadTargetDialog.vue";
import type { HostView } from "../../stores/hosts";

const hosts: HostView[] = [
  {
    id: "local",
    label: "This Mac",
    kind: "local",
    baseUrl: "http://127.0.0.1:7680",
    apiKey: "local-key",
    status: "ready",
    primary: true,
    queueDepth: 0,
    queueCapacity: 2,
    version: "0.18.0",
  },
  {
    id: "studio-7680",
    label: "Studio GPU",
    kind: "remote",
    baseUrl: "http://studio:7680",
    apiKey: "remote-key",
    status: "ready",
    primary: false,
    queueDepth: 1,
    queueCapacity: 4,
    version: "0.18.0",
  },
];

describe("DownloadTargetDialog", () => {
  afterEach(() => {
    document.body.innerHTML = "";
  });

  it("offers every ready host and returns the selected target", async () => {
    const wrapper = mount(DownloadTargetDialog, {
      props: { modelName: "Flux Dev", hosts },
      attachTo: document.body,
    });

    const dialog = document.body.querySelector<HTMLElement>("[role='dialog']")!;
    expect(dialog.getAttribute("aria-modal")).toBe("true");
    expect(dialog.textContent).toContain("Choose where to download Flux Dev");
    expect(dialog.textContent).toContain("This Mac");
    expect(dialog.textContent).toContain("Studio GPU");

    (
      document.body.querySelector('[data-test="download-target-studio-7680"]') as HTMLElement
    ).click();
    expect(wrapper.emitted("select")?.[0]).toEqual([hosts[1]]);
    wrapper.unmount();
  });

  it("moves focus into the dialog and restores it when closed", async () => {
    const opener = document.createElement("button");
    document.body.appendChild(opener);
    opener.focus();
    const wrapper = mount(DownloadTargetDialog, {
      props: { modelName: "Flux Dev", hosts },
      attachTo: document.body,
    });

    await wrapper.vm.$nextTick();
    expect(document.activeElement?.getAttribute("aria-label")).toBe("Close download target picker");

    wrapper.unmount();
    expect(document.activeElement).toBe(opener);
  });
});
