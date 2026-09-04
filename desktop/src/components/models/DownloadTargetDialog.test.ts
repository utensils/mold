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
    instanceId: null,
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
    instanceId: null,
  },
];

/** Every listed machine is missing the model — the plain first-download case. */
function install(list: HostView[]) {
  return list.map((host) => ({ host, action: "install" as const }));
}

/** Every listed machine already owns the model. */
function repair(list: HostView[]) {
  return list.map((host) => ({ host, action: "repair" as const }));
}

describe("DownloadTargetDialog", () => {
  afterEach(() => {
    document.body.innerHTML = "";
  });

  it("offers every ready host and returns the selected target", async () => {
    const wrapper = mount(DownloadTargetDialog, {
      props: { modelName: "Flux Dev", targets: install(hosts) },
      attachTo: document.body,
    });

    const dialog = document.body.querySelector<HTMLElement>("[role='dialog']")!;
    expect(dialog.getAttribute("aria-modal")).toBe("true");
    expect(dialog.textContent).toContain("Where should Flux Dev go?");
    expect(dialog.textContent).toContain("This Mac");
    expect(dialog.textContent).toContain("Studio GPU");

    (
      document.body.querySelector('[data-test="download-target-studio-7680"]') as HTMLElement
    ).click();
    expect(wrapper.emitted("select")?.[0]).toEqual([hosts[1]]);
    wrapper.unmount();
  });

  it("says per machine whether this installs or repairs", () => {
    // The whole point of a mixed list: one machine is missing the model, the
    // other already has it, and the user must be able to tell them apart.
    const wrapper = mount(DownloadTargetDialog, {
      props: {
        modelName: "Flux Dev",
        targets: [
          { host: hosts[0]!, action: "install" as const },
          { host: hosts[1]!, action: "repair" as const },
        ],
      },
      attachTo: document.body,
    });

    const dialog = document.body.querySelector<HTMLElement>("[role='dialog']")!;
    expect(dialog.textContent).toContain("Where should Flux Dev go?");
    expect(
      document.body.querySelector('[data-test="download-target-local"]')?.textContent,
    ).toContain("Get it");
    expect(
      document.body.querySelector('[data-test="download-target-studio-7680"]')?.textContent,
    ).toContain("Already here · repair");
    // The body copy must not promise a fresh install for the whole list when
    // one of the machines on it can only be repaired.
    expect(dialog.textContent).toContain("machines that already have it are repaired instead");
    wrapper.unmount();
  });

  it("promises only an install when no listed machine already has the model", () => {
    const wrapper = mount(DownloadTargetDialog, {
      props: { modelName: "Flux Dev", targets: install(hosts) },
      attachTo: document.body,
    });

    const dialog = document.body.querySelector<HTMLElement>("[role='dialog']")!;
    expect(dialog.textContent).toContain(
      "The style and everything it needs are kept on the machine you pick.",
    );
    expect(dialog.textContent).not.toContain("repaired instead");
    wrapper.unmount();
  });

  it("moves focus into the dialog and restores it when closed", async () => {
    const opener = document.createElement("button");
    document.body.appendChild(opener);
    opener.focus();
    const wrapper = mount(DownloadTargetDialog, {
      props: { modelName: "Flux Dev", targets: install(hosts) },
      attachTo: document.body,
    });

    await wrapper.vm.$nextTick();
    expect(document.activeElement).toBe(document.body.querySelector("[role='dialog']"));

    wrapper.unmount();
    expect(document.activeElement).toBe(opener);
  });

  it("uses repair-specific confirmation copy", () => {
    const wrapper = mount(DownloadTargetDialog, {
      props: { modelName: "Flux Dev", targets: repair(hosts) },
      attachTo: document.body,
    });

    const dialog = document.body.querySelector<HTMLElement>("[role='dialog']")!;
    expect(dialog.textContent).toContain("Which machine should repair Flux Dev?");
    expect(dialog.textContent).toContain(
      "Only the missing or damaged files are fetched on the machine you pick.",
    );
    document.body.querySelector<HTMLButtonElement>('[data-test="download-target-cancel"]')!.click();
    expect(wrapper.emitted("close")).toHaveLength(1);
    wrapper.unmount();
  });
});
