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
    expect(dialog.textContent).toContain("Choose where to install Flux Dev");
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
    expect(dialog.textContent).toContain("Choose where to install Flux Dev");
    expect(
      document.body.querySelector('[data-test="download-target-local"]')?.textContent,
    ).toContain("Install");
    expect(
      document.body.querySelector('[data-test="download-target-studio-7680"]')?.textContent,
    ).toContain("Already installed");
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
    expect(document.activeElement?.getAttribute("aria-label")).toBe("Close install target picker");

    wrapper.unmount();
    expect(document.activeElement).toBe(opener);
  });

  it("uses repair-specific confirmation copy", () => {
    const wrapper = mount(DownloadTargetDialog, {
      props: { modelName: "Flux Dev", targets: repair(hosts) },
      attachTo: document.body,
    });

    const dialog = document.body.querySelector<HTMLElement>("[role='dialog']")!;
    expect(dialog.textContent).toContain("Choose where to repair Flux Dev");
    expect(dialog.textContent).toContain(
      "Only missing or damaged files will be fetched on the selected host.",
    );
    expect(
      document.body
        .querySelector<HTMLButtonElement>("[aria-label='Close repair target picker']")
        ?.getAttribute("aria-label"),
    ).toBe("Close repair target picker");
    wrapper.unmount();
  });
});
