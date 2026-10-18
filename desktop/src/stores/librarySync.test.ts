import { beforeEach, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import { reactive, nextTick } from "vue";
const mocks = vi.hoisted(() => ({
  copy: vi.fn(),
  read: vi.fn(),
  write: vi.fn(),
  gallery: {} as any,
  hosts: {} as any,
}));
vi.mock("@studio/api/galleryCopy", async (original) => ({
  ...(await original<typeof import("@studio/api/galleryCopy")>()),
  copyGalleryPrint: mocks.copy,
}));
vi.mock("../lib/ipc", () => ({
  inTauri: () => true,
  nativeLibraryCopyHooks: () => ({}),
  ipc: { librarySyncRead: mocks.read, librarySyncWrite: mocks.write },
}));
vi.mock("./gallery", () => ({ useGalleryStore: () => mocks.gallery }));
vi.mock("./hosts", () => ({ useHostsStore: () => mocks.hosts }));
import { useLibrarySyncStore } from "./librarySync";
import { copyMachineIdentity } from "@studio/api/galleryCopy";
const rows = [
  {
    sourceKey: "source",
    item: { filename: "one.png", timestamp: 1, metadata: { prompt: "one" } },
    hostLabel: "Source",
    availableOn: [],
  },
];
const settle = async () => {
  for (let index = 0; index < 20; index++) await Promise.resolve();
  await nextTick();
};
beforeEach(() => {
  setActivePinia(createPinia());
  mocks.copy.mockReset();
  mocks.read.mockReset();
  mocks.write.mockReset();
  mocks.hosts = reactive({
    all: [
      {
        id: "source",
        label: "Source",
        status: "ready",
        instanceId: "s1",
        baseUrl: "https://source",
        apiKey: "s",
      },
      {
        id: "destination",
        label: "Destination",
        status: "ready",
        instanceId: "d1",
        baseUrl: "https://destination",
        apiKey: "d",
      },
    ],
  });
  mocks.gallery = reactive({
    merged: rows,
    buckets: {},
    collectionsByHost: {},
    targetOf: (id: string) => {
      const host = mocks.hosts.all.find((host: any) => host.id === id);
      return host ? { baseUrl: host.baseUrl, apiKey: host.apiKey } : null;
    },
    fetchAll: vi.fn(async () => undefined),
  });
  let document: unknown = null;
  mocks.read.mockImplementation(async () => document);
  mocks.write.mockImplementation(async (next) => {
    document = structuredClone(next);
  });
  mocks.copy.mockResolvedValue({ stage: "complete" });
});
it("persists Keep by installation and never grants it to a replacement destination", async () => {
  const sync = useLibrarySyncStore();
  await sync.initialize();
  const destination = sync.machines.find((machine) => machine.id === "destination")!;
  await sync.setKeep(destination, true);
  await settle();
  expect(sync.keep[copyMachineIdentity(destination)]).toBe(true);
  mocks.hosts.all[1].instanceId = "replaced";
  await nextTick();
  const replacement = sync.machines.find((machine) => machine.id === "destination")!;
  expect(sync.keep[copyMachineIdentity(replacement)]).not.toBe(true);
  sync.stop();
});
it("serializes manual selection copies and Stop clears follow-ups", async () => {
  const sync = useLibrarySyncStore();
  await sync.initialize();
  let finish!: () => void;
  mocks.copy.mockImplementationOnce(
    () =>
      new Promise((resolve) => {
        finish = () => resolve({ stage: "complete" });
      }),
  );
  const destination = sync.machines[1]!;
  sync.copyTo(rows as never, destination);
  sync.copyTo(rows as never, destination);
  await settle();
  expect(mocks.copy).toHaveBeenCalledTimes(1);
  const signal = mocks.copy.mock.calls[0]![0].signal as AbortSignal;
  sync.stop();
  expect(signal.aborted).toBe(true);
  finish();
  await settle();
  expect(mocks.copy).toHaveBeenCalledTimes(1);
  expect(sync.busy).toBe(false);
});
it("dismissal survives ordinary progress and Stop remains independent", async () => {
  const sync = useLibrarySyncStore();
  await sync.initialize();
  let finish!: () => void;
  mocks.copy.mockImplementationOnce(
    () =>
      new Promise((resolve) => {
        finish = () => resolve({ stage: "complete" });
      }),
  );
  sync.copyTo(rows as never, sync.machines[1]!);
  await settle();
  sync.dismissed = true;
  mocks.copy.mock.calls[0]![0].progress("sources");
  expect(sync.dismissed).toBe(true);
  expect(sync.busy).toBe(true);
  sync.stop();
  finish();
  await settle();
  expect(sync.busy).toBe(false);
});
it("automatic repeats skip already verified unchanged rows but manual Copy remains deliberate", async () => {
  const sync = useLibrarySyncStore();
  await sync.initialize();
  const destination = sync.machines[1]!;
  sync.syncAll(destination, true);
  await settle();
  expect(mocks.copy).toHaveBeenCalledTimes(1);
  sync.syncAll(destination, true);
  await settle();
  expect(mocks.copy).toHaveBeenCalledTimes(1);
  sync.copyTo(rows as never, destination);
  await settle();
  expect(mocks.copy).toHaveBeenCalledTimes(2);
  sync.stop();
});

it("Sync All considers every physical source even when display names group different bytes", async () => {
  mocks.hosts.all.push({
    id: "other-source",
    label: "Other",
    status: "ready",
    instanceId: "other-installation",
    baseUrl: "https://other-source",
    apiKey: "other-key",
  });
  mocks.gallery.merged = [
    {
      ...rows[0],
      copies: [
        { sourceKey: "source", item: { ...rows[0]!.item, media_version: "bytes-a" } },
        { sourceKey: "other-source", item: { ...rows[0]!.item, media_version: "bytes-b" } },
      ],
    },
  ];
  const sync = useLibrarySyncStore();
  await sync.initialize();
  sync.syncAll(sync.machines.find((machine) => machine.id === "destination")!);
  await settle();
  expect(mocks.copy.mock.calls.map(([options]) => options.source.id)).toEqual([
    "source",
    "other-source",
  ]);
  expect(mocks.copy.mock.calls.every(([options]) => options.filename === "one.png")).toBe(true);
  sync.stop();
});
