import { effectScope } from "vue";
import { expect, it } from "vitest";
import { useMediaFileReader } from "./useMediaFileReader";

it("resets the picker and ignores read completion after the consuming scope closes", async () => {
  let complete!: (value: string) => void;
  const scope = effectScope();
  const media = scope.run(() =>
    useMediaFileReader({
      encode: () =>
        new Promise((resolve) => {
          complete = resolve;
        }),
    }),
  )!;
  const input = { files: [new File(["bytes"], "clip.mp4")], value: "clip.mp4" };
  const commits: unknown[] = [];
  const pending = media.read(
    "sourceVideo",
    { target: input } as unknown as Event,
    (value) => commits.push(value),
  );
  expect(input.value).toBe("");
  scope.stop();
  complete("Ynl0ZXM=");
  await pending;
  expect(commits).toEqual([]);
});
