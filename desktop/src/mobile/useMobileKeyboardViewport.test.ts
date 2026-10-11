import { afterEach, expect, it, vi } from "vitest";
import { effectScope, nextTick } from "vue";
import { useMobileKeyboardViewport } from "./useMobileKeyboardViewport";

afterEach(() => {
  vi.useRealTimers();
  document.body.innerHTML = "";
});

it("owns focus listeners and cancels keyboard animation work when its scope stops", async () => {
  vi.useFakeTimers();
  const restore = vi.fn();
  const scope = effectScope();
  const keyboard = scope.run(() =>
    useMobileKeyboardViewport({ android: false, restoreNativeViewport: restore }),
  )!;
  keyboard.start();
  const input = document.createElement("input");
  document.body.append(input);
  input.focus();
  await Promise.resolve();
  await nextTick();
  expect(restore).toHaveBeenCalled();
  const count = restore.mock.calls.length;
  scope.stop();
  await vi.runAllTimersAsync();
  input.blur();
  input.focus();
  await Promise.resolve();
  expect(restore).toHaveBeenCalledTimes(count);
  expect(document.documentElement.style.getPropertyValue("--mobile-visual-viewport-page-top")).toBe(
    "",
  );
});
