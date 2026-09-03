<script setup lang="ts">
import { computed } from "vue";

interface FamilyIdentity {
  mark: string;
  label: string;
  tone: "cool" | "warm" | "neutral";
}

const props = defineProps<{ family: string; layout: "grid" | "table" }>();

const identity = computed<FamilyIdentity>(() => {
  const family = props.family.trim().toLowerCase();
  if (family === "flux2" || family.startsWith("flux.2")) {
    return { mark: "F2", label: "FLUX.2", tone: "cool" };
  }
  if (family.startsWith("flux")) return { mark: "F", label: "FLUX", tone: "cool" };
  if (family === "sdxl") return { mark: "XL", label: "SDXL", tone: "warm" };
  if (family === "sd15" || family.includes("1.5")) {
    return { mark: "1.5", label: "SD 1.5", tone: "warm" };
  }
  if (family.startsWith("sd3")) return { mark: "3", label: "SD 3", tone: "warm" };
  if (family.startsWith("qwen")) return { mark: "Q", label: "QWEN", tone: "neutral" };
  if (family.startsWith("zimage") || family.startsWith("z-image")) {
    return { mark: "Z", label: "Z-IMAGE", tone: "cool" };
  }
  if (family.startsWith("ltx")) return { mark: "LTX", label: "LTX VIDEO", tone: "neutral" };
  if (family.startsWith("wuerstchen")) {
    return { mark: "W", label: "WUERSTCHEN", tone: "warm" };
  }

  const label = family.replaceAll("-", " ").toUpperCase() || "MODEL";
  return { mark: label.slice(0, 3), label, tone: "neutral" };
});
</script>

<template>
  <div
    class="family-placeholder relative isolate shrink-0 overflow-hidden"
    :class="layout === 'grid' ? 'h-32 w-full rounded-t-window' : 'h-12 w-16 rounded-inner'"
    :data-tone="identity.tone"
    :data-layout="layout"
    data-test="family-placeholder"
    aria-hidden="true"
  >
    <span class="family-orbit absolute rounded-control border" aria-hidden="true" />
    <span class="family-satellite absolute rounded-full" aria-hidden="true" />
    <span
      class="family-mark font-mono text-xs absolute inset-0 flex items-center justify-center font-semibold"
      :class="layout === 'grid' ? 'text-lg' : 'text-micro'"
    >
      {{ identity.mark }}
    </span>
    <span
      class="family-label font-mono absolute text-micro"
      :class="layout === 'grid' ? 'bottom-2.5 left-3' : 'sr-only'"
    >
      {{ identity.label }}
    </span>
  </div>
</template>

<style scoped>
.family-placeholder {
  --family-primary: var(--mold-sapphire);
  --family-secondary: var(--mold-blue);
  color: var(--family-primary);
  background: color-mix(in srgb, var(--family-primary) 9%, var(--mold-bg));
}

.family-placeholder[data-tone="warm"] {
  --family-primary: var(--mold-blue);
  --family-secondary: var(--mold-sapphire);
}

.family-placeholder[data-tone="neutral"] {
  --family-primary: color-mix(in srgb, var(--mold-text) 72%, transparent);
  --family-secondary: var(--mold-sapphire);
}

.family-orbit {
  width: 58%;
  aspect-ratio: 1;
  right: -8%;
  top: -38%;
  border-color: color-mix(in srgb, var(--family-primary) 30%, transparent);
}

.family-orbit::after {
  position: absolute;
  inset: 18%;
  border: 1px solid color-mix(in srgb, var(--family-secondary) 20%, transparent);
  border-radius: 999px;
  content: "";
}

.family-satellite {
  width: 7px;
  height: 7px;
  right: 22%;
  top: 19%;
  background: var(--family-secondary);
  box-shadow: 0 0 0 4px color-mix(in srgb, var(--family-secondary) 12%, transparent);
}

.family-mark {
  color: color-mix(in srgb, var(--family-primary) 88%, var(--mold-text));
  letter-spacing: -0.04em;
}

.family-label {
  color: color-mix(in srgb, var(--family-primary) 72%, var(--mold-text));
}

.family-placeholder[data-layout="table"] .family-orbit {
  width: 42px;
  right: -14px;
  top: -20px;
}

.family-placeholder[data-layout="table"] .family-satellite {
  width: 4px;
  height: 4px;
  right: 11px;
  top: 8px;
  box-shadow: none;
}
</style>
