/*
 * Storage key for the iPhone app's local generation templates.
 *
 * It lives in a plain module rather than inside `MobileTemplates.vue` because
 * a `.vue` file's named exports are only visible to a toolchain whose Vue
 * language plugin resolves the SFC. The apps also compile against a
 * `declare module "*.vue"` shim that exposes the default export alone, so a
 * consumer importing this name from the component type-checks against a warm
 * incremental cache and fails in a clean sandbox build.
 */
export const MOBILE_GENERATION_TEMPLATES_STORAGE_KEY = "mold.mobile.generation.templates.v1";
