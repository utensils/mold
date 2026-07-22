/*
 * design-sync CSS carrier — pulls kit.css into the esbuild graph so
 * _ds_bundle.css ships the shared effects (shimmer, fade-up, focus ring).
 * tokens.css deliberately rides cfg.cssEntry instead: the append path copies
 * it verbatim, preserving its documentation comments for the design agent.
 */
import "../ui/kit.css";
export {};
