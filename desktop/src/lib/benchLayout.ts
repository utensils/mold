/*
 * The New-image workbench is a fixed-height flex column with
 * `overflow-hidden`, and it stacks two incompressible pieces: the canvas and
 * the composer. Whatever the canvas floor does not leave, `overflow-hidden`
 * eats off the BOTTOM — and the bottom is the composer, which carries the
 * only Generate button. So the floor has to be a number the smallest
 * supported window can actually give back.
 *
 * The canvas floor lives here rather than as a utility class beside the
 * markup because the view BINDS the canvas's `min-height` from it: a
 * hard-coded class beside a different constant is how the CSS came to say one
 * number while the layout reserved another and clipped the composer away.
 *
 * The scene-by-scene bench this file also used to clamp is retired — a clip
 * has one way of being made, so there is no third piece between the canvas
 * and the composer any more.
 */

/** How little canvas the workbench may be squeezed to. */
export const MIN_CANVAS_HEIGHT = 320;

/**
 * Height the composer occupies before it has been measured: one prompt row
 * plus one control row plus the card's padding.
 */
export const COMPOSER_FALLBACK_HEIGHT = 114;
