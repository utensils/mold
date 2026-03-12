pub mod scheduler;
pub mod tokenizer;

// FLUX model loading and sampling.
//
// This module will implement the full FLUX diffusion pipeline using candle.
// Reference: candle/candle-examples/examples/flux/
//
// Pipeline stages:
//   1. Encode prompt via T5 + CLIP tokenizers
//   2. Create latent noise tensor
//   3. Run denoising loop (Euler scheduler)
//   4. Decode latents through VAE
//   5. Convert to RGB image
//
// For now, actual inference is stubbed in engine.rs.
