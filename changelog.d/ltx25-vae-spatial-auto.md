### Fixed

- LTX convolutional video VAE decoding now selects upstream aspect-coupled spatial Auto tiles independently of the temporal memory estimate, avoiding an untiled decode when that estimate understates CUDA workspace. Explicit tiling overrides remain unchanged; long clips still require sufficient memory for temporal context and output accumulation.
- LTX CUDA out-of-memory messages no longer recommend the exact failed shape as fitting; smaller retry suggestions are explicitly estimates.
