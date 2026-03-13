// FLUX inference pipeline — implemented via candle_transformers::models::flux.
//
// The actual model loading, sampling, and VAE decoding is handled by the
// FluxEngine in engine.rs using candle's built-in FLUX support:
//
//   - flux::model::Flux — the transformer
//   - flux::sampling — noise generation, State, get_schedule, denoise, unpack
//   - flux::autoencoder::AutoEncoder — VAE decoder
//   - t5::T5EncoderModel — text encoder
//   - clip::text_model::ClipTextTransformer — CLIP encoder
