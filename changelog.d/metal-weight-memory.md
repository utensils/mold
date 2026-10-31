- **Repeated renders on a Mac no longer run out of Metal memory.** A second
  render could fail with `kIOGPUCommandBufferCallbackErrorOutOfMemory`, and later
  requests were refused with "0.00 GB usable" until the device was toggled. Mold
  now loads half-precision weights at their exact size and releases memory after
  a failed render, so no unload or toggle is needed. On a 128 GiB M5 Max, Z-Image
  peaks at 25.1 GiB instead of 89.7 GiB and renders in 88 s and 70 s instead of
  120 s and 98 s; FLUX.2 Klein 9B peaks at 53 GiB instead of 86 GiB. Pixels are
  unchanged at the same seed and prompt. The Metal budget is still shared with
  other GPU apps.
