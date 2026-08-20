- **`mold update` works again for older installations.** Release checksum
  manifests once again record archive names exactly as GitHub publishes them,
  without a leading `./` that Mold 0.9 and other legacy clients could not
  match. The release contract now guards every rolling and stable publication
  phase against reintroducing the incompatible path form.
