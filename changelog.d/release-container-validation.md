- **Validate containers before tagging releases.** Install the Protobuf compiler
  required by mesh matting in CUDA images, build real images on relevant pull
  requests and all six GPU targets for release candidates, and preserve sibling
  build results when one target fails. Retire FlakeHub publication and its README
  installation reference; GitHub releases and Nix remain available.
