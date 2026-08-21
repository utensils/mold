### Fixed

- Fixed the MiniMax H3 private runtime failing to compile against the new `GenerateResponse.request_warnings` field, which broke every `h3`-featured release build; CI now compiles the shipping `h3` feature so a shared response-type change cannot land without it.
