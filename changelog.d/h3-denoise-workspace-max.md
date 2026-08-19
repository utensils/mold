### Changed

- MiniMax H3 admission now charges the denoise phase's attention and FFN workspaces as the larger of the two bounds instead of their sum — the block forward runs them strictly sequentially, so the transients never coexist. Cuts the compact FL2VA predicted device peak by ~9.4 GiB on the qualified envelope, a step toward 24 GB-class (RTX 4090) admission.
