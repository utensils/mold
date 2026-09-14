- **Memory readings agree with the card in your machine.** VRAM and RAM were
  formatted with three different divisors across the apps — a 24 GiB RTX 4090
  read `25.8 GB` on the Machines meter, `24.0 GB` on the compute-plan card
  directly below it, and `23.4 GB` on the phone's host card, all spelled "GB".
  Every memory figure now comes from one authority and reads the number the
  hardware is sold as; storage and downloads keep the decimal units Hugging
  Face and drive vendors use. The legacy `/api/status` `vram_*_mb` fields are
  also read as the mebibytes the server actually sends, which had been
  under-reporting every older host by 4.86%.
- **Styles no longer clips every description.** "Good for" was pinned to a
  fixed width while the Name column absorbed all the spare space, so on a wide
  window every row read `FLUX.1 Dev BF16 …` beside a column of empty space. The
  description now grows with the window, and never starts narrower than before.
- **The Create rail stops printing its heading twice.** The source group read
  `START FROM A PHOTO` over `SOURCE`; the well's legend now stands down when
  the group is already titled and it would only repeat it, while a legend that
  tells two wells apart (First frame, Last frame, Target) still renders.
