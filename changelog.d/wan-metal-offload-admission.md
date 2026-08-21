- **Wan memory admission no longer credits inactive block offload on Metal.**
  Oversized clips are refused before inference unless an explicit override can
  actually park every transformer block, while CUDA automatic offload keeps
  receiving the relief it can use ([#1060](https://github.com/utensils/mold/issues/1060)).
