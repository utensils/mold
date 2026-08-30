- **LTX-2.5 sequences no longer reset to the opening image at every clip.** A
  smooth continuation used to append the chain's opening image as a soft
  "identity anchor" token at the first frame past the motion tail. LTX-2.5 was
  trained with keyframe conditioning and reads exactly that token shape as a
  keyframe it must reach, so every clip after the first cut back to the source
  image. On keyframe-trained checkpoints the continuation now drops the
  repeated image and lets the motion-tail carry own continuity; LTX-2 and
  LTX-2.3 keep the anchor they were qualified with.
