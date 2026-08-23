- **MiniMax H3 no longer needs a manual unload after another model.** A
  generation refused for want of host memory now releases mold's own idle model
  cache least-recently-used first, re-samples, and retries admission once before
  answering. If the gap still cannot be closed, the refusal names what was given
  back — "released 9.8 GB by unloading 2 idle models; still 2.9 GB short" — so
  the number a user reads already includes every byte mold could return
  ([#1289](https://github.com/utensils/mold/issues/1289)).
