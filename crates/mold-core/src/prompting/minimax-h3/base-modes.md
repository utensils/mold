# MiniMax H3 base modes

Use for FL2VA identities. Text-only generation describes the whole shot in the
three-section H3 grammar. For first-frame or first/last-frame conditioning,
anchor `<Picture 1>` at 0.00 seconds, preserve identity, clothing, layout,
lighting, and composition, then describe a continuous observable path forward.
Do not contradict the supplied frame.

```text
For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.

integrated_multimodal_description: [Shot 1] Live-action, cinematic, the presenter shown in <Picture 1> retains the exact face, clothing, workstation layout, lighting, and framing. The camera trucks right slowly as the presenter turns toward the lens. The presenter with a bright voice and relaxed pace (S1) says: <d>[English] With Mold, your ideas render right here.</d> Their lips synchronize; they gesture once, then hold a natural final expression.

overall_soundscape: Low workstation airflow continues beneath clean close-miked speech. One quiet interface chime sounds after the final word.

non_diegetic_music: N/A
```
