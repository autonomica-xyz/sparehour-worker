# dworker video

A real, captioned dashboard demonstration, edited for length. A self-hosted
NVIDIA worker advertises selected work, claims an actual controller-dispatched
SDXL job, and returns its generated image. No simulated interface or metrics.
Model weights and ML runtimes are provisioned separately; this is not a
cold-start or throughput benchmark.

| Video | Duration | Poster | Editable captions |
| --- | --- | --- | --- |
| [dworker: compute on your terms](worker-demo.mp4) | 60 seconds | [PNG](worker-demo-poster.png) | [English SRT](worker-demo.srt) |

The demonstration shows GPU inventory, route eligibility, live Run/Stop
selection, pause/resume of new claims, GPU execution, and the delivered PNG.
The [controller demonstration and joint overview](https://github.com/autonomica-xyz/distributed-gpu/tree/main/assets/videos)
are published in the companion repository.

## Share or embed

- MP4: 1920×1080, 24 fps, H.264, browser-friendly fast start, no audio track.
- Captions are already burned into the MP4. The SRT sidecar supports
  accessibility, translation, and re-editing.
- The poster is a 1920×1080 PNG. Use a clickable poster linking to the MP4
  in GitHub Markdown; use the poster with an HTML5 video player on a website.
- No music or voiceover; the video works muted in a social feed.
- Keep the “edited for length” and separately provisioned model/runtime
  context when reusing footage; do not present the edit as a benchmark timing.

## Licenses

The worker is MIT-licensed and depends on the GPL-3.0-or-later
`distributed-gpu` package. Model weights have separate license terms and
are not included with these media assets.
