# Bounded SAM mask inspection

Observed on 2026-10-29 for issue #167: the old flattened preview loses real mask
information, and the retained teacher masks also have visible coverage gaps.
This is a six-frame diagnostic, not a segmentation benchmark or evidence of
semantic object accuracy.

## Inputs and method

- Two 64×64 frames recovered from the supplied screenshot, using pixel centers
  in crops `(152,73,252,252)` and `(1251,73,252,252)`. The five solid screenshot
  colors map unambiguously to palette indices 0, 5, 10, 11 and 12. These remain
  **screenshot-derived approximate inputs**: saved RGB values differ from the
  dashboard palette, and nearest-neighbor reconstruction differs at 61 of 63,504
  displayed pixels per crop. Original sample metadata is unavailable.
- Exact current/next arrays from the recorded `mc01-v1` object sample, batch
  9221, step 404, and `bf02-v1` prediction sample, batch 9221, step 0, seed 4147520.
  The source was `model/runs/sequences-objects-cuda-20261029/report.json`, captured
  once before analysis. No game implementation or held-out data was inspected.
- The existing `SAM2Segmenter` ran on CPU float32 with two Torch threads,
  `CUDA_VISIBLE_DEVICES=''`, capacity 32, an 8×8 uniform prompt grid and the existing
  detail prompts/filters. Six uncached calls took 31.38 seconds in total.
  This is a rerun, **not an exact replay of the training run's CUDA BF16 masks**.
  No controlled CPU/BF16 comparison was run.
- The adapter declares upstream revision
  `2b90b9f5ceec907a1c18123530e92e794ad901a4`; its checkpoint identity check verified
  SHA-256 `7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69`.
  Local source-file hashes are retained with the artifacts. Torch was 2.6.0+cu124,
  with inference and resident parameters on CPU.

Each retained binary mask was inspected independently and compared with its
surviving pixels after the existing last-mask-wins flattening. Small-region
probes are four-connected equal-color regions of 1–64 native pixels. Their best
mask IoU measures geometric agreement; these regions are **not ground-truth
objects**, and IoU ≥0.5 is only the stated diagnostic threshold.

## Results

| Frame | Retained masks | Union coverage | Fully hidden by flattening | Partly hidden | Small probes with best IoU ≥0.5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Screenshot current | 14 | 92.19% | 2 | 2 | 6/6 |
| Screenshot next | 14 | 92.21% | 2 | 2 | 6/6 |
| `mc01-v1` current | 5 | 9.91% | 1 | 1 | 2/2 |
| `mc01-v1` next | 5 | 9.91% | 1 | 1 | 2/2 |
| `bf02-v1` current | 8 | 2.98% | 3 | 0 | 7/8 |
| `bf02-v1` next | 10 | 3.76% | 6 | 0 | 9/11 |

Coverage is the union of retained masks over the whole frame. Low coverage in a
mostly uniform scene is not itself a failure: the existing filter discards masks
larger than 90% of the image.

- In the screenshot current frame, the 147- and 283-pixel masks disappear
  completely in flattening. Only 8 of 346 pixels and 63 of 380 pixels survive for
  two other masks. The next frame has the same pattern: 148/283 completely hidden,
  and 9/346 and 61/380 pixels surviving.
- **Both 25-pixel blue markers are retained exactly**, with best IoU 1.0, and all
  25 pixels remain visible in both flattened frames. Their preview colors change;
  they are not hidden or missed in this CPU rerun.
- A genuine retained-mask coverage gap remains in the screenshot: 320 current
  and 319 next pixels of the large connected black region have no mask at all,
  predominantly in its upper-left portion. The 1,325-pixel connected region's
  best single-mask IoU is 0.4943. Independent mask inspection cannot restore these
  missing proposals. This does not identify which upstream prompt/filter stage
  caused the gap.
- In `mc01-v1`, a nine-pixel yellow region has an exact mask (IoU 1.0), but its
  entire mask is overwritten in both flattened frames. A five-pixel orange mask
  is also exact, while only three/current and four/next pixels remain visible.
- In `bf02-v1`, exact two- and three-pixel masks are hidden in the current frame;
  four exact one-pixel masks are hidden in the next frame. Coarse grouping also
  exists independently of display loss: a seven-pixel red region's best current
  mask covers 28 pixels (IoU 0.25). Next-frame 28- and five-pixel probes have best
  IoUs 0.4375 and 0.1389. Such differences may reflect parts versus whole objects;
  there is no semantic ground truth here.

## Retained local evidence

Generated artifacts are ignored under `artifacts/sam-mask-inspection-167/` in the
coordination checkout: `inputs.json`, `analysis.json`, `source-file-hashes.json`,
the analysis script, three `sample-N.json` fixtures, and six overview/individual
mask figure pairs. The fixtures use the version-1 `packbits-little-base64`
instance format and round-trip against the production encoder, including mask
areas. Their predicted sets are empty and explicitly marked unavailable; no
student prediction was reconstructed or invented.

The captured report SHA-256 is
`90662ae5e8fc0c9ab9742a8b0a2026194e3810b3ea3a261e46846500265f4a73`;
the screenshot SHA-256 is
`d65dc9ec4a424ea80b6b1040cf97407b7a92932335ea0b8047ea4a2c0dd55361`.
The script consumes the captured input arrays and existing offline checkpoint;
it neither updates weights nor changes collection, losses, prompts or training
configuration. Inspecting all instances fixes visibility of retained masks, but
does not repair the teacher's uncovered regions.

The six-call analysis command, from the coordination checkout with its existing
environment and captured input file, is:

```sh
CUDA_VISIBLE_DEVICES='' MPLCONFIGDIR=/tmp/mpl-sam-inspection-167 \
  PYTHONPATH=.:src:artifacts/sam2-objects-162/deps:artifacts/sam2-objects-162/source \
  .venv/bin/python artifacts/sam-mask-inspection-167/inspect_sam_masks.py
```
