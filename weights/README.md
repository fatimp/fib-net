# Model weights

Release checkpoints are distributed separately from the source repository. Place
them in this directory using these stable names:

| File | Role | SHA256 of manuscript checkpoint |
| --- | --- | --- |
| `fibnet_source_v0.1.pt` | Source-domain model used to initialize adaptation | `7aac292ec85bed362b9c56ef0a8f4bbdfc17ecfa1e750837e9ebecbe9dac078f` |
| `fibnet_target_seed142_v0.1.pt` | Target ensemble member, seed 142 | `47784a2cb3cab2380fe101dab3e1eed3e016ca875239542531d01cb9b566a4c1` |
| `fibnet_target_seed242_v0.1.pt` | Target ensemble member, seed 242 | `e8b8666a2cf045eeb2740cadb9f3d5e4c1a4dcf8247f212a4752ff513b0431cf` |
| `fibnet_target_seed342_v0.1.pt` | Target ensemble member, seed 342 | `dfc8041c869a32d4c62abe0e47f98081eeb00489d11a3722a9b0ec7d647d6541` |

The source checkpoint is a pretrained starting point for a new stack. The three
target checkpoints are the frozen ensemble used for the manuscript target-stack
inference. Checkpoint binaries are not bundled with the Git repository.
