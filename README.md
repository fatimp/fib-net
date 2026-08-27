# FIB-NET

FIB-NET is an annotation-efficient binary segmentation workflow for pore-space
analysis in serial soil focused ion beam scanning electron microscopy (FIB-SEM)
images. It combines a residual 2D U-Net with neighbouring-section context,
full-resolution tiled inference, and stack-specific adaptation from a small set
of manual masks.

## Overview

The workflow is source-domain training, limited annotation of a new stack,
stack-specific model adaptation, full-stack inference, probability ensembling,
and quantitative evaluation. The manuscript study used five adaptation masks
for one 137-section target stack; this is a study-specific annotation budget,
not a claim that five masks are universally sufficient.

## Features

- Six-channel input combining adjacent sections and relief-sensitive features
- Residual U-Net with GroupNorm and residual convolution blocks
- Source-domain training and stack-specific transfer learning
- Native-resolution weighted tiled inference
- Arithmetic ensembling of unquantized float32 probability maps
- Pixel-wise segmentation and two-dimensional pore morphology metrics
- Digital-surface correlation functions, including Fss and Fsv

## Installation

FIB-NET requires Python 3.10 or newer. With [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/fatimp/fib-net.git
cd fib-net
uv sync --group dev
```

An ordinary pip installation is also supported:

```bash
python -m venv .venv
python -m pip install -e .
```

Install a CUDA-enabled PyTorch build appropriate for the local driver when GPU
training or inference is required.

## Quick Start

This smoke example generates a three-section stack and runs the complete
six-channel tiled inference path with an untrained compact model. It requires no
microscopy data or checkpoint:

```bash
uv run python examples/synthetic_stack.py
```

## Input Data

Keep microscopy data outside version control. Source-domain images and masks use
matching relative paths and normalized file stems:

```text
data/
  original/
    soil1/
      000.tif
      001.tif
  segmented/
    soil1/
      000.tif
      001.tif
```

A target stack and its sparse manual masks can use:

```text
target/
  0.tiff
  1.tiff
  ...
ideal/
  46.tif
  47.tif
  ...
```

Serial image names must sort in acquisition order. Manual masks use **black
pixels (value 0) for pores** and nonzero pixels for solid material. FIB-NET
prediction PNGs use the inverse display convention: nonzero pixels are pores.
Missing previous or next sections at stack boundaries are replaced by the
boundary section itself.

## Model Architecture

FIB-NET produces one pore logit map for the current section. Its six input
channels are the previous, current, and next normalized grayscale sections,
local contrast, gradient magnitude, and signed vertical relief. See
[docs/architecture.md](docs/architecture.md) for details.

## Training

Train the manuscript source-domain architecture on paired stacks:

```bash
uv run fibnet-train \
  --images-dir data/original \
  --masks-dir data/segmented \
  --output-dir outputs/source \
  --model-arch resunet \
  --feature-mode stack_relief \
  --train-mode patch \
  --image-size 384 \
  --patches-per-image 24 \
  --batch-size 2 \
  --epochs 24 \
  --augmentation-mode affine_noise \
  --mask-threshold 0 \
  --min-mask-fraction 1e-9 \
  --max-mask-fraction 1.0 \
  --bce-weight 0.30 \
  --dice-weight 0.40 \
  --tversky-weight 0.30 \
  --tversky-alpha 0.35 \
  --tversky-beta 0.65 \
  --seed 44 \
  --device cuda \
  --amp
```

The complete source configuration is recorded in
[configs/manuscript_source.yaml](configs/manuscript_source.yaml).

## Target-Stack Adaptation

Adapt the source model to one stack with manually segmented sections:

```bash
uv run python scripts/adapt_stack_from_few_slices.py \
  --base-checkpoint weights/fibnet_source_v0.1.pt \
  --target-dir target \
  --ideal-dir ideal \
  --output-dir outputs/target_seed142 \
  --stems 46 47 60 70 86 \
  --seed 142 \
  --device cuda \
  --amp
```

Repeat with seeds 242 and 342 to reproduce the manuscript ensemble. The indices
46, 47, 60, 70, and 86 are specific to the investigated stack. For another
stack, choose representative annotations distributed through that stack.

## Full-Stack Inference

Run each frozen target model with native-resolution tiling:

```bash
uv run fibnet-infer \
  --checkpoint weights/fibnet_target_seed142_v0.1.pt \
  --input target \
  --context-dir target \
  --output-dir outputs/seed142 \
  --feature-mode stack_relief \
  --model-arch resunet \
  --mode tile \
  --tile-size 384 \
  --overlap 96 \
  --threshold 0.60 \
  --save-probability \
  --device cuda
```

After producing analogous directories for all three seeds, average the float32
maps and threshold once:

```bash
uv run fibnet-ensemble \
  --probability-dirs outputs/seed142 outputs/seed242 outputs/seed342 \
  --output-dir outputs/ensemble \
  --threshold 0.60
```

The manuscript inference used no test-time augmentation, component filtering,
hole filling, morphological smoothing, or manual correction. The locked settings
are in [configs/manuscript_target.yaml](configs/manuscript_target.yaml).

## Model Weights

Place the separately distributed source model and three target ensemble members
under `weights/` using the names and checksums in
[weights/README.md](weights/README.md). The source model initializes adaptation;
the target models reproduce the frozen manuscript target-stack inference.

## Evaluation

Pixel overlap and two-dimensional morphology:

```bash
uv run fibnet-evaluate \
  --manual-dir ideal \
  --pred-dir outputs/ensemble \
  --manual-pore-value zero \
  --pred-pore-value nonzero \
  --output-csv outputs/metrics.csv \
  --output-json outputs/metrics.json
```

Surface correlations are computed with
[`CorrelationFunctions.jl`](https://github.com/fatimp/CorrelationFunctions.jl),
version 0.14.0, using its native 2D `Directional.surf2` and
`Directional.surfvoid` implementations. Install Julia 1.10 or newer and
instantiate the pinned bundled Julia environment once:

```bash
julia --project=fibnet/julia -e 'using Pkg; Pkg.instantiate()'
```

The manuscript calculation compares manual and predicted masks slice by slice:

```bash
uv run fibnet-correlations \
  --mask-dir outputs/ensemble \
  --pore-value nonzero \
  --manual-dir ideal \
  --manual-pore-value zero \
  --pixel-size 0.111 \
  --output-prefix outputs/correlations \
  --max-distance 64 \
  --step 2 \
  --mode slices2d
```

This path preserves every image as a native 2D array: pore/void is 0, solid is 1,
the solid phase is selected, `ConvKernel(7)` creates the convolution-based
interfacial field, and boundaries are non-periodic. It evaluates x/y directions
at 0--64 px in 2 px steps, combines them by valid-pair counts, and reports RMSE,
NRMSE (RMSE divided by the manual-curve range), and `E_CF`. The separate
`--mode volume3d` option remains available for genuine volumes; it is not the
manuscript calculation. Quantitative thresholding always uses float32 `.npy`
maps, never 8-bit preview images.

## Reproducing the Manuscript Workflow

1. Organize the externally supplied source and target data as documented above.
2. Train the source model with `configs/manuscript_source.yaml` settings.
3. Adapt three independent models on target slices 46, 47, 60, 70, and 86.
4. Infer all 137 sections with 384-pixel tiles and 96-pixel overlap.
5. Average the three probability maps and apply the fixed 0.60 threshold.
6. Evaluate against masks not used for adaptation or parameter selection.

Exact numerical reproduction additionally requires the original microscopy data
and separately distributed checkpoints.

## Repository Structure

```text
fibnet/       reusable models, features, training, inference, and metrics
scripts/      workflow and analysis command-line utilities
configs/      frozen manuscript method configurations
docs/         technical documentation
examples/     data-free runnable examples
tests/        CPU-only unit tests
weights/      checkpoint naming and checksum documentation
```

## Tests

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

## Citation

Please cite the software using [CITATION.cff](CITATION.cff).

Release-specific changes are documented in
[the v0.2.0 notes](docs/releases/v0.2.0.md).

## License

FIB-NET is released under the [MIT License](LICENSE).
