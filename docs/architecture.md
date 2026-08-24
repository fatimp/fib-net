# FIB-NET architecture

FIB-NET is a residual two-dimensional U-Net that predicts a binary pore logit
for every pixel in the current FIB-SEM section. The model processes sections in
their native acquisition plane; neighbouring sections provide contextual input
but the network output remains two-dimensional.

## Six-channel stack-relief representation

For section `i`, the input tensor contains:

1. normalized grayscale section `i - 1`;
2. normalized grayscale section `i`;
3. normalized grayscale section `i + 1`;
4. local contrast from the difference between Gaussian-blurred current images;
5. current-image gradient magnitude;
6. signed vertical current-image gradient as a relief-sensitive cue.

Grayscale values are scaled to `[0, 1]` and normalized with mean 0.5 and
standard deviation 0.5. Derived channels are clipped to stable ranges. At the
first and last stack positions, the absent neighbour is replaced by the boundary
section itself.

## Residual U-Net

The encoder has four feature levels with widths 32, 64, 128, and 256, followed
by a 512-channel bottleneck. Each residual block contains two 3 x 3 convolutions,
GroupNorm, and SiLU activations. A 1 x 1 projection is used when the residual and
main paths have different channel counts. Max pooling performs downsampling;
transposed convolutions and encoder skip connections form the decoder. A final
1 x 1 convolution emits one pore logit channel.

Training uses a weighted sum of binary cross-entropy, soft Dice, and Tversky
losses. The manuscript target models used weights 0.10, 0.65, and 0.25,
respectively, with Tversky alpha 0.35 and beta 0.65.

## Full-resolution inference

Large sections are divided into overlapping 384 x 384 tiles. Tile probabilities
are blended with identical spatial weights in the numerator and denominator,
which preserves probabilities at image borders and overlap regions. Three
independently adapted models produce float32 probability maps. Their arithmetic
mean is thresholded at 0.60 to obtain the manuscript binary segmentation.

No test-time augmentation or morphological post-processing is part of the final
manuscript target inference protocol.
