# Durumari

An experimental lossless data compression research project exploring finite-field Fourier transforms, exponential prediction models, and byte predictors.

The project investigates whether algebraic structure in binary data can provide compact, exactly reversible representations. It also includes DCT reconstruction experiments and comparisons with reciprocal-derived real orthogonal bases.

**This is a research codebase, not a complete file compression tool.** The tracked sources contain transform checks, predictor implementations, and payload-cost estimators. A unified file format and an end-to-end file encoder/decoder are not yet available. Compression gains are hypotheses to test, not assumed properties of the transforms.

The repository layout, implementation descriptions, and usage guidance below cover files tracked in Git. The Research progress section also summarizes local exploratory work, including experiments excluded from Git. Those observations describe research progress and do not imply that the corresponding implementations or inputs are available in a checkout.

## Research approach

The finite-field experiments use:

- Prime modulus: p = 257
- Field: F_257
- Default block length: 256 bytes
- Input byte values: 0..255
- Field element values: 0..256

Because 257 - 1 = 256, the nonzero field elements provide a natural basis family for 256-element blocks.

### Finite-field Fourier / Vandermonde transform

For a = 1..256 and n = 0..255, the synthesis and analysis matrices are:

    V[a - 1, n] = a^n mod 257
    A[a - 1, n] = -a^(-n) mod 257

    c = A @ x mod 257
    x = V.T @ c mod 257

The scripts in [archive/main.py](archive/main.py) and [archive/main_whole_file_test.py](archive/main_whole_file_test.py) construct these matrices and include assertions checking reconstruction in both directions on the first input block.

Invertibility establishes information preservation. It does not establish compression: the model, coefficients, residuals, and metadata must together cost less than the original representation.

Including a Fourier component at least once is a design requirement for the proposed compression path. RAW and simpler predictors remain useful experimental controls. Their results help distinguish the contribution of Fourier modeling from improvements due to other components.

## Repository layout

| Path | Purpose |
| --- | --- |
| [archive/main.py](archive/main.py) | First-block Fourier checks and exhaustive single-exponential model search, minimizing estimated Huffman residual payload. |
| [archive/main_whole_file_test.py](archive/main_whole_file_test.py) | Whole-file comparison of raw-byte and finite-field-transform Huffman payload costs. |
| [archive/bases.py](archive/bases.py) | Mod-256 predictor implementations and a file-level size estimator. |
| [codecs/const.py](codecs/const.py) | Standalone Constant encoder returning a list of integer fields. |
| [main.py](main.py) | DCT coefficient rounding and reconstruction experiment in 65,536-byte chunks; not a compression CLI. |
| [which percentile.py](which%20percentile.py) | DCT reconstruction checks and coefficient-rank exploration. |
| [archive/reciprocal_basis_experiment.py](archive/reciprocal_basis_experiment.py) | Reciprocal-derived real basis construction, SVD orthogonalization, and DFT/DCT comparisons. |
| [archive/reciprocal_basis_experiment_output.txt](archive/reciprocal_basis_experiment_output.txt) | Saved diagnostics and comparison metrics from reciprocal-basis experiments. |
| [archive/constant_basis test.py](archive/constant_basis%20test.py) | Example invocation of the Constant predictor. |
| [archive/ndarray tests.py](archive/ndarray%20tests.py) | NumPy concatenation, reshaping, and flattening experiment. |
| [archive/unary even sine test.py](archive/unary%20even%20sine%20test.py) | Periodic waveform generation and pairwise orthogonality assertions. |
| [archive/unary staccato sine test.py](archive/unary%20staccato%20sine%20test.py) | Sparse periodic waveform generation and pairwise orthogonality assertions. |
| [n-errors-by-type&norm.png](n-errors-by-type%26norm.png) | Transform type/normalization comparison figure. |

## Implemented predictor models

[archive/bases.py](archive/bases.py) implements the following representations for blocks of at most 256 bytes:

| Model | Function | Representation |
| --- | --- | --- |
| RAW | raw_basis() | Original bytes, with an assumed one-byte mode cost. |
| Constant | constant_basis() | Most frequent byte plus nonzero mod-256 residuals. |
| Delta | delta_basis() | First byte plus nonzero mod-256 first differences. |
| Delta2 | delta2_basis() | First two bytes plus residuals from the prediction 2*x[n-1] - x[n-2]. |
| Periodic | periodic_basis() | Period, initial seed, and residuals relative to x[n-T]. |

Supporting functions and types:

- BasisResult holds the mode name, parameter bytes, residual bytes, estimated cost, and residual count.
- _pack_sparse_residual() stores each nonzero residual as a one-byte position and one-byte value.
- periodic_basis() considers periods 2, 3, 4, 5, 6, 7, 8, 16, and 32 by default.
- best_basis() compares the five models by estimated cost and favors RAW on a tie.
- analyze_file() accumulates model-selection counts and estimated costs over complete and short final blocks.
- run_sanity_tests() prints costs for a single-outlier block, a ramp, and a period-four example. It does not perform decoder round-trip assertions.

The separate encode() in [codecs/const.py](codecs/const.py) stores a method identifier, the most frequent byte, an exception count, and position/original-value pairs. Its format differs from the sparse residual representation in bases.py. It has no matching decoder in the tracked sources.

### Single-exponential Fourier model search

[archive/main.py](archive/main.py) models the first 256-byte block as:

    predicted[n] = c * a^n mod 257
    residual[n] = (input[n] - predicted[n]) mod 257

It searches a = 2..256 and c = 1..256, totaling 65,280 candidates. The constant basis and zero coefficient are excluded. The winning candidate minimizes the Huffman payload estimate returned by huffman_bits().

This search does not include the storage cost of the basis, coefficient, codebook, or headers. It is a residual-distribution experiment, not a complete Fourier codec.

## Whole-file Huffman experiment

[archive/main_whole_file_test.py](archive/main_whole_file_test.py) compares:

1. A global Huffman model for the original bytes.
2. A global Huffman model for finite-field Fourier coefficients from complete 256-byte blocks.

Blocks are processed in batches using:

    coefficients = (blocks @ A.T) mod 257

The final short block is left as raw symbols and added to the transformed-stream histogram. huffman_bits_from_counts() computes the optimal binary Huffman payload length from frequency merge costs.

The script reports payload bits, rounded-up payload bytes, ratios, and symbol counts. It does not generate compressed bitstreams or include codebooks, original-length metadata, or file headers. The tracked sources do not contain a saved execution log for this experiment.

## DCT experiments

[main.py](main.py) and [which percentile.py](which%20percentile.py) read signed byte samples in 65,536-byte chunks, compute a DCT, round coefficients to integers, apply the inverse transform, and count mismatching samples.

Both scripts repeat a short final chunk to fill the working buffer. They do not define a file format for recovering the original length. In main.py, p00s remains empty before its mean is printed. In which percentile.py, the coefficient-rank search locates approximately the lower quartile of signed coefficient values; it does not search for a reconstruction-error threshold.

Rounding real-valued transform coefficients can introduce reconstruction error. A lossless construction would need to preserve any residual and all parameters required for decoding. These scripts do not implement that complete path.

The tracked comparison figure illustrates transform settings, but is not a complete compressed-size benchmark. Coefficient rounding under different normalizations corresponds to different effective quantization steps.

## Reciprocal-derived orthogonal basis

[archive/reciprocal_basis_experiment.py](archive/reciprocal_basis_experiment.py) explores an alternative real-valued basis:

- signed_inverse_reciprocal() maps a nonzero field element to the real reciprocal of its centered modular inverse.
- build_reciprocal_matrix() applies this mapping to finite-field exponential sequences.
- polar_orthogonalize() uses the SVD polar factor to obtain a real orthogonal matrix.
- build_real_dft() and build_dct2() supply comparison bases.
- transform_metrics() measures energy concentration, rounded-coefficient statistics, estimated Huffman payload, and reconstruction RMSE.

The saved [output](archive/reciprocal_basis_experiment_output.txt) reports a full-rank matrix and a maximum orthogonality error of approximately 1.67e-15. Its text and decoded-grayscale comparisons generally show weaker energy concentration than DFT/DCT, while specially constructed synthetic signals can favor the custom basis.

These are transform metrics, not complete lossless compression ratios. The experiment removes block means, discards incomplete blocks, and rounds coefficients without including the cost of means, residuals, or codebooks.

The saved output contains results from an earlier input configuration and is not an exact record of the script's current input list.

## Research progress

This section summarizes the research beyond the tracked source inventory. Local exploratory records, including Git-ignored experiments, are used only to describe progress here. Saved outputs were inspected rather than rerun for this update. Their sizes have different accounting boundaries and are not directly interchangeable compressed-file measurements.

### Finite-field models and exact recurrences

Local experiments record successful matrix-identity and reconstruction checks for the finite-field transform. A synthetic signal built from four exponential bases has four nonzero transform coefficients, confirming the expected representation for a matching signal.

An exact linear-recurrence prototype using Berlekamp–Massey and 9-bit field-element packing has also been explored:

| 256-byte input | Recurrence order | Prototype representation size |
| --- | ---: | ---: |
| Repeated ABCD pattern | 4 | 12 B |
| All zeros | 0 | 3 B |
| Seeded random bytes | 128 | 291 B |

The byte examples include saved round-trip assertions. These successes demonstrate compact representations of particular structured inputs, not general compression performance. The recurrence prototype is local exploratory work, not part of the tracked implementation documented above, and an independent recurrence-only result does not establish a benefit from Fourier modeling.

A local comparison on the first 256 bytes of a ZIP archive selected RAW at 257 fields over Periodic at 267, Linear at 291, Constant at 311, Fourier1 at 312, Delta at 366, and Delta2 at 383. The competition compares returned-list lengths: Linear has explicit byte packing, while Fourier1 still has an unresolved field-element-256 serialization case. The header-rich sample does not characterize the full archive.

The research handover also reports a historical whole-file comparison on 26,246,026 bytes: original-byte Huffman payload of 209,625,002 bits versus 210,170,179 bits after the finite-field transform. The tracked script implements this experiment, but these exact totals were not found in a saved execution log. They remain reported historical observations rather than independently reproduced measurements.

### DCT approximation and residual cost

A local experiment on 65,536 binary-file bytes compared the original Huffman payload with rounded DCT coefficients plus a correction residual:

| Representation | Estimated size |
| --- | ---: |
| Original Huffman payload | 62,514 B |
| Retain 1% of DCT coefficients, plus residual | 68,546 B |
| Retain 100% of rounded DCT coefficients, plus residual | 147,445 B |

All displayed nonzero coefficient-retention settings exceeded the original Huffman payload. Even retaining every rounded coefficient leaves a residual. Codebooks and other framing are excluded, so these are partial storage estimates rather than complete codec results.

### Current DST exploration

The recent direction uses DST-II, scales coefficients into the signed 8-bit range, rounds them, and examines the residual needed for exact reconstruction. One 65,536-byte binary sample produced a saved sum of absolute reconstruction errors of 148,634; this is an error metric, not a byte count.

A separate experiment applies a magnitude threshold before scaling and rounding the coefficients of a 65,536-sample PNG-file byte block. It constructs a residual Huffman tree and reports 56 residual symbols and a payload of 316,350 bits, rounded up to 39,544 bytes. The input is encoded PNG bytes, not decoded image pixels.

The residual figure excludes coefficients, scale, codebook, original length, and padding metadata. Storing all 65,536 quantized coefficients as one byte each would already bring coefficients plus residual payload to 105,080 bytes before metadata. A sparse coefficient format could change that cost, but a complete stored representation and decoder have not been demonstrated.

Normalization and quantization are active evaluation concerns. Rounding coefficients under different normalizations applies different effective quantization steps. Flooring coefficients to multiples of 256 introduces a nonpositive coefficient error: for example, -1.5 maps to -256 while 1.5 maps to 0. Such biased errors can accumulate at particular inverse-transform positions. Neither small coefficient magnitudes nor a small residual payload alone establishes a compression advantage.

The next research milestone is a fully accounted model-plus-residual comparison with exact reconstruction, preserving the requirement that the proposed compression path include a Fourier component.

## Size accounting and limitations

The tracked implementations use different cost boundaries:

- bases.py constructs parameter/residual bytes and adds an assumed mode-byte cost, but does not serialize a complete block stream.
- codecs/const.py returns integer fields and compares their count with the input length.
- The Fourier scripts calculate Huffman payload costs without storing the model or codebook.
- The reciprocal experiment measures rounded-coefficient payload and energy statistics without a lossless reconstruction format.

These quantities must not all be interpreted as complete compressed-file sizes.

Known limitations include:

- Original length, block boundaries, mode identifiers, and final-block handling are not unified.
- A matching file decoder is not implemented.
- Sparse position/value pairs can be expensive when many residuals are nonzero.
- F_257 coefficients and residuals can include 256, requiring an explicit representation beyond a raw byte.
- Real-valued energy concentration and reconstruction error alone do not establish compression efficiency.
- Saved results and current experiment configurations are not always synchronized.

## Working with the experiments

There is no unified installation procedure or compression command. The tracked experiments use NumPy, SciPy, SymPy, and, for optional decoded-image comparisons, Pillow. A pinned dependency manifest is not currently provided.

Before running an experiment:

1. Inspect its input paths and parameters and supply the required local input data.
2. Choose the working directory deliberately: several paths are relative to it, not to the script location.
3. Distinguish estimated payload costs from complete file sizes.
4. Record the source version, input identity, and configuration when collecting results.

Some scripts execute substantial work at module level. Importing archive/main.py can start an exhaustive model search, and importing archive/main_whole_file_test.py can start a full-file scan.

## Research priorities

1. Define a complete byte format and compare every model using actual serialized size, including metadata.
2. Connect block representations to file-level encoding and exact reconstruction checks.
3. Compare position/value pairs, bitmap-based exception locations, and entropy-coded residuals under consistent cost accounting.
4. Evaluate Fourier models against simple predictors across complete files and diverse data classes.
5. Investigate shared Fourier models and multiple exponential bases with residuals.
6. Evaluate real-valued transform approximations using coefficient, parameter, and residual costs together.

Negative results remain part of the research record. Positive synthetic examples need validation on independent real data before any practical compression claim.
