# Durumari

An experimental lossless data compression research project exploring finite-field transforms, exponential models, and linear recurrences.

The project investigates whether algebraic structure in binary data can provide compact, exactly reversible representations. It also compares finite-field models with simpler byte predictors and explores real-valued transforms such as DCT.

**This is a research codebase, not a complete file compression tool.** Individual codecs, mathematical checks, and size-estimation experiments are implemented, but a unified file format and an end-to-end file encoder/decoder are not yet available. Compression gains are hypotheses to test, not assumed properties of the transforms.

## Research approach

The main finite-field experiments use:

- Prime modulus: `p = 257`
- Field: `F_257`
- Default block length: 256 bytes
- Input byte values: `0..255`
- Field element values: `0..256`

Because `257 - 1 = 256`, the nonzero field elements provide a natural basis family for 256-element blocks.

### Finite-field Fourier / Vandermonde transform

For `a = 1..256` and `n = 0..255`, the synthesis and analysis matrices are:

```text
V[a - 1, n] = a^n mod 257
A[a - 1, n] = -a^(-n) mod 257

c = A @ x mod 257
x = V.T @ c mod 257
```

The notebooks contain checks of `A @ V.T mod 257 == I` and exact reconstruction. They also demonstrate that a signal such as

```text
x[n] = 6 + 9*2^n + 8*3^n + 7*4^n mod 257
```

has only four nonzero transform coefficients.

Invertibility establishes information preservation. It does not establish compression: the model, coefficients, residuals, and metadata must together cost less than the original representation.

### Linear recurrences

`berlekamp_massey()` finds a minimal linear recurrence over `F_257` for the supplied finite sequence:

```text
x[n] = c[0]*x[n-1] + ... + c[k-1]*x[n-k] mod 257
```

The Linear codec stores the first `k` values and the `k` recurrence coefficients. Its current representation is:

```text
[0x07][k: uint16 little-endian][2*k field elements packed into 9 bits each]
```

Its size is `3 + ceil(18*k / 8)` bytes, excluding any outer framing. The decoder receives the original length separately.

Berlekamp–Massey finds an exact recurrence. It does not optimize a lower-order approximate recurrence together with a residual encoding.

## Repository layout

| Path | Purpose |
| --- | --- |
| `archive/main.ipynb` | Transform construction and checks; Constant, Delta, Delta2, Periodic, and Fourier1 codecs; first-block comparison on `Wildlife.wmv`. |
| `archive/main_with_linear.ipynb` | A related notebook adding Berlekamp–Massey, the Linear codec, and a RAW candidate; first-block comparison on `gui.zip`. |
| `archive/main.py` | Exhaustive single-exponential model search on the first block, minimizing estimated Huffman residual payload. |
| `archive/main_whole_file_test.py` | Whole-file comparison of raw-byte and finite-field-transform Huffman payload costs. |
| `archive/bases.py` | Separate mod-256 predictor implementation and file-level size estimator. |
| `codecs/const.py` | Standalone Constant encoder returning a list of integer fields. |
| `dct.ipynb` | DCT energy analysis and model-plus-residual size estimates for a 64 KiB sample. |
| `main.py` | DCT rounding and reconstruction experiment on a JPEG; not a compression CLI. |
| `which percentile.py` | DCT coefficient-rank exploration. |
| `which percentile.ipynb` | Incomplete exploratory notebook. |
| `archive/reciprocal_basis_experiment.py` | Reciprocal-derived real basis construction, SVD orthogonalization, and DFT/DCT comparisons. |
| `archive/reciprocal_basis_experiment_output.txt` | Saved results from reciprocal-basis experiments. |
| Other files in `archive/` | Modular arithmetic, periodic waveforms, orthogonality, FFT, and byte-distribution explorations. |

The two main notebooks share their initial transform and codec implementations, apart from the input file selection. The Linear notebook adds functionality; filenames alone should not be treated as a version history.

Some experiments require local sample files, including `Wildlife.wmv`, `archive/gui.zip`, and JPEG images. These samples are excluded by `.gitignore` and may not be present in a fresh checkout. Most notebooks are also ignored; files available locally may therefore be absent from a clone. The already tracked `dct.ipynb` remains tracked despite the ignore rule.

## Implemented models

The main notebooks implement the following block representations:

| Model | Functions | Representation |
| --- | --- | --- |
| RAW | Lambda in the competition cell | Method identifier followed by the input bytes. |
| Constant | `encoder_const()`, `decoder_const()` | Most frequent byte plus position/value exceptions. |
| Delta | `encoder_delta()`, `decoder_delta()` | First byte plus Constant encoding of mod-256 first differences. |
| Delta2 | `encoder_delta2()`, `decoder_delta2()` | First two bytes plus Constant encoding of mod-256 second differences. |
| Periodic | `encoder_periodic()`, `decoder_periodic()` | Period, initial seed, and position/value exceptions; searches periods `2..n//2`. |
| Fourier1 | `encoder_fourier1()`, `decoder_fourier1()` | One finite-field exponential model plus position/value exceptions. |
| Linear | `encoder_linear()`, `decoder_linear()` | Initial values and exact recurrence coefficients, packed as 9-bit field elements. |

Fourier1 searches each basis and chooses the coefficient satisfying the largest number of positions. It is distinct from `archive/main.py`, which searches models by estimated Huffman residual payload cost.

`archive/bases.py` is a separate implementation:

- `BasisResult` holds model parameters, sparse residual bytes, and an estimated cost.
- `_pack_sparse_residual()` stores nonzero mod-256 residuals as one-byte position/value pairs.
- `best_basis()` compares RAW, Constant, Delta, Delta2, and Periodic; it does not include Fourier1 or Linear.
- `analyze_file()` accumulates estimated costs and model-selection statistics over a file.

The two implementations are not interchangeable. For example, the notebook Delta codec models the most frequent difference, while `bases.py` stores nonzero differences relative to zero. Their Periodic candidate sets and overhead accounting also differ.

## Recorded observations

The following figures are saved notebook outputs, not a guarantee for other inputs or a fresh execution.

### Linear examples

| Input | Input size | Recurrence order | Encoded size |
| --- | ---: | ---: | ---: |
| `b"ABCD" * 64` | 256 B | 4 | 12 B |
| All zeros | 256 B | 0 | 3 B |
| Seeded random bytes | 256 B | 128 | 291 B |

These examples include encoder/decoder round-trip assertions. The four-exponential synthetic signal separately produces a recurrence of order four.

### First block of a ZIP file

`archive/main_with_linear.ipynb` records:

| Model | Reported size |
| --- | ---: |
| RAW | 257 |
| Periodic | 267 |
| Linear | 291 |
| Constant | 311 |
| Fourier1 | 312 |
| Delta | 366 |
| Delta2 | 383 |

RAW wins this comparison. These are lengths of returned integer lists; Linear has explicit byte packing, while Fourier1 still has an unresolved serialization case for coefficient 256. The sample includes ZIP headers and filenames and does not characterize the entire archive.

The older `main.ipynb` comparison has no RAW candidate and reports Constant as best at 267 fields for a 256-byte Wildlife block. Winning that candidate set does not imply compression.

### Whole-file finite-field transform experiment

`archive/main_whole_file_test.py` compares two global Huffman payload estimates:

1. Original byte frequencies.
2. Frequencies after transforming complete 256-byte blocks, with the final short block left as raw symbols.

The research handover reports the following historical observations for a 26,246,026-byte `Wildlife.wmv` file:

| Representation | Huffman payload |
| --- | ---: |
| Original bytes | 209,625,002 bits |
| Finite-field Fourier coefficients | 210,170,179 bits |

The corresponding experiment is implemented, but these exact totals were not found in saved output files during the codebase review. They are historical reported observations, not independently reproduced results. They indicate a negative result for this file and this coding model, not for every data class or entropy model.

### DCT and reciprocal-derived bases

The current `dct.ipynb` analyzes the first 65,536 bytes of Wildlife. Its saved estimates include:

| Representation | Estimated size |
| --- | ---: |
| Original Huffman payload | 62,514 B |
| DCT retaining 1% of coefficients, plus residual | 68,546 B |
| DCT retaining 100% of rounded coefficients, plus residual | 147,445 B |

Every displayed nonzero KEEP setting costs more than the original Huffman payload. Rounding leaves a residual even when all coefficients are retained. These estimates exclude codebooks and other framing; no complete DCT file codec is implemented.

The reciprocal experiment constructs real-valued waveforms from centered finite-field inverses, then uses the SVD polar factor to obtain an orthogonal matrix. Saved results confirm numerical orthogonality, but do not demonstrate a practical lossless compression advantage over DFT/DCT. The saved output uses inputs different from the current script's input list, so it should not be treated as an exact record of the current configuration.

## Size accounting and limitations

Reported sizes currently have different meanings:

- Several notebook encoders return logical `list[int]` fields and compare list lengths.
- Linear explicitly packs field elements into bytes.
- `bases.py` builds parameter/residual bytes but assumes a mode-byte cost without implementing a complete stream format.
- Huffman helpers calculate payload length from frequency merge costs; they do not generate codebooks or compressed bitstreams.

Consequently, these figures must not all be interpreted as complete compressed-file sizes.

Known gaps and caveats include:

- Fourier1 can return coefficient `256`. Its candidate scoring includes a provisional extra byte, but no escape serializer implements that representation and the final competition uses list length.
- Original length, block boundaries, final-block policy, and model dispatch are not unified.
- Decoder validation of truncated or malformed payloads is incomplete.
- Some exploratory cells interpret bytes as signed `int8`; this differs from embedding `0..255` directly in `F_257`. The actual Fourier1 encoder uses unsigned input.
- Small-prime adaptation examples mask off upper bits before reconstruction. They do not implement lossless adaptation of the original full bytes.
- DCT and reciprocal energy metrics are exploratory measurements, not lossless compression ratios.
- Saved notebook outputs, comments, and current input settings are not always synchronized.
- `run_sanity_tests()` in `bases.py` prints model costs; it is not a decoder round-trip test suite.

## Working with the experiments

There is no unified installation procedure or compression command. Dependencies vary by experiment and include NumPy, SciPy, SymPy, Matplotlib, and, in some explorations, pandas and Pillow. A pinned dependency manifest is not currently provided.

Before running an experiment:

1. Inspect its input paths and parameters.
2. Make the required sample files available.
3. Choose the working directory deliberately: several paths are relative to it, not to the script location.
4. Distinguish saved outputs from newly reproduced measurements.

Some scripts execute substantial work at module level. In particular, `archive/main.py` performs an exhaustive model search, and `archive/main_whole_file_test.py` scans the full input file. Importing these scripts can therefore start experiments.

## Research priorities

1. Define a complete byte format and compare every model using actual serialized size, including metadata.
2. Connect block codecs to file-level encoding and exact reconstruction checks, including short final blocks.
3. Measure recurrence-order distributions and model-selection costs across complete files and diverse data classes.
4. Compare the existing predictor variants under consistent storage rules.
5. Investigate lower-order recurrences or multiple exponential bases with residuals, optimizing total stored cost rather than exact-match count alone.

Negative results remain part of the research record. Positive synthetic examples need validation on independent real data before any practical compression claim.
