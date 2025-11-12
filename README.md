# Mesh Normalization, Quantization, and Error Analysis

This project implements a comprehensive pipeline for processing 3D mesh files, encompassing normalization, quantization, reconstruction, and error analysis.

## Overview

The assignment focuses on understanding and implementing data preprocessing for 3D meshes:
1. Loading and inspecting mesh data
2. Applying normalization (Min-Max and Unit Sphere)
3. Applying quantization (discretizing coordinates into bins)
4. Reversing transformations (dequantize, denormalize)
5. Measuring and visualizing differences between original and transformed meshes

## Requirements

- Python 3.7+
- NumPy
- Trimesh
- Matplotlib (for visualization - works in headless environments)
- Open3D (optional - not used, matplotlib is used instead for better headless support)

## Installation
1. Virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Project Structure

```
.
├── 8samples/                    # Input mesh files (.obj)
├── output/                      # Output directory
│   ├── normalized_minmax/      # Min-Max normalized meshes
│   ├── normalized_unitsphere/  # Unit Sphere normalized meshes
│   ├── quantized_minmax/       # Quantized Min-Max meshes
│   ├── quantized_unitsphere/   # Quantized Unit Sphere meshes
│   ├── reconstructed_minmax/   # Reconstructed Min-Max meshes
│   ├── reconstructed_unitsphere/ # Reconstructed Unit Sphere meshes
│   └── visualizations/         # Visualization images
├── mesh_utils.py               # Utility functions
├── task1_load_inspect.py       # Task 1: Load and inspect meshes
├── task2_normalize_quantize.py # Task 2: Normalize and quantize
├── task3_reconstruct_error.py  # Task 3: Reconstruct and measure error
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── REPORT.md                   # Comprehensive analysis report
```
NOTE: The files for the bonus task are explained towards the end. Not mentioned above. 

## How to Run

### Task 1: Load and Inspect Meshes

Load all mesh files, extract vertex coordinates, compute statistics, and generate visualizations:

```bash
python task1_load_inspect.py
```

**Output:**
- `task1_statistics.txt`: Statistics for all meshes
- `output/visualizations/task1/`: Visualization images of original meshes

**Observations:**
- Each mesh is loaded and its vertex coordinates are extracted
- Statistics include number of vertices, min/max/mean/std for each axis (x, y, z)
- Visualizations help understand the mesh structure and geometry

### Task 2: Normalize and Quantize Meshes

Apply two normalization methods (Min-Max and Unit Sphere), quantize the normalized vertices, and save results:

```bash
python task2_normalize_quantize.py
```

**Output:**
- `output/normalized_minmax/`: Min-Max normalized meshes
- `output/normalized_unitsphere/`: Unit Sphere normalized meshes
- `output/quantized_minmax/`: Quantized Min-Max meshes
- `output/quantized_unitsphere/`: Quantized Unit Sphere meshes
- `output/visualizations/task2/`: Comparison visualizations
- `task2_comparison.txt`: Comparison analysis of normalization methods

**Observations:**
- Min-Max normalization scales vertices to [0, 1] range, preserving relative distances
- Unit Sphere normalization centers the mesh and scales it to fit in a unit sphere
- Quantization with 1024 bins discretizes the normalized coordinates
- Both methods preserve mesh structure effectively

### Task 3: Reconstruct and Measure Error

Reverse the normalization and quantization processes, reconstruct original meshes, and measure reconstruction errors:

```bash
python task3_reconstruct_error.py
```

**Output:**
- `output/reconstructed_minmax/`: Reconstructed Min-Max meshes
- `output/reconstructed_unitsphere/`: Reconstructed Unit Sphere meshes
- `output/visualizations/task3/`: Error visualization plots
- `task3_errors.csv`: Error metrics (MSE and MAE) for all meshes
- `task3_conclusion.txt`: Analysis conclusion

**Observations:**
- Reconstruction errors are very small, indicating effective quantization
- MSE and MAE are computed per axis (x, y, z) and overall
- Error patterns vary depending on mesh geometry
- One normalization method may produce lower errors depending on mesh characteristics

## Running All Tasks

To run all tasks in sequence:

```bash
python task1_load_inspect.py
python task2_normalize_quantize.py
python task3_reconstruct_error.py
```

## Key Concepts

### Normalization

1. **Min-Max Normalization:**
   - Formula: `x' = (x - x_min) / (x_max - x_min)`
   - Brings all vertex coordinates into [0, 1] range
   - Preserves relative distances within the mesh

2. **Unit Sphere Normalization:**
   - Centers mesh at origin
   - Scales so all vertices fit inside a sphere of radius 1
   - Vertices lie in approximately [-1, 1] range

### Quantization

- Quantization discretizes continuous values into discrete bins
- With bin size 1024, each axis can take integer values between 0 and 1023
- Formula: `q = int(x' × (n_bins - 1))`
- Dequantization: `x' = q / (n_bins - 1)`

### Error Measurement

- **Mean Squared Error (MSE):** Average of squared differences
- **Mean Absolute Error (MAE):** Average of absolute differences
- Errors are computed per axis (x, y, z) and overall

## Output Files

1. **Statistics Files:**
   - `task1_statistics.txt`: Mesh inspection statistics

2. **Mesh Files:**
   - Normalized meshes (`.obj` format)
   - Quantized meshes (`.obj` format)
   - Reconstructed meshes (`.obj` format)

3. **Analysis Files:**
   - `task2_comparison.txt`: Normalization method comparison
   - `task3_errors.csv`: Error metrics in CSV format
   - `task3_conclusion.txt`: Error analysis conclusion

4. **Visualizations:**
   - Original mesh visualizations
   - Normalized mesh visualizations
   - Quantized mesh visualizations
   - Reconstructed mesh visualizations
   - Error plots (MSE and MAE comparisons)

## Notes

- All mesh files in the `8samples/` directory are processed automatically
- Visualization images are saved as PNG files
- Error metrics are saved in both CSV and text formats
- The quantization bin size is set to 1024 by default (can be modified in the scripts)

## Troubleshooting

1. **Import Errors:**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`

2. **Visualization Issues:**
   - The code uses matplotlib for visualization, which works in headless environments (no display required)
   - If visualizations fail, the scripts will continue processing meshes and print a warning
   - For very large meshes, visualizations may sample vertices/faces to avoid memory issues

3. **File Not Found Errors:**
   - Ensure the `8samples/` directory exists and contains `.obj` files
   - Check that file paths are correct

4. **Headless Server Issues:**
   - The code is designed to work on headless servers (no display)
   - Matplotlib uses the 'Agg' backend which doesn't require a display
   - All visualizations are saved as PNG files

# Bonus Task: Seam Tokenization Prototype

## Overview

This bonus task implements a prototype system for tokenizing mesh seams—edges where UV texture mappings break. The system enables sequential representation of seam structures, which is a crucial step toward SeamGPT-style processing for 3D meshes.

## Files

- `seam_tokenization.py`: Core implementation of seam detection and tokenization
- `bonus_seam_example.py`: Example script demonstrating usage
- `bonus_seam_report.md`: Comprehensive report and explanation

## Quick Start

### Running the Example

```bash
python bonus_seam_example.py
```

This will:
1. Load all meshes from `8samples/` directory
2. Detect seams in meshes with UV coordinates
3. Encode seams as token sequences
4. Decode tokens back to verify accuracy
5. Save results to `output/seam_tokenization/`

### Usage

```python
from seam_tokenization import (
    load_mesh_with_uv, detect_seams, build_seam_paths,
    encode_seams, decode_seam_tokens, tokens_to_string
)

# Load mesh with UV coordinates
mesh, uv_coords, face_uv_indices = load_mesh_with_uv("mesh.obj")

# Detect seams
seam_edges = detect_seams(mesh, uv_coords, face_uv_indices)

# Build seam paths
seam_paths = build_seam_paths(seam_edges)

# Encode to tokens
token_sequences = encode_seams(seam_paths)

# Decode back
for tokens in token_sequences:
    seam_id, path = decode_seam_tokens(tokens)
    print(f"Seam {seam_id}: {path}")
```

## Token Encoding Format

Tokens are encoded as integers:
- `SEAM_START` (0): Begin a new seam path
- `SEAM_END` (1): End current seam path
- `VERTEX` (2): Followed by vertex index
- `SEAM_ID` (6): Seam path identifier

**Example:**
```
[SEAM_START, SEAM_ID, 0, VERTEX, 5, VERTEX, 12, VERTEX, 8, SEAM_END]
```

## Key Features

1. **Seam Detection**: Identifies edges where UV mappings break
2. **Path Construction**: Groups seam edges into connected paths
3. **Token Encoding**: Converts seam paths to discrete token sequences
4. **Lossless Decoding**: Reconstructs seam paths from tokens with perfect accuracy
5. **Visualization**: Visualizes detected seams on meshes

## Output

The example script generates:
- Token sequences for each seam path
- Statistics (number of seams, token count, etc.)
- Visualization images
- Text files with detailed token information

## Connection to SeamGPT

This tokenization approach enables:
- **Learning**: Transformer models can learn patterns in seam structures
- **Generation**: Models can generate new seam patterns for novel meshes
- **Optimization**: Seam placement can be optimized using learned models
- **Compression**: Seam information can be compressed efficiently

## Limitations

1. Requires meshes with UV coordinates
2. Current implementation handles simple paths; complex branching may require enhancements
3. Long seam paths result in long token sequences

## Future Work

- Hierarchical encoding for multi-level seam representation
- Compression techniques to reduce token sequence length
- Enhanced algorithms for complex seam topologies
- Geometric seam detection (sharp edges, creases) when UV coordinates are unavailable
- Transformer models trained on seam token sequences



