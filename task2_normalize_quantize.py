"""
Task 2: Normalize and Quantize Meshes

This script applies two normalization methods (Min-Max and Unit Sphere),
quantizes the normalized vertices, and saves the results.
"""

import os
import numpy as np
import glob
import trimesh
from mesh_utils import (
    load_mesh, minmax_normalize, unitsphere_normalize,
    quantize, save_mesh, visualize_mesh
)


def process_mesh_normalization(mesh_path, output_base_dir, n_bins=1024):
    """
    Process a single mesh through normalization and quantization.
    
    Args:
        mesh_path: Path to the input .obj file
        output_base_dir: Base directory for output files
        n_bins: Number of quantization bins
        np
    Returns:
        dict: Dictionary containing processing results and comparison data
    """
    print(f"\nProcessing: {mesh_path}")
    mesh = load_mesh(mesh_path)
    vertices = mesh.vertices
    faces = mesh.faces if hasattr(mesh, 'faces') else None
    filename = os.path.basename(mesh_path)
    basename = os.path.splitext(filename)[0]
    
    results = {
        'filename': filename,
        'num_vertices': len(vertices),
    }
    
    # Min-Max Normalization
    print("  Applying Min-Max normalization...")
    vertices_minmax_norm, min_vals, max_vals = minmax_normalize(vertices)
    results['minmax_params'] = {'min_vals': min_vals, 'max_vals': max_vals}
    
    # Save normalized mesh (Min-Max)
    output_dir_minmax_norm = os.path.join(output_base_dir, 'normalized_minmax')
    os.makedirs(output_dir_minmax_norm, exist_ok=True)
    output_path_minmax_norm = os.path.join(output_dir_minmax_norm, filename)
    save_mesh(vertices_minmax_norm, faces, output_path_minmax_norm)
    print(f"  Saved normalized (Min-Max) mesh to {output_path_minmax_norm}")
    
    # Quantize (Min-Max)
    print("  Quantizing (Min-Max)...")
    vertices_minmax_quant = quantize(vertices_minmax_norm, n_bins)
    results['minmax_quantized'] = vertices_minmax_quant
    
    # Save quantized mesh (Min-Max) - convert back to float for saving
    vertices_minmax_quant_float = vertices_minmax_quant.astype(np.float64) / (n_bins - 1)
    output_dir_minmax_quant = os.path.join(output_base_dir, 'quantized_minmax')
    os.makedirs(output_dir_minmax_quant, exist_ok=True)
    output_path_minmax_quant = os.path.join(output_dir_minmax_quant, filename)
    save_mesh(vertices_minmax_quant_float, faces, output_path_minmax_quant)
    print(f"  Saved quantized (Min-Max) mesh to {output_path_minmax_quant}")
    
    # Unit Sphere Normalization
    print("  Applying Unit Sphere normalization...")
    vertices_unitsphere_norm, center, scale = unitsphere_normalize(vertices)
    results['unitsphere_params'] = {'center': center, 'scale': scale}
    
    # For unit sphere normalization, we need to shift to [0, 1] range for quantization
    # since unit sphere gives range [-1, 1]
    vertices_unitsphere_for_quant = (vertices_unitsphere_norm + 1.0) / 2.0
    vertices_unitsphere_for_quant = np.clip(vertices_unitsphere_for_quant, 0.0, 1.0)
    
    # Save normalized mesh (Unit Sphere) - save the actual normalized coordinates
    output_dir_unitsphere_norm = os.path.join(output_base_dir, 'normalized_unitsphere')
    os.makedirs(output_dir_unitsphere_norm, exist_ok=True)
    output_path_unitsphere_norm = os.path.join(output_dir_unitsphere_norm, filename)
    save_mesh(vertices_unitsphere_norm, faces, output_path_unitsphere_norm)
    print(f"  Saved normalized (Unit Sphere) mesh to {output_path_unitsphere_norm}")
    
    # Quantize (Unit Sphere) - use the [0, 1] shifted version
    print("  Quantizing (Unit Sphere)...")
    vertices_unitsphere_quant = quantize(vertices_unitsphere_for_quant, n_bins)
    results['unitsphere_quantized'] = vertices_unitsphere_quant
    
    # Save quantized mesh (Unit Sphere) - convert back to float for saving
    vertices_unitsphere_quant_float = vertices_unitsphere_quant.astype(np.float64) / (n_bins - 1)
    # Convert back to [-1, 1] range for visualization
    vertices_unitsphere_quant_float = vertices_unitsphere_quant_float * 2.0 - 1.0
    output_dir_unitsphere_quant = os.path.join(output_base_dir, 'quantized_unitsphere')
    os.makedirs(output_dir_unitsphere_quant, exist_ok=True)
    output_path_unitsphere_quant = os.path.join(output_dir_unitsphere_quant, filename)
    save_mesh(vertices_unitsphere_quant_float, faces, output_path_unitsphere_quant)
    print(f"  Saved quantized (Unit Sphere) mesh to {output_path_unitsphere_quant}")
    
    # Generate visualizations
    vis_dir = os.path.join(output_base_dir, 'visualizations', 'task2')
    os.makedirs(vis_dir, exist_ok=True)
    
    try:
        # Original mesh
        vis_original = os.path.join(vis_dir, f"{basename}_original.png")
        visualize_mesh(mesh, title=f"{basename} - Original", save_path=vis_original)
        
        # Min-Max normalized - create new mesh with normalized vertices
        if faces is not None and len(faces) > 0:
            mesh_minmax_norm = trimesh.Trimesh(vertices=vertices_minmax_norm, faces=faces)
        else:
            mesh_minmax_norm = trimesh.PointCloud(vertices=vertices_minmax_norm)
        vis_minmax_norm = os.path.join(vis_dir, f"{basename}_minmax_normalized.png")
        visualize_mesh(mesh_minmax_norm, title=f"{basename} - Min-Max Normalized", save_path=vis_minmax_norm)
        
        # Unit Sphere normalized - create new mesh with normalized vertices
        if faces is not None and len(faces) > 0:
            mesh_unitsphere_norm = trimesh.Trimesh(vertices=vertices_unitsphere_norm, faces=faces)
        else:
            mesh_unitsphere_norm = trimesh.PointCloud(vertices=vertices_unitsphere_norm)
        vis_unitsphere_norm = os.path.join(vis_dir, f"{basename}_unitsphere_normalized.png")
        visualize_mesh(mesh_unitsphere_norm, title=f"{basename} - Unit Sphere Normalized", save_path=vis_unitsphere_norm)
        
        # Quantized Min-Max - create new mesh with quantized vertices
        if faces is not None and len(faces) > 0:
            mesh_minmax_quant = trimesh.Trimesh(vertices=vertices_minmax_quant_float, faces=faces)
        else:
            mesh_minmax_quant = trimesh.PointCloud(vertices=vertices_minmax_quant_float)
        vis_minmax_quant = os.path.join(vis_dir, f"{basename}_minmax_quantized.png")
        visualize_mesh(mesh_minmax_quant, title=f"{basename} - Min-Max Quantized", save_path=vis_minmax_quant)
        
        # Quantized Unit Sphere - create new mesh with quantized vertices
        if faces is not None and len(faces) > 0:
            mesh_unitsphere_quant = trimesh.Trimesh(vertices=vertices_unitsphere_quant_float, faces=faces)
        else:
            mesh_unitsphere_quant = trimesh.PointCloud(vertices=vertices_unitsphere_quant_float)
        vis_unitsphere_quant = os.path.join(vis_dir, f"{basename}_unitsphere_quantized.png")
        visualize_mesh(mesh_unitsphere_quant, title=f"{basename} - Unit Sphere Quantized", save_path=vis_unitsphere_quant)
        
    except Exception as e:
        print(f"  Warning: Could not generate visualizations: {e}")
    
    # Store range information for comparison
    results['minmax_range'] = {
        'min': float(vertices_minmax_norm.min()),
        'max': float(vertices_minmax_norm.max()),
        'mean': float(vertices_minmax_norm.mean()),
        'std': float(vertices_minmax_norm.std())
    }
    
    results['unitsphere_range'] = {
        'min': float(vertices_unitsphere_norm.min()),
        'max': float(vertices_unitsphere_norm.max()),
        'mean': float(vertices_unitsphere_norm.mean()),
        'std': float(vertices_unitsphere_norm.std())
    }
    
    return results


def compare_normalization_methods(all_results):
    """
    Compare the two normalization methods and generate a report.
    
    Args:
        all_results: List of results dictionaries from processing
        
    Returns:
        str: Comparison analysis text
    """
    analysis = []
    analysis.append("Normalization Method Comparison")
    analysis.append("=" * 80)
    analysis.append("")
    
    analysis.append("Min-Max Normalization:")
    analysis.append("-" * 80)
    analysis.append("  - Scales vertices to [0, 1] range")
    analysis.append("  - Preserves relative distances within the mesh")
    analysis.append("  - Shape is preserved, but scale is normalized")
    analysis.append("  - Good for meshes with varying scales")
    analysis.append("")
    
    analysis.append("Unit Sphere Normalization:")
    analysis.append("-" * 80)
    analysis.append("  - Centers mesh at origin and scales to fit in unit sphere")
    analysis.append("  - Vertices lie in approximately [-1, 1] range")
    analysis.append("  - Preserves overall shape and proportions")
    analysis.append("  - Good for maintaining mesh structure relative to center")
    analysis.append("")
    
    analysis.append("Observations per Mesh:")
    analysis.append("-" * 80)
    
    for result in all_results:
        analysis.append(f"\n{result['filename']}:")
        analysis.append(f"  Min-Max range: [{result['minmax_range']['min']:.4f}, {result['minmax_range']['max']:.4f}]")
        analysis.append(f"  Min-Max mean: {result['minmax_range']['mean']:.4f}, std: {result['minmax_range']['std']:.4f}")
        analysis.append(f"  Unit Sphere range: [{result['unitsphere_range']['min']:.4f}, {result['unitsphere_range']['max']:.4f}]")
        analysis.append(f"  Unit Sphere mean: {result['unitsphere_range']['mean']:.4f}, std: {result['unitsphere_range']['std']:.4f}")
    
    analysis.append("\n" + "=" * 80)
    analysis.append("Conclusion:")
    analysis.append("-" * 80)
    analysis.append("Both normalization methods preserve the mesh structure effectively.")
    analysis.append("Min-Max normalization provides a consistent [0, 1] range which is")
    analysis.append("advantageous for quantization, while Unit Sphere normalization")
    analysis.append("maintains the mesh centered at the origin which can be useful for")
    analysis.append("rotation-invariant processing. The choice depends on the specific")
    analysis.append("application requirements.")
    
    return "\n".join(analysis)


def main():
    """
    Main function to process all meshes through normalization and quantization.
    """
    # Get all .obj files from 8samples folder
    mesh_dir = '8samples'
    mesh_files = glob.glob(os.path.join(mesh_dir, '*.obj'))
    mesh_files.sort()
    
    if not mesh_files:
        print(f"Error: No .obj files found in {mesh_dir}/")
        return
    
    print(f"Found {len(mesh_files)} mesh files")
    print("=" * 80)
    
    # Output directory
    output_base_dir = 'output'
    os.makedirs(output_base_dir, exist_ok=True)
    
    n_bins = 1024
    print(f"Using quantization bin size: {n_bins}")
    
    all_results = []
    
    # Process each mesh
    for mesh_path in mesh_files:
        results = process_mesh_normalization(mesh_path, output_base_dir, n_bins)
        all_results.append(results)
    
    # Generate comparison analysis
    comparison_text = compare_normalization_methods(all_results)
    
    # Save comparison to file
    comparison_file = 'task2_comparison.txt'
    with open(comparison_file, 'w') as f:
        f.write(comparison_text)
    
    print("\n" + "=" * 80)
    print("Task 2 completed successfully!")
    print(f"Processed {len(mesh_files)} meshes")
    print(f"Comparison analysis saved to {comparison_file}")
    print(f"Output meshes saved to {output_base_dir}/")
    print("\n" + comparison_text)


if __name__ == "__main__":
    main()

