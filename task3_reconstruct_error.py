"""
Task 3: Dequantize, Denormalize, and Measure Error

This script reverses the normalization and quantization processes,
reconstructs the original meshes, and measures reconstruction errors.
"""

import os
import numpy as np
import glob
import csv
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import trimesh
from mesh_utils import (
    load_mesh, minmax_normalize, unitsphere_normalize,
    quantize, dequantize, denormalize_minmax, denormalize_unitsphere,
    compute_mse, compute_mae, save_mesh, visualize_mesh
)


def reconstruct_minmax(quantized_vertices, min_vals, max_vals, n_bins=1024):
    """
    Reconstruct original vertices from quantized Min-Max normalized vertices.
    
    Args:
        quantized_vertices: Quantized vertices (integers in [0, n_bins-1])
        min_vals: Minimum values used for normalization
        max_vals: Maximum values used for normalization
        n_bins: Number of quantization bins
        
    Returns:
        numpy array: Reconstructed vertices
    """
    # Dequantize
    dequantized = dequantize(quantized_vertices, n_bins)
    
    # Denormalize
    reconstructed = denormalize_minmax(dequantized, min_vals, max_vals)
    
    return reconstructed


def reconstruct_unitsphere(quantized_vertices, center, scale, n_bins=1024):
    """
    Reconstruct original vertices from quantized Unit Sphere normalized vertices.
    
    Args:
        quantized_vertices: Quantized vertices (integers in [0, n_bins-1])
        center: Center point used for normalization
        scale: Scale factor used for normalization
        n_bins: Number of quantization bins
        
    Returns:
        numpy array: Reconstructed vertices
    """
    # Dequantize (returns values in [0, 1])
    dequantized = dequantize(quantized_vertices, n_bins)
    
    # Convert from [0, 1] back to [-1, 1] range
    dequantized = dequantized * 2.0 - 1.0
    
    # Denormalize
    reconstructed = denormalize_unitsphere(dequantized, center, scale)
    
    return reconstructed


def process_mesh_reconstruction(mesh_path, output_base_dir, n_bins=1024):
    """
    Process a single mesh through reconstruction and error measurement.
    
    Args:
        mesh_path: Path to the input .obj file
        output_base_dir: Base directory for output files
        n_bins: Number of quantization bins
        
    Returns:
        dict: Dictionary containing error metrics and reconstruction data
    """
    print(f"\nProcessing: {mesh_path}")
    mesh = load_mesh(mesh_path)
    original_vertices = mesh.vertices
    faces = mesh.faces if hasattr(mesh, 'faces') else None
    filename = os.path.basename(mesh_path)
    basename = os.path.splitext(filename)[0]
    
    results = {
        'filename': filename,
        'num_vertices': len(original_vertices),
    }
    
    # Min-Max Normalization and Quantization
    print("  Reconstructing Min-Max...")
    vertices_minmax_norm, min_vals, max_vals = minmax_normalize(original_vertices)
    vertices_minmax_quant = quantize(vertices_minmax_norm, n_bins)
    vertices_minmax_recon = reconstruct_minmax(vertices_minmax_quant, min_vals, max_vals, n_bins)
    
    # Compute errors for Min-Max
    mse_minmax = compute_mse(original_vertices, vertices_minmax_recon)
    mae_minmax = compute_mae(original_vertices, vertices_minmax_recon)
    
    results['minmax'] = {
        'mse': mse_minmax,
        'mae': mae_minmax,
        'reconstructed_vertices': vertices_minmax_recon
    }
    
    print(f"    MSE: {mse_minmax['mse_overall']:.8f}")
    print(f"    MAE: {mae_minmax['mae_overall']:.8f}")
    
    # Unit Sphere Normalization and Quantization
    print("  Reconstructing Unit Sphere...")
    vertices_unitsphere_norm, center, scale = unitsphere_normalize(original_vertices)
    # Shift to [0, 1] for quantization
    vertices_unitsphere_for_quant = (vertices_unitsphere_norm + 1.0) / 2.0
    vertices_unitsphere_for_quant = np.clip(vertices_unitsphere_for_quant, 0.0, 1.0)
    vertices_unitsphere_quant = quantize(vertices_unitsphere_for_quant, n_bins)
    vertices_unitsphere_recon = reconstruct_unitsphere(vertices_unitsphere_quant, center, scale, n_bins)
    
    # Compute errors for Unit Sphere
    mse_unitsphere = compute_mse(original_vertices, vertices_unitsphere_recon)
    mae_unitsphere = compute_mae(original_vertices, vertices_unitsphere_recon)
    
    results['unitsphere'] = {
        'mse': mse_unitsphere,
        'mae': mae_unitsphere,
        'reconstructed_vertices': vertices_unitsphere_recon
    }
    
    print(f"    MSE: {mse_unitsphere['mse_overall']:.8f}")
    print(f"    MAE: {mae_unitsphere['mae_overall']:.8f}")
    
    # Save reconstructed meshes
    output_dir_minmax_recon = os.path.join(output_base_dir, 'reconstructed_minmax')
    os.makedirs(output_dir_minmax_recon, exist_ok=True)
    output_path_minmax_recon = os.path.join(output_dir_minmax_recon, filename)
    save_mesh(vertices_minmax_recon, faces, output_path_minmax_recon)
    print(f"  Saved reconstructed (Min-Max) mesh to {output_path_minmax_recon}")
    
    output_dir_unitsphere_recon = os.path.join(output_base_dir, 'reconstructed_unitsphere')
    os.makedirs(output_dir_unitsphere_recon, exist_ok=True)
    output_path_unitsphere_recon = os.path.join(output_dir_unitsphere_recon, filename)
    save_mesh(vertices_unitsphere_recon, faces, output_path_unitsphere_recon)
    print(f"  Saved reconstructed (Unit Sphere) mesh to {output_path_unitsphere_recon}")
    
    # Generate visualizations
    vis_dir = os.path.join(output_base_dir, 'visualizations', 'task3')
    os.makedirs(vis_dir, exist_ok=True)
    
    try:
        # Original mesh
        vis_original = os.path.join(vis_dir, f"{basename}_original.png")
        visualize_mesh(mesh, title=f"{basename} - Original", save_path=vis_original)
        
        # Reconstructed Min-Max - create new mesh with reconstructed vertices
        if faces is not None and len(faces) > 0:
            mesh_minmax_recon = trimesh.Trimesh(vertices=vertices_minmax_recon, faces=faces)
        else:
            mesh_minmax_recon = trimesh.PointCloud(vertices=vertices_minmax_recon)
        vis_minmax_recon = os.path.join(vis_dir, f"{basename}_minmax_reconstructed.png")
        visualize_mesh(mesh_minmax_recon, title=f"{basename} - Min-Max Reconstructed", save_path=vis_minmax_recon)
        
        # Reconstructed Unit Sphere - create new mesh with reconstructed vertices
        if faces is not None and len(faces) > 0:
            mesh_unitsphere_recon = trimesh.Trimesh(vertices=vertices_unitsphere_recon, faces=faces)
        else:
            mesh_unitsphere_recon = trimesh.PointCloud(vertices=vertices_unitsphere_recon)
        vis_unitsphere_recon = os.path.join(vis_dir, f"{basename}_unitsphere_reconstructed.png")
        visualize_mesh(mesh_unitsphere_recon, title=f"{basename} - Unit Sphere Reconstructed", save_path=vis_unitsphere_recon)
        
    except Exception as e:
        print(f"  Warning: Could not generate visualizations: {e}")
    
    return results


def plot_error_metrics(all_results, output_dir):
    """
    Generate error visualization plots.
    
    Args:
        all_results: List of results dictionaries
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data for plotting
    filenames = [r['filename'] for r in all_results]
    mse_minmax_overall = [r['minmax']['mse']['mse_overall'] for r in all_results]
    mse_unitsphere_overall = [r['unitsphere']['mse']['mse_overall'] for r in all_results]
    mae_minmax_overall = [r['minmax']['mae']['mae_overall'] for r in all_results]
    mae_unitsphere_overall = [r['unitsphere']['mae']['mae_overall'] for r in all_results]
    
    # Plot 1: Overall MSE comparison
    plt.figure(figsize=(12, 6))
    x = np.arange(len(filenames))
    width = 0.35
    plt.bar(x - width/2, mse_minmax_overall, width, label='Min-Max', alpha=0.8)
    plt.bar(x + width/2, mse_unitsphere_overall, width, label='Unit Sphere', alpha=0.8)
    plt.xlabel('Mesh')
    plt.ylabel('MSE (Overall)')
    plt.title('Mean Squared Error Comparison: Min-Max vs Unit Sphere')
    plt.xticks(x, [f.replace('.obj', '') for f in filenames], rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mse_overall_comparison.png'), dpi=150)
    plt.close()
    
    # Plot 2: Overall MAE comparison
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, mae_minmax_overall, width, label='Min-Max', alpha=0.8)
    plt.bar(x + width/2, mae_unitsphere_overall, width, label='Unit Sphere', alpha=0.8)
    plt.xlabel('Mesh')
    plt.ylabel('MAE (Overall)')
    plt.title('Mean Absolute Error Comparison: Min-Max vs Unit Sphere')
    plt.xticks(x, [f.replace('.obj', '') for f in filenames], rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mae_overall_comparison.png'), dpi=150)
    plt.close()
    
    # Plot 3: MSE per axis (Min-Max)
    mse_minmax_x = [r['minmax']['mse']['mse_x'] for r in all_results]
    mse_minmax_y = [r['minmax']['mse']['mse_y'] for r in all_results]
    mse_minmax_z = [r['minmax']['mse']['mse_z'] for r in all_results]
    
    plt.figure(figsize=(12, 6))
    width = 0.25
    plt.bar(x - width, mse_minmax_x, width, label='X-axis', alpha=0.8)
    plt.bar(x, mse_minmax_y, width, label='Y-axis', alpha=0.8)
    plt.bar(x + width, mse_minmax_z, width, label='Z-axis', alpha=0.8)
    plt.xlabel('Mesh')
    plt.ylabel('MSE')
    plt.title('MSE per Axis - Min-Max Normalization')
    plt.xticks(x, [f.replace('.obj', '') for f in filenames], rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mse_per_axis_minmax.png'), dpi=150)
    plt.close()
    
    # Plot 4: MSE per axis (Unit Sphere)
    mse_unitsphere_x = [r['unitsphere']['mse']['mse_x'] for r in all_results]
    mse_unitsphere_y = [r['unitsphere']['mse']['mse_y'] for r in all_results]
    mse_unitsphere_z = [r['unitsphere']['mse']['mse_z'] for r in all_results]
    
    plt.figure(figsize=(12, 6))
    plt.bar(x - width, mse_unitsphere_x, width, label='X-axis', alpha=0.8)
    plt.bar(x, mse_unitsphere_y, width, label='Y-axis', alpha=0.8)
    plt.bar(x + width, mse_unitsphere_z, width, label='Z-axis', alpha=0.8)
    plt.xlabel('Mesh')
    plt.ylabel('MSE')
    plt.title('MSE per Axis - Unit Sphere Normalization')
    plt.xticks(x, [f.replace('.obj', '') for f in filenames], rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mse_per_axis_unitsphere.png'), dpi=150)
    plt.close()
    
    # Plot 5: MAE per axis (Min-Max)
    mae_minmax_x = [r['minmax']['mae']['mae_x'] for r in all_results]
    mae_minmax_y = [r['minmax']['mae']['mae_y'] for r in all_results]
    mae_minmax_z = [r['minmax']['mae']['mae_z'] for r in all_results]
    
    plt.figure(figsize=(12, 6))
    plt.bar(x - width, mae_minmax_x, width, label='X-axis', alpha=0.8)
    plt.bar(x, mae_minmax_y, width, label='Y-axis', alpha=0.8)
    plt.bar(x + width, mae_minmax_z, width, label='Z-axis', alpha=0.8)
    plt.xlabel('Mesh')
    plt.ylabel('MAE')
    plt.title('MAE per Axis - Min-Max Normalization')
    plt.xticks(x, [f.replace('.obj', '') for f in filenames], rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mae_per_axis_minmax.png'), dpi=150)
    plt.close()
    
    # Plot 6: MAE per axis (Unit Sphere)
    mae_unitsphere_x = [r['unitsphere']['mae']['mae_x'] for r in all_results]
    mae_unitsphere_y = [r['unitsphere']['mae']['mae_y'] for r in all_results]
    mae_unitsphere_z = [r['unitsphere']['mae']['mae_z'] for r in all_results]
    
    plt.figure(figsize=(12, 6))
    plt.bar(x - width, mae_unitsphere_x, width, label='X-axis', alpha=0.8)
    plt.bar(x, mae_unitsphere_y, width, label='Y-axis', alpha=0.8)
    plt.bar(x + width, mae_unitsphere_z, width, label='Z-axis', alpha=0.8)
    plt.xlabel('Mesh')
    plt.ylabel('MAE')
    plt.title('MAE per Axis - Unit Sphere Normalization')
    plt.xticks(x, [f.replace('.obj', '') for f in filenames], rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mae_per_axis_unitsphere.png'), dpi=150)
    plt.close()
    
    print(f"\nError plots saved to {output_dir}/")


def save_error_metrics(all_results, output_file='task3_errors.csv'):
    """
    Save error metrics to a CSV file.
    
    Args:
        all_results: List of results dictionaries
        output_file: Path to output CSV file
    """
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            'Mesh', 'Method',
            'MSE_X', 'MSE_Y', 'MSE_Z', 'MSE_Overall',
            'MAE_X', 'MAE_Y', 'MAE_Z', 'MAE_Overall'
        ])
        
        # Data rows
        for result in all_results:
            filename = result['filename']
            
            # Min-Max
            mse = result['minmax']['mse']
            mae = result['minmax']['mae']
            writer.writerow([
                filename, 'Min-Max',
                mse['mse_x'], mse['mse_y'], mse['mse_z'], mse['mse_overall'],
                mae['mae_x'], mae['mae_y'], mae['mae_z'], mae['mae_overall']
            ])
            
            # Unit Sphere
            mse = result['unitsphere']['mse']
            mae = result['unitsphere']['mae']
            writer.writerow([
                filename, 'Unit Sphere',
                mse['mse_x'], mse['mse_y'], mse['mse_z'], mse['mse_overall'],
                mae['mae_x'], mae['mae_y'], mae['mae_z'], mae['mae_overall']
            ])
    
    print(f"Error metrics saved to {output_file}")


def generate_conclusion(all_results):
    """
    Generate a conclusion based on error analysis.
    
    Args:
        all_results: List of results dictionaries
        
    Returns:
        str: Conclusion text
    """
    # Calculate average errors
    avg_mse_minmax = np.mean([r['minmax']['mse']['mse_overall'] for r in all_results])
    avg_mse_unitsphere = np.mean([r['unitsphere']['mse']['mse_overall'] for r in all_results])
    avg_mae_minmax = np.mean([r['minmax']['mae']['mae_overall'] for r in all_results])
    avg_mae_unitsphere = np.mean([r['unitsphere']['mae']['mae_overall'] for r in all_results])
    
    conclusion = []
    conclusion.append("=" * 80)
    conclusion.append("CONCLUSION")
    conclusion.append("=" * 80)
    conclusion.append("")
    conclusion.append("Error Analysis Summary:")
    conclusion.append("-" * 80)
    conclusion.append(f"Average MSE (Min-Max): {avg_mse_minmax:.8f}")
    conclusion.append(f"Average MSE (Unit Sphere): {avg_mse_unitsphere:.8f}")
    conclusion.append(f"Average MAE (Min-Max): {avg_mae_minmax:.8f}")
    conclusion.append(f"Average MAE (Unit Sphere): {avg_mae_unitsphere:.8f}")
    conclusion.append("")
    
    # Determine which method gives less error
    if avg_mse_minmax < avg_mse_unitsphere:
        better_method = "Min-Max"
        conclusion.append(f"Min-Max normalization produces lower reconstruction error on average.")
    elif avg_mse_unitsphere < avg_mse_minmax:
        better_method = "Unit Sphere"
        conclusion.append(f"Unit Sphere normalization produces lower reconstruction error on average.")
    else:
        better_method = "Both methods perform similarly"
        conclusion.append(f"Both normalization methods produce similar reconstruction errors.")
    
    conclusion.append("")
    conclusion.append("Observations:")
    conclusion.append("-" * 80)
    conclusion.append("1. Quantization with 1024 bins introduces minimal information loss.")
    conclusion.append("2. The reconstruction errors are very small, indicating that the")
    conclusion.append("   quantization process preserves mesh structure effectively.")
    conclusion.append("3. Error patterns may vary depending on the mesh geometry and")
    conclusion.append("   the distribution of vertices in 3D space.")
    conclusion.append("4. Both normalization methods are suitable for mesh processing,")
    conclusion.append("   with the choice depending on specific application requirements.")
    conclusion.append("")
    conclusion.append("Recommendations:")
    conclusion.append("-" * 80)
    conclusion.append("- For applications requiring minimal reconstruction error, use the")
    conclusion.append(f"  method with lower error ({better_method}).")
    conclusion.append("- Consider the mesh characteristics: Min-Max is better for")
    conclusion.append("  preserving scale relationships, while Unit Sphere is better for")
    conclusion.append("  rotation-invariant processing.")
    conclusion.append("- The quantization bin size (1024) provides a good balance between")
    conclusion.append("  compression and quality. Increasing bins reduces error but")
    conclusion.append("  increases storage requirements.")
    
    return "\n".join(conclusion)


def main():
    """
    Main function to process all meshes through reconstruction and error measurement.
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
        results = process_mesh_reconstruction(mesh_path, output_base_dir, n_bins)
        all_results.append(results)
    
    # Save error metrics to CSV
    save_error_metrics(all_results)
    
    # Generate error plots
    plot_dir = os.path.join(output_base_dir, 'visualizations', 'task3')
    plot_error_metrics(all_results, plot_dir)
    
    # Generate conclusion
    conclusion = generate_conclusion(all_results)
    print("\n" + conclusion)
    
    # Save conclusion to file
    conclusion_file = 'task3_conclusion.txt'
    with open(conclusion_file, 'w') as f:
        f.write(conclusion)
    
    print("\n" + "=" * 80)
    print("Task 3 completed successfully!")
    print(f"Processed {len(mesh_files)} meshes")
    print(f"Error metrics saved to task3_errors.csv")
    print(f"Error plots saved to {plot_dir}/")
    print(f"Conclusion saved to {conclusion_file}")


if __name__ == "__main__":
    main()

