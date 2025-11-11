"""
Task 1: Load and Inspect Mesh Data

This script loads .obj mesh files, extracts vertex coordinates,
computes statistics, and generates visualizations.
"""

import os
import glob
from mesh_utils import load_mesh, visualize_mesh


def inspect_mesh(mesh_path):
    """
    Inspect a single mesh and return statistics.
    
    Args:
        mesh_path: Path to the .obj file
        
    Returns:
        dict: Dictionary containing mesh statistics
    """
    print(f"\nProcessing: {mesh_path}")
    mesh = load_mesh(mesh_path)
    vertices = mesh.vertices
    
    stats = {
        'filename': os.path.basename(mesh_path),
        'num_vertices': len(vertices),
        'num_faces': len(mesh.faces) if hasattr(mesh, 'faces') else 0,
        'min_x': float(vertices[:, 0].min()),
        'max_x': float(vertices[:, 0].max()),
        'mean_x': float(vertices[:, 0].mean()),
        'std_x': float(vertices[:, 0].std()),
        'min_y': float(vertices[:, 1].min()),
        'max_y': float(vertices[:, 1].max()),
        'mean_y': float(vertices[:, 1].mean()),
        'std_y': float(vertices[:, 1].std()),
        'min_z': float(vertices[:, 2].min()),
        'max_z': float(vertices[:, 2].max()),
        'mean_z': float(vertices[:, 2].mean()),
        'std_z': float(vertices[:, 2].std()),
    }
    
    return mesh, stats


def print_statistics(stats):
    """
    Print formatted statistics for a mesh.
    
    Args:
        stats: Dictionary containing mesh statistics
    """
    print(f"\n{'='*60}")
    print(f"Mesh: {stats['filename']}")
    print(f"{'='*60}")
    print(f"Number of vertices: {stats['num_vertices']}")
    print(f"Number of faces: {stats['num_faces']}")
    print(f"\nX-axis:")
    print(f"  Min: {stats['min_x']:.6f}")
    print(f"  Max: {stats['max_x']:.6f}")
    print(f"  Mean: {stats['mean_x']:.6f}")
    print(f"  Std: {stats['std_x']:.6f}")
    print(f"\nY-axis:")
    print(f"  Min: {stats['min_y']:.6f}")
    print(f"  Max: {stats['max_y']:.6f}")
    print(f"  Mean: {stats['mean_y']:.6f}")
    print(f"  Std: {stats['std_y']:.6f}")
    print(f"\nZ-axis:")
    print(f"  Min: {stats['min_z']:.6f}")
    print(f"  Max: {stats['max_z']:.6f}")
    print(f"  Mean: {stats['mean_z']:.6f}")
    print(f"  Std: {stats['std_z']:.6f}")


def save_statistics(all_stats, output_file='task1_statistics.txt'):
    """
    Save statistics to a text file.
    
    Args:
        all_stats: List of statistics dictionaries
        output_file: Path to output file
    """
    with open(output_file, 'w') as f:
        f.write("Mesh Inspection Statistics\n")
        f.write("=" * 80 + "\n\n")
        
        for stats in all_stats:
            f.write(f"Mesh: {stats['filename']}\n")
            f.write("-" * 80 + "\n")
            f.write(f"Number of vertices: {stats['num_vertices']}\n")
            f.write(f"Number of faces: {stats['num_faces']}\n")
            f.write(f"\nX-axis: Min={stats['min_x']:.6f}, Max={stats['max_x']:.6f}, "
                   f"Mean={stats['mean_x']:.6f}, Std={stats['std_x']:.6f}\n")
            f.write(f"Y-axis: Min={stats['min_y']:.6f}, Max={stats['max_y']:.6f}, "
                   f"Mean={stats['mean_y']:.6f}, Std={stats['std_y']:.6f}\n")
            f.write(f"Z-axis: Min={stats['min_z']:.6f}, Max={stats['max_z']:.6f}, "
                   f"Mean={stats['mean_z']:.6f}, Std={stats['std_z']:.6f}\n")
            f.write("\n" + "=" * 80 + "\n\n")
        
        # Summary statistics
        f.write("\nSummary Across All Meshes:\n")
        f.write("-" * 80 + "\n")
        total_vertices = sum(s['num_vertices'] for s in all_stats)
        total_faces = sum(s['num_faces'] for s in all_stats)
        f.write(f"Total number of meshes: {len(all_stats)}\n")
        f.write(f"Total vertices: {total_vertices}\n")
        f.write(f"Total faces: {total_faces}\n")
        f.write(f"Average vertices per mesh: {total_vertices / len(all_stats):.2f}\n")
        f.write(f"Average faces per mesh: {total_faces / len(all_stats):.2f}\n")
    
    print(f"\nStatistics saved to {output_file}")


def main():
    """
    Main function to process all meshes in the 8samples folder.
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
    
    # Create output directory for visualizations
    vis_dir = 'output/visualizations/task1'
    os.makedirs(vis_dir, exist_ok=True)
    
    all_stats = []
    all_meshes = []
    
    # Process each mesh
    for mesh_path in mesh_files:
        mesh, stats = inspect_mesh(mesh_path)
        print_statistics(stats)
        all_stats.append(stats)
        all_meshes.append((mesh, stats['filename']))
        
        # Generate visualization
        vis_path = os.path.join(vis_dir, f"{stats['filename']}_original.png")
        try:
            visualize_mesh(mesh, title=stats['filename'], save_path=vis_path)
            print(f"Visualization saved to {vis_path}")
        except Exception as e:
            print(f"Warning: Could not generate visualization: {e}")
    
    # Save statistics to file
    save_statistics(all_stats)
    
    print("\n" + "=" * 80)
    print("Task 1 completed successfully!")
    print(f"Processed {len(mesh_files)} meshes")
    print(f"Visualizations saved to {vis_dir}/")


if __name__ == "__main__":
    main()

