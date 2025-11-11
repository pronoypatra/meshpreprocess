"""
Utility functions for mesh normalization, quantization, and error analysis.
"""

import numpy as np
import trimesh
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def load_mesh(path):
    """
    Load a mesh from an .obj file using trimesh.
    
    Args:
        path: Path to the .obj file
        
    Returns:
        trimesh.Trimesh: Loaded mesh object
    """
    mesh = trimesh.load(path)
    if isinstance(mesh, trimesh.Scene):
        # If scene, get the first mesh
        mesh = list(mesh.geometry.values())[0]
    return mesh


def minmax_normalize(vertices):
    """
    Apply Min-Max normalization to vertices, bringing them into [0, 1] range.
    
    Args:
        vertices: numpy array of shape (N, 3) containing vertex coordinates
        
    Returns:
        tuple: (normalized_vertices, min_vals, max_vals) where min_vals and max_vals
               are needed for denormalization
    """
    vertices = np.array(vertices, dtype=np.float64)
    min_vals = vertices.min(axis=0)
    max_vals = vertices.max(axis=0)
    
    # Handle case where min == max (constant values)
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1.0  # Avoid division by zero
    
    normalized = (vertices - min_vals) / ranges
    
    return normalized, min_vals, max_vals


def denormalize_minmax(normalized_vertices, min_vals, max_vals):
    """
    Reverse Min-Max normalization.
    
    Args:
        normalized_vertices: numpy array of normalized vertices in [0, 1]
        min_vals: Minimum values for each axis (x, y, z)
        max_vals: Maximum values for each axis (x, y, z)
        
    Returns:
        numpy array: Denormalized vertices
    """
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1.0  # Avoid division by zero
    denormalized = normalized_vertices * ranges + min_vals
    return denormalized


def unitsphere_normalize(vertices):
    """
    Apply Unit Sphere normalization to vertices.
    Centers the mesh and scales it so all vertices fit inside a sphere of radius 1.
    
    Args:
        vertices: numpy array of shape (N, 3) containing vertex coordinates
        
    Returns:
        tuple: (normalized_vertices, center, scale) where center and scale
               are needed for denormalization
    """
    vertices = np.array(vertices, dtype=np.float64)
    
    # Center the mesh
    center = vertices.mean(axis=0)
    centered = vertices - center
    
    # Find the maximum distance from origin
    distances = np.linalg.norm(centered, axis=1)
    max_distance = distances.max()
    
    # Handle case where all vertices are at the same point
    if max_distance == 0:
        scale = 1.0
    else:
        scale = 1.0 / max_distance
    
    # Scale to fit in unit sphere
    normalized = centered * scale
    
    return normalized, center, scale


def denormalize_unitsphere(normalized_vertices, center, scale):
    """
    Reverse Unit Sphere normalization.
    
    Args:
        normalized_vertices: numpy array of normalized vertices
        center: Center point used for normalization
        scale: Scale factor used for normalization
        
    Returns:
        numpy array: Denormalized vertices
    """
    # Reverse scaling
    if scale == 0:
        scale = 1.0
    descaled = normalized_vertices / scale
    
    # Reverse centering
    denormalized = descaled + center
    
    return denormalized


def quantize(vertices, n_bins=1024):
    """
    Quantize normalized vertices into discrete bins.
    
    Args:
        vertices: numpy array of normalized vertices (should be in [0, 1] range)
        n_bins: Number of quantization bins (default: 1024)
        
    Returns:
        numpy array: Quantized vertices as integers in [0, n_bins-1]
    """
    # Clip vertices to [0, 1] range to handle any floating point errors
    vertices = np.clip(vertices, 0.0, 1.0)
    
    # Quantize: q = int(x' × (n_bins - 1))
    quantized = np.round(vertices * (n_bins - 1)).astype(np.int32)
    
    # Ensure values are within valid range
    quantized = np.clip(quantized, 0, n_bins - 1)
    
    return quantized


def dequantize(quantized_vertices, n_bins=1024):
    """
    Dequantize vertices back to normalized coordinates.
    
    Args:
        quantized_vertices: numpy array of quantized vertices (integers in [0, n_bins-1])
        n_bins: Number of quantization bins (default: 1024)
        
    Returns:
        numpy array: Dequantized vertices in [0, 1] range
    """
    # Dequantize: x' = q / (n_bins - 1)
    dequantized = quantized_vertices.astype(np.float64) / (n_bins - 1)
    return dequantized


def compute_mse(original, reconstructed):
    """
    Compute Mean Squared Error between original and reconstructed vertices.
    
    Args:
        original: numpy array of original vertices
        reconstructed: numpy array of reconstructed vertices
        
    Returns:
        dict: Dictionary containing MSE per axis (x, y, z) and overall MSE
    """
    error = original - reconstructed
    mse_per_axis = np.mean(error ** 2, axis=0)
    mse_overall = np.mean(error ** 2)
    
    return {
        'mse_x': float(mse_per_axis[0]),
        'mse_y': float(mse_per_axis[1]),
        'mse_z': float(mse_per_axis[2]),
        'mse_overall': float(mse_overall)
    }


def compute_mae(original, reconstructed):
    """
    Compute Mean Absolute Error between original and reconstructed vertices.
    
    Args:
        original: numpy array of original vertices
        reconstructed: numpy array of reconstructed vertices
        
    Returns:
        dict: Dictionary containing MAE per axis (x, y, z) and overall MAE
    """
    error = np.abs(original - reconstructed)
    mae_per_axis = np.mean(error, axis=0)
    mae_overall = np.mean(error)
    
    return {
        'mae_x': float(mae_per_axis[0]),
        'mae_y': float(mae_per_axis[1]),
        'mae_z': float(mae_per_axis[2]),
        'mae_overall': float(mae_overall)
    }


def visualize_mesh(mesh, title="Mesh", save_path=None):
    """
    Visualize a mesh using matplotlib (works in headless environments).
    
    Args:
        mesh: trimesh.Trimesh object or numpy array of vertices
        title: Title for the visualization
        save_path: Optional path to save the visualization as an image
        
    Returns:
        trimesh.Trimesh: Mesh object
    """
    # Convert to trimesh if needed
    if isinstance(mesh, np.ndarray):
        # If only vertices provided, create a point cloud
        mesh = trimesh.PointCloud(vertices=mesh)
    elif not isinstance(mesh, (trimesh.Trimesh, trimesh.PointCloud)):
        # If it's already a trimesh object, use it as is
        return mesh
    
    # If save_path is provided, render to image using matplotlib
    if save_path:
        try:
            # Create directory if needed
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Extract vertices
            vertices = mesh.vertices
            
            # Check if mesh has faces
            has_faces = (isinstance(mesh, trimesh.Trimesh) and 
                        hasattr(mesh, 'faces') and 
                        mesh.faces is not None and 
                        len(mesh.faces) > 0)
            
            if has_faces:
                # Plot mesh with faces
                faces = mesh.faces
                
                # Sample faces for large meshes to avoid memory issues
                max_faces = 10000
                if len(faces) > max_faces:
                    # Randomly sample faces
                    indices = np.random.choice(len(faces), max_faces, replace=False)
                    faces = faces[indices]
                
                # Create Poly3DCollection
                triangles = vertices[faces]
                collection = Poly3DCollection(triangles, alpha=0.7, facecolors='cyan', 
                                            edgecolors='darkblue', linewidths=0.5)
                ax.add_collection3d(collection)
            else:
                # Plot point cloud
                # Sample points if too many
                max_points = 50000
                if len(vertices) > max_points:
                    indices = np.random.choice(len(vertices), max_points, replace=False)
                    vertices = vertices[indices]
                
                ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                          c=vertices[:, 2], cmap='viridis', s=1, alpha=0.6)
            
            # Set labels and title
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(title, fontsize=12, pad=20)
            
            # Set equal aspect ratio with centered limits
            max_range = np.array([vertices[:, 0].max() - vertices[:, 0].min(),
                                 vertices[:, 1].max() - vertices[:, 1].min(),
                                 vertices[:, 2].max() - vertices[:, 2].min()]).max() / 2.0
            mid_x = (vertices[:, 0].max() + vertices[:, 0].min()) * 0.5
            mid_y = (vertices[:, 1].max() + vertices[:, 1].min()) * 0.5
            mid_z = (vertices[:, 2].max() + vertices[:, 2].min()) * 0.5
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
            
            # Save figure
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
        except Exception as e:
            print(f"Warning: Could not create visualization: {e}")
            # Create a simple fallback visualization
            try:
                fig, ax = plt.subplots(figsize=(8, 6))
                vertices = mesh.vertices
                # Create 2D projection
                ax.scatter(vertices[:, 0], vertices[:, 1], s=1, alpha=0.5)
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_title(f"{title} (2D projection)")
                plt.tight_layout()
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
            except Exception as e2:
                print(f"Warning: Fallback visualization also failed: {e2}")
    
    return mesh


def save_mesh(vertices, faces, output_path):
    """
    Save vertices and faces to an .obj file.
    
    Args:
        vertices: numpy array of vertices
        faces: numpy array of face indices (or None for point cloud)
        output_path: Path to save the .obj file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Create trimesh object
    if faces is not None and len(faces) > 0:
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    else:
        # If no faces, create a point cloud (trimesh can still save it)
        mesh = trimesh.PointCloud(vertices=vertices)
    
    # Export to .obj
    mesh.export(output_path)

