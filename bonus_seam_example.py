"""
Bonus Task: Seam Tokenization Example

This script demonstrates seam detection and tokenization on sample meshes.
"""

import os
import glob
import numpy as np
from seam_tokenization import (
    load_mesh_with_uv, detect_seams, build_seam_paths,
    encode_seams, decode_seam_tokens, tokens_to_string
)
from mesh_utils import load_mesh, visualize_mesh


def process_mesh_seams(mesh_path: str, output_dir: str = "output/seam_tokenization"):
    """
    Process a single mesh: detect seams, tokenize, and demonstrate encoding/decoding.
    
    Args:
        mesh_path: Path to .obj mesh file
        output_dir: Output directory for results
    """
    print(f"\n{'='*80}")
    print(f"Processing: {mesh_path}")
    print(f"{'='*80}")
    
    filename = os.path.basename(mesh_path)
    basename = os.path.splitext(filename)[0]
    
    # Load mesh with UV coordinates
    mesh, uv_coords, face_uv_indices = load_mesh_with_uv(mesh_path)
    
    print(f"\nMesh Statistics:")
    print(f"  Vertices: {len(mesh.vertices)}")
    print(f"  Faces: {len(mesh.faces)}")
    print(f"  UV coordinates: {len(uv_coords) if uv_coords is not None else 0}")
    print(f"  Face UV indices: {len(face_uv_indices) if face_uv_indices is not None else 0}")
    
    if uv_coords is None or face_uv_indices is None:
        print(f"\n  ⚠️  No UV coordinates found. Skipping seam detection.")
        return None
    
    # Detect seams (pass obj_path for direct parsing)
    print(f"\nDetecting seams...")
    seam_edges = detect_seams(mesh, uv_coords, face_uv_indices, obj_path=mesh_path)
    print(f"  Found {len(seam_edges)} seam edges")
    
    if len(seam_edges) == 0:
        print(f"  ⚠️  No seams detected. Mesh may have continuous UV mapping.")
        return None
    
    # Build seam paths
    print(f"\nBuilding seam paths...")
    seam_paths = build_seam_paths(seam_edges)
    print(f"  Found {len(seam_paths)} seam paths")
    
    for i, path in enumerate(seam_paths):
        print(f"    Path {i}: {len(path)} vertices")
    
    # Encode seams to tokens
    print(f"\nEncoding seams to tokens...")
    token_sequences = encode_seams(seam_paths)
    print(f"  Generated {len(token_sequences)} token sequences")
    
    total_tokens = sum(len(seq) for seq in token_sequences)
    print(f"  Total tokens: {total_tokens}")
    
    # Display example token sequences
    print(f"\nExample Token Sequences:")
    for i, tokens in enumerate(token_sequences[:3]):  # Show first 3
        token_str = tokens_to_string(tokens)
        print(f"  Path {i}: {token_str}")
        if len(token_str) > 100:
            print(f"    (truncated, full length: {len(tokens)} tokens)")
    
    # Decode tokens back to paths
    print(f"\nDecoding tokens...")
    decoded_paths = []
    for tokens in token_sequences:
        try:
            seam_id, path = decode_seam_tokens(tokens)
            decoded_paths.append(path)
            print(f"  Path {seam_id}: Decoded {len(path)} vertices")
        except Exception as e:
            print(f"  Error decoding tokens: {e}")
    
    # Verify decoding
    print(f"\nVerification:")
    if len(decoded_paths) == len(seam_paths):
        all_match = True
        for orig, dec in zip(seam_paths, decoded_paths):
            if orig != dec:
                print(f"  ⚠️  Path mismatch detected")
                all_match = False
                break
        if all_match:
            print(f"  ✓ All paths decoded correctly")
    else:
        print(f"  ⚠️  Path count mismatch: {len(seam_paths)} original, {len(decoded_paths)} decoded")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    # Save token sequences to file
    token_file = os.path.join(output_dir, f"{basename}_tokens.txt")
    with open(token_file, 'w') as f:
        f.write(f"Seam Tokenization Results for {filename}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Mesh: {filename}\n")
        f.write(f"Vertices: {len(mesh.vertices)}\n")
        f.write(f"Faces: {len(mesh.faces)}\n")
        f.write(f"Seam edges: {len(seam_edges)}\n")
        f.write(f"Seam paths: {len(seam_paths)}\n")
        f.write(f"Total tokens: {total_tokens}\n\n")
        f.write("Token Sequences:\n")
        f.write("-" * 80 + "\n")
        for i, tokens in enumerate(token_sequences):
            f.write(f"\nPath {i}:\n")
            f.write(f"  Token sequence: {tokens_to_string(tokens)}\n")
            f.write(f"  Vertex path: {seam_paths[i]}\n")
            f.write(f"  Length: {len(seam_paths[i])} vertices, {len(tokens)} tokens\n")
    
    print(f"\n  Results saved to {token_file}")
    
    # Create visualization
    try:
        vis_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        # Visualize original mesh
        vis_path = os.path.join(vis_dir, f"{basename}_mesh.png")
        visualize_mesh(mesh, title=f"{basename} - Mesh", save_path=vis_path)
        print(f"  Visualization saved to {vis_path}")
    except Exception as e:
        print(f"  Warning: Could not create visualization: {e}")
    
    return {
        'mesh': mesh,
        'seam_edges': seam_edges,
        'seam_paths': seam_paths,
        'token_sequences': token_sequences,
        'decoded_paths': decoded_paths
    }


def main():
    """Main function to process all meshes"""
    print("=" * 80)
    print("Seam Tokenization Prototype")
    print("=" * 80)
    
    # Get all .obj files from 8samples folder
    mesh_dir = '8samples'
    mesh_files = glob.glob(os.path.join(mesh_dir, '*.obj'))
    mesh_files.sort()
    
    if not mesh_files:
        print(f"Error: No .obj files found in {mesh_dir}/")
        return
    
    print(f"\nFound {len(mesh_files)} mesh files")
    
    results = []
    
    # Process each mesh
    for mesh_path in mesh_files:
        try:
            result = process_mesh_seams(mesh_path)
            if result is not None:
                results.append((os.path.basename(mesh_path), result))
        except Exception as e:
            print(f"\n  ⚠️  Error processing {mesh_path}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print(f"\n{'='*80}")
    print("Summary")
    print(f"{'='*80}")
    print(f"Processed {len(results)} meshes with seams")
    
    if results:
        print(f"\nMesh Seam Statistics:")
        for filename, result in results:
            print(f"  {filename}:")
            print(f"    Seam edges: {len(result['seam_edges'])}")
            print(f"    Seam paths: {len(result['seam_paths'])}")
            total_tokens = sum(len(seq) for seq in result['token_sequences'])
            print(f"    Total tokens: {total_tokens}")
    
    print(f"\n{'='*80}")
    print("Seam tokenization completed!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()


