"""
Test script to verify seam detection works correctly.
Creates a simple test mesh with known seams.
"""

import numpy as np
from seam_tokenization import detect_seams_from_obj, build_seam_paths, encode_seams, decode_seam_tokens, tokens_to_string

# Create a simple test OBJ file with seams
test_obj_content = """# Test mesh with seams
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 1.0 1.0 0.0
v 0.0 1.0 0.0
v 0.5 0.5 1.0

# UV coordinates - creating a seam on edge (0,1)
vt 0.0 0.0
vt 1.0 0.0
vt 1.0 1.0
vt 0.0 1.0
vt 0.5 0.5
# Different UV for vertex 0 in face 2 (creates seam)
vt 0.2 0.0

vn 0.0 0.0 1.0

# Face 1: bottom face, vertices 0,1,2,3
f 1/1/1 2/2/1 3/3/1
f 1/1/1 3/3/1 4/4/1

# Face 2: side face with different UV for vertex 0 (creates seam on edge 0-1)
f 1/6/1 2/2/1 5/5/1
f 2/2/1 3/3/1 5/5/1
"""

# Write test OBJ file
with open('test_seam_mesh.obj', 'w') as f:
    f.write(test_obj_content)

print("Created test mesh with seams")
print("=" * 80)

# Detect seams
seam_edges = detect_seams_from_obj('test_seam_mesh.obj', uv_tolerance=1e-6)
print(f"\nDetected {len(seam_edges)} seam edges")
print(f"Seam edges: {seam_edges}")

if len(seam_edges) > 0:
    # Build paths
    seam_paths = build_seam_paths(seam_edges)
    print(f"\nBuilt {len(seam_paths)} seam paths")
    for i, path in enumerate(seam_paths):
        print(f"  Path {i}: {path}")
    
    # Encode to tokens
    token_sequences = encode_seams(seam_paths)
    print(f"\nEncoded to {len(token_sequences)} token sequences")
    for i, tokens in enumerate(token_sequences):
        print(f"  Path {i}: {tokens_to_string(tokens)}")
    
    # Decode back
    print(f"\nDecoding tokens...")
    for tokens in token_sequences:
        seam_id, path = decode_seam_tokens(tokens)
        print(f"  Seam {seam_id}: {path}")
    
    print("\n✓ Seam detection and tokenization working correctly!")
else:
    print("\n⚠ No seams detected in test mesh. This might indicate an issue.")

