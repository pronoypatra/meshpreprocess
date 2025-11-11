"""
Seam Tokenization Prototype

This module implements seam detection and tokenization for 3D meshes.
Seams are edges where UV texture mappings break (same 3D edge has different UV coordinates).
"""

import numpy as np
import trimesh
import os
from collections import defaultdict
from typing import List, Dict, Tuple, Set, Optional


# Token vocabulary
class TokenType:
    """Token types for seam encoding"""
    SEAM_START = 0
    SEAM_END = 1
    VERTEX = 2
    EDGE_FWD = 3  # Forward edge direction
    EDGE_BWD = 4  # Backward edge direction
    UV_BREAK = 5  # UV discontinuity marker
    SEAM_ID = 6   # Seam path identifier


TOKEN_NAMES = {
    TokenType.SEAM_START: "SEAM_START",
    TokenType.SEAM_END: "SEAM_END",
    TokenType.VERTEX: "VERTEX",
    TokenType.EDGE_FWD: "EDGE_FWD",
    TokenType.EDGE_BWD: "EDGE_BWD",
    TokenType.UV_BREAK: "UV_BREAK",
    TokenType.SEAM_ID: "SEAM_ID",
}


def load_mesh_with_uv(mesh_path: str) -> Tuple[trimesh.Trimesh, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load mesh and extract UV coordinates from OBJ file.
    
    Args:
        mesh_path: Path to .obj file
        
    Returns:
        tuple: (mesh, uv_coordinates, face_uv_indices) where:
            - mesh: trimesh.Trimesh object
            - uv_coordinates: (n_uv, 2) array of UV coordinates
            - face_uv_indices: (n_faces, 3) array of UV indices per face vertex
    """
    mesh = trimesh.load(mesh_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = list(mesh.geometry.values())[0]
    
    # Try to get UV coordinates from mesh visual
    uv_coords = None
    face_uv_indices = None
    
    if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
        uv_coords = mesh.visual.uv
        # If UV coordinates exist, create face UV indices
        # This is a simplified approach - in reality, OBJ files can have
        # per-face UV mappings that differ from vertex UV mappings
        if len(uv_coords) == len(mesh.vertices):
            face_uv_indices = mesh.faces.copy()
    
    # If no UV coordinates in visual, try to parse from OBJ file directly
    if uv_coords is None:
        uv_coords, face_uv_indices = _parse_uv_from_obj(mesh_path)
    
    return mesh, uv_coords, face_uv_indices


def _parse_uv_from_obj(obj_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Parse UV coordinates and face UV indices directly from OBJ file.
    
    Args:
        obj_path: Path to .obj file
        
    Returns:
        tuple: (uv_coordinates, face_uv_indices)
    """
    uv_coords = []
    face_uv_indices = []
    vertex_count = 0
    
    try:
        with open(obj_path, 'r') as f:
            lines = f.readlines()
            
        # First pass: collect UV coordinates
        for line in lines:
            line = line.strip()
            if line.startswith('vt '):
                parts = line.split()[1:]
                if len(parts) >= 2:
                    uv_coords.append([float(parts[0]), float(parts[1])])
            elif line.startswith('v '):
                vertex_count += 1
        
        # Second pass: collect face UV indices
        for line in lines:
            line = line.strip()
            if line.startswith('f '):
                parts = line.split()[1:]
                face_uv = []
                has_uv = False
                for part in parts:
                    # Format: v/vt/vn or v//vn or v/vt
                    vertex_parts = part.split('/')
                    if len(vertex_parts) >= 2 and vertex_parts[1]:
                        # UV index (1-based in OBJ, convert to 0-based)
                        uv_idx = int(vertex_parts[1]) - 1
                        face_uv.append(uv_idx)
                        has_uv = True
                    else:
                        # No UV coordinate for this vertex
                        face_uv.append(-1)
                
                if has_uv and len(face_uv) >= 3:
                    # Store only first 3 vertices (triangles)
                    face_uv_indices.append(face_uv[:3])
        
        if uv_coords and face_uv_indices:
            return np.array(uv_coords), np.array(face_uv_indices)
        
    except Exception as e:
        print(f"Warning: Could not parse UV from OBJ file: {e}")
    
    return None, None


def get_edge_key(v1: int, v2: int) -> Tuple[int, int]:
    """Get canonical edge key (smaller vertex first)"""
    return (min(v1, v2), max(v1, v2))


def detect_seams_from_obj(obj_path: str, uv_tolerance: float = 1e-6) -> Set[Tuple[int, int]]:
    """
    Detect seam edges directly from OBJ file by comparing UV coordinates.
    
    A seam edge is detected when the same 3D edge (same vertex indices) appears
    in different faces with different UV coordinates.
    
    Args:
        obj_path: Path to OBJ file
        uv_tolerance: Tolerance for UV coordinate comparison
        
    Returns:
        set: Set of seam edges as (v1, v2) tuples (vertex indices from OBJ)
    """
    # Parse OBJ file to get vertices, UVs, and faces
    vertices = []
    uv_coords = []
    faces = []  # List of (vertex_indices, uv_indices) per face
    
    try:
        with open(obj_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('v '):
                    # Vertex position
                    parts = line.split()[1:]
                    if len(parts) >= 3:
                        vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])
                elif line.startswith('vt '):
                    # UV coordinate
                    parts = line.split()[1:]
                    if len(parts) >= 2:
                        uv_coords.append([float(parts[0]), float(parts[1])])
                elif line.startswith('f '):
                    # Face with vertex/uv/normal indices
                    parts = line.split()[1:]
                    face_verts = []
                    face_uvs = []
                    for part in parts:
                        vertex_parts = part.split('/')
                        if len(vertex_parts) >= 1:
                            # Vertex index (1-based in OBJ, convert to 0-based)
                            v_idx = int(vertex_parts[0]) - 1
                            face_verts.append(v_idx)
                            
                            # UV index (if present)
                            if len(vertex_parts) >= 2 and vertex_parts[1]:
                                uv_idx = int(vertex_parts[1]) - 1
                                face_uvs.append(uv_idx)
                            else:
                                face_uvs.append(-1)
                    
                    if len(face_verts) >= 3 and len(face_uvs) >= 3:
                        # Triangulate if needed (take first 3 vertices)
                        faces.append((face_verts[:3], face_uvs[:3]))
    except Exception as e:
        print(f"Error parsing OBJ file: {e}")
        return set()
    
    if not uv_coords or not faces:
        return set()
    
    uv_coords = np.array(uv_coords)
    
    # Build edge-to-faces mapping: edge (v1, v2) -> list of (face_idx, uv1, uv2)
    edge_faces = defaultdict(list)
    
    for face_idx, (face_verts, face_uvs) in enumerate(faces):
        # Check each edge of the triangle
        for i in range(3):
            v1_idx = face_verts[i]
            v2_idx = face_verts[(i + 1) % 3]
            uv1_idx = face_uvs[i]
            uv2_idx = face_uvs[(i + 1) % 3]
            
            # Canonical edge key
            edge = get_edge_key(v1_idx, v2_idx)
            
            # Store UV indices for this edge in this face
            if uv1_idx >= 0 and uv2_idx >= 0:
                edge_faces[edge].append((face_idx, uv1_idx, uv2_idx, v1_idx < v2_idx))
    
    # Find seams: edges where UV coordinates differ across faces
    seam_edges = set()
    edges_checked = 0
    edges_with_multiple_faces = 0
    
    for edge, face_data in edge_faces.items():
        if len(face_data) < 2:
            # Boundary edge or edge with only one face
            continue
        
        edges_with_multiple_faces += 1
        edges_checked += 1
        
        # Collect UV pairs for this edge
        uv_pairs = []
        for face_idx, uv1_idx, uv2_idx, is_forward in face_data:
            if uv1_idx >= 0 and uv2_idx >= 0 and uv1_idx < len(uv_coords) and uv2_idx < len(uv_coords):
                uv1 = uv_coords[uv1_idx]
                uv2 = uv_coords[uv2_idx]
                # Store UV pair in canonical order (always v1 < v2)
                if is_forward:
                    uv_pairs.append((uv1, uv2))
                else:
                    # Reverse order to match canonical edge direction
                    uv_pairs.append((uv2, uv1))
        
        # Check if UV pairs differ
        if len(uv_pairs) >= 2:
            is_seam = False
            base_uv1, base_uv2 = uv_pairs[0]
            
            for uv1, uv2 in uv_pairs[1:]:
                # Check if UV coordinates differ significantly
                dist1 = np.linalg.norm(uv1 - base_uv1)
                dist2 = np.linalg.norm(uv2 - base_uv2)
                if dist1 > uv_tolerance or dist2 > uv_tolerance:
                    is_seam = True
                    break
            
            if is_seam:
                seam_edges.add(edge)
    
    # Debug output (commented out for production, but useful for debugging)
    # if edges_with_multiple_faces == 0:
    #     print(f"Debug: No edges shared by multiple faces found")
    # else:
    #     print(f"Debug: Checked {edges_checked} edges, {edges_with_multiple_faces} shared by multiple faces, found {len(seam_edges)} seams")
    
    return seam_edges


def detect_seams(mesh: trimesh.Trimesh, uv_coords: Optional[np.ndarray], 
                 face_uv_indices: Optional[np.ndarray], 
                 obj_path: Optional[str] = None,
                 uv_tolerance: float = 1e-6) -> Set[Tuple[int, int]]:
    """
    Detect seam edges where UV mappings break.
    
    This function first tries to detect seams directly from the OBJ file if provided,
    which is more reliable than using the loaded mesh (which may have merged vertices).
    
    Args:
        mesh: trimesh.Trimesh object
        uv_coords: (n_uv, 2) array of UV coordinates
        face_uv_indices: (n_faces, 3) array of UV indices per face vertex
        obj_path: Optional path to OBJ file for direct parsing
        uv_tolerance: Tolerance for UV coordinate comparison
        
    Returns:
        set: Set of seam edges as (v1, v2) tuples (canonical order)
    """
    # Try OBJ file parsing first (more reliable)
    if obj_path and os.path.exists(obj_path):
        try:
            seams = detect_seams_from_obj(obj_path, uv_tolerance)
            if len(seams) > 0:
                return seams
        except Exception as e:
            print(f"Warning: Could not detect seams from OBJ file: {e}")
    
    # Fallback to mesh-based detection
    if uv_coords is None or face_uv_indices is None:
        return set()
    
    if len(face_uv_indices) != len(mesh.faces):
        return set()
    
    # Build edge-to-faces mapping using mesh faces and UV indices
    edge_faces = defaultdict(list)
    
    for face_idx, face in enumerate(mesh.faces):
        face_uv = face_uv_indices[face_idx]
        
        for i in range(3):
            v1 = face[i]
            v2 = face[(i + 1) % 3]
            edge = get_edge_key(v1, v2)
            
            uv1_idx = face_uv[i]
            uv2_idx = face_uv[(i + 1) % 3]
            
            if uv1_idx >= 0 and uv2_idx >= 0:
                edge_faces[edge].append((face_idx, uv1_idx, uv2_idx, v1 < v2))
    
    # Find seams
    seam_edges = set()
    for edge, face_data in edge_faces.items():
        if len(face_data) < 2:
            continue
        
        uv_pairs = []
        for face_idx, uv1_idx, uv2_idx, is_forward in face_data:
            if uv1_idx < len(uv_coords) and uv2_idx < len(uv_coords):
                uv1 = uv_coords[uv1_idx]
                uv2 = uv_coords[uv2_idx]
                if is_forward:
                    uv_pairs.append((uv1, uv2))
                else:
                    uv_pairs.append((uv2, uv1))
        
        if len(uv_pairs) >= 2:
            base_uv1, base_uv2 = uv_pairs[0]
            for uv1, uv2 in uv_pairs[1:]:
                if (np.linalg.norm(uv1 - base_uv1) > uv_tolerance or 
                    np.linalg.norm(uv2 - base_uv2) > uv_tolerance):
                    seam_edges.add(edge)
                    break
    
    return seam_edges


def build_seam_graph(seam_edges: Set[Tuple[int, int]]) -> Dict[int, List[int]]:
    """
    Build graph of seam edges for path traversal.
    
    Args:
        seam_edges: Set of seam edges as (v1, v2) tuples
        
    Returns:
        dict: Graph as {vertex: [neighbor_vertices]} adjacency list
    """
    graph = defaultdict(list)
    
    for v1, v2 in seam_edges:
        graph[v1].append(v2)
        graph[v2].append(v1)
    
    return dict(graph)


def build_seam_paths(seam_edges: Set[Tuple[int, int]]) -> List[List[int]]:
    """
    Group seam edges into connected paths using Eulerian path algorithm.
    
    Args:
        seam_edges: Set of seam edges as (v1, v2) tuples
        
    Returns:
        list: List of seam paths, each path is a list of vertex indices
    """
    if not seam_edges:
        return []
    
    # Build graph (make a copy with lists for modification)
    graph = build_seam_graph(seam_edges)
    graph = {v: list(neighbors) for v, neighbors in graph.items()}
    
    paths = []
    used_edges = set()
    
    def find_path(start_vertex: int) -> List[int]:
        """Find a path starting from a vertex using available edges"""
        path = [start_vertex]
        current = start_vertex
        
        while current in graph and graph[current]:
            # Find an unused edge from current vertex
            next_vertex = None
            neighbors_to_remove = []
            
            for neighbor in graph[current]:
                edge = get_edge_key(current, neighbor)
                if edge not in used_edges:
                    next_vertex = neighbor
                    used_edges.add(edge)
                    neighbors_to_remove.append(neighbor)
                    break
            
            if next_vertex is None:
                break
            
            # Remove edge from graph (both directions)
            for neighbor in neighbors_to_remove:
                if neighbor in graph[current]:
                    graph[current].remove(neighbor)
                if neighbor in graph and current in graph[neighbor]:
                    graph[neighbor].remove(current)
            
            path.append(next_vertex)
            current = next_vertex
        
        return path
    
    # Find vertices with odd degree (endpoints)
    degree = {v: len(graph.get(v, [])) for v in graph}
    endpoints = [v for v, d in degree.items() if d % 2 == 1]
    
    # Start from endpoints first
    for start_vertex in endpoints:
        if start_vertex in graph and graph[start_vertex]:
            path = find_path(start_vertex)
            if len(path) >= 2:
                paths.append(path)
    
    # Process remaining cycles (all vertices have even degree)
    while any(graph.get(v, []) for v in graph):
        # Find any vertex with remaining edges
        start_vertex = None
        for vertex, neighbors in graph.items():
            if neighbors:
                start_vertex = vertex
                break
        
        if start_vertex is None:
            break
        
        path = find_path(start_vertex)
        if len(path) >= 2:
            paths.append(path)
    
    return paths


def encode_seam_path(path: List[int], seam_id: int) -> List[int]:
    """
    Encode a seam path into a token sequence.
    
    Token format:
    [SEAM_START, SEAM_ID, seam_id, VERTEX, v0, VERTEX, v1, ..., VERTEX, vn, SEAM_END]
    
    Args:
        path: List of vertex indices along the seam path
        seam_id: Identifier for this seam path
        
    Returns:
        list: Token sequence as integers
    """
    tokens = [TokenType.SEAM_START, TokenType.SEAM_ID, seam_id]
    
    for vertex in path:
        tokens.append(TokenType.VERTEX)
        tokens.append(vertex)
    
    tokens.append(TokenType.SEAM_END)
    
    return tokens


def encode_seams(seam_paths: List[List[int]]) -> List[List[int]]:
    """
    Encode all seam paths into token sequences.
    
    Args:
        seam_paths: List of seam paths, each path is a list of vertex indices
        
    Returns:
        list: List of token sequences, one per seam path
    """
    token_sequences = []
    
    for seam_id, path in enumerate(seam_paths):
        tokens = encode_seam_path(path, seam_id)
        token_sequences.append(tokens)
    
    return token_sequences


def decode_seam_tokens(tokens: List[int]) -> Tuple[int, List[int]]:
    """
    Decode a token sequence back into a seam path.
    
    Args:
        tokens: Token sequence as integers
        
    Returns:
        tuple: (seam_id, path) where path is a list of vertex indices
    """
    if len(tokens) < 4:
        raise ValueError("Invalid token sequence: too short")
    
    if tokens[0] != TokenType.SEAM_START:
        raise ValueError("Invalid token sequence: must start with SEAM_START")
    
    if tokens[-1] != TokenType.SEAM_END:
        raise ValueError("Invalid token sequence: must end with SEAM_END")
    
    # Extract seam ID
    if tokens[1] != TokenType.SEAM_ID:
        raise ValueError("Invalid token sequence: expected SEAM_ID after SEAM_START")
    
    seam_id = tokens[2]
    
    # Extract vertices
    path = []
    i = 3
    while i < len(tokens) - 1:
        if tokens[i] == TokenType.VERTEX:
            if i + 1 < len(tokens) - 1:
                vertex = tokens[i + 1]
                path.append(vertex)
                i += 2
            else:
                break
        else:
            i += 1
    
    return seam_id, path


def tokens_to_string(tokens: List[int]) -> str:
    """
    Convert token sequence to human-readable string.
    
    Args:
        tokens: Token sequence as integers
        
    Returns:
        str: Human-readable token sequence
    """
    parts = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token in TOKEN_NAMES:
            parts.append(TOKEN_NAMES[token])
            if token == TokenType.SEAM_ID and i + 1 < len(tokens):
                parts.append(str(tokens[i + 1]))
                i += 1
            elif token == TokenType.VERTEX and i + 1 < len(tokens):
                parts.append(str(tokens[i + 1]))
                i += 1
        else:
            parts.append(str(token))
        i += 1
    
    return " ".join(parts)


def get_seam_vertices(seam_edges: Set[Tuple[int, int]]) -> Set[int]:
    """
    Get all vertices that are part of seam edges.
    
    Args:
        seam_edges: Set of seam edges
        
    Returns:
        set: Set of vertex indices that are part of seams
    """
    seam_vertices = set()
    for v1, v2 in seam_edges:
        seam_vertices.add(v1)
        seam_vertices.add(v2)
    return seam_vertices

