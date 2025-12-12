#!/usr/bin/env python3
"""로컬 3D 뷰어 - trimesh 사용"""

import trimesh
import numpy as np
import os

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'ZOUIF2W4')
UPPER_PATH = os.path.join(ASSETS_DIR, 'ZOUIF2W4_upper.obj')
LOWER_PATH = os.path.join(ASSETS_DIR, 'ZOUIF2W4_lower.obj')

def main():
    print("Loading meshes...")

    # Load meshes
    upper = trimesh.load(UPPER_PATH)
    lower = trimesh.load(LOWER_PATH)

    # Set colors
    if hasattr(upper, 'visual'):
        upper.visual.face_colors = [200, 200, 180, 255]  # 상악 - 밝은 베이지
    if hasattr(lower, 'visual'):
        lower.visual.face_colors = [180, 200, 220, 255]  # 하악 - 연한 파랑

    print(f"Upper: {len(upper.vertices)} vertices, {len(upper.faces)} faces")
    print(f"Lower: {len(lower.vertices)} vertices, {len(lower.faces)} faces")

    # Create scene
    scene = trimesh.Scene()
    scene.add_geometry(upper, node_name='maxilla')
    scene.add_geometry(lower, node_name='mandible')

    print("Opening viewer... (close window to exit)")
    scene.show()

if __name__ == "__main__":
    main()
