from __future__ import annotations

from functools import lru_cache

import torch


_RAY_START_HEIGHT = 100.0
_FOOTPRINT_RADIUS = 0.08


@lru_cache(maxsize=8)
def _load_reference_terrain_mesh(terrain_prim_path: str, device: str) -> tuple[object, bool]:
    import numpy as np
    import omni.usd
    from pxr import UsdGeom

    import isaaclab.sim as sim_utils
    from isaaclab.terrains.trimesh.utils import make_plane
    from isaaclab.utils.warp import convert_to_warp_mesh

    matching_prim_paths = sim_utils.find_matching_prim_paths(terrain_prim_path)
    if len(matching_prim_paths) == 0:
        raise RuntimeError(f"No terrain prims matched the path expression: {terrain_prim_path}")

    reference_prim_path = next(
        (prim_path for prim_path in matching_prim_paths if "/env_0/" in prim_path),
        matching_prim_paths[0],
    )
    terrain_prim = sim_utils.get_first_matching_child_prim(
        reference_prim_path,
        lambda prim: prim.GetTypeName() == "Plane",
    )
    is_env_0_reference = "/env_0/" in reference_prim_path

    if terrain_prim is None:
        terrain_prim = sim_utils.get_first_matching_child_prim(
            reference_prim_path,
            lambda prim: prim.GetTypeName() == "Mesh",
        )
        if terrain_prim is None or not terrain_prim.IsValid():
            raise RuntimeError(f"Invalid terrain prim path: {reference_prim_path}")

        terrain_prim = UsdGeom.Mesh(terrain_prim)
        points = np.asarray(terrain_prim.GetPointsAttr().Get())
        transform_matrix = np.array(omni.usd.get_world_transform_matrix(terrain_prim)).T
        points = np.matmul(points, transform_matrix[:3, :3].T)
        points += transform_matrix[:3, 3]
        indices = np.asarray(terrain_prim.GetFaceVertexIndicesAttr().Get())
        wp_mesh = convert_to_warp_mesh(points, indices, device=device)
    else:
        mesh = make_plane(size=(2e6, 2e6), height=0.0, center_zero=True)
        wp_mesh = convert_to_warp_mesh(mesh.vertices, mesh.faces, device=device)

    return wp_mesh, is_env_0_reference


def terrain_surface_heights(
    terrain_prim_path: str,
    positions_w: torch.Tensor,
    env_origins: torch.Tensor,
    env_ids: torch.Tensor,
    *,
    footprint_radius: float = _FOOTPRINT_RADIUS,
    ray_start_height: float = _RAY_START_HEIGHT,
) -> torch.Tensor:
    '计算地形表面高度'
    from isaaclab.utils.warp import raycast_mesh

    if positions_w.ndim != 2 or positions_w.shape[-1] != 3:
        raise ValueError(
            f"positions_w must have shape (N, 3), got {tuple(positions_w.shape)}"
        )

    env_ids = env_ids.to(device=positions_w.device, dtype=torch.long)
    wp_mesh, is_env_0_reference = _load_reference_terrain_mesh(
        terrain_prim_path, str(positions_w.device)
    )

    if is_env_0_reference:
        reference_origin = env_origins[0].to(device=positions_w.device, dtype=positions_w.dtype)
    else:
        reference_origin = torch.zeros(3, device=positions_w.device, dtype=positions_w.dtype)

    current_origins = env_origins[env_ids].to(device=positions_w.device, dtype=positions_w.dtype)
    delta = reference_origin.unsqueeze(0) - current_origins

    offsets = torch.tensor(
        (
            (0.0, 0.0),
            (footprint_radius, 0.0),
            (-footprint_radius, 0.0),
            (0.0, footprint_radius),
            (0.0, -footprint_radius),
        ),
        device=positions_w.device,
        dtype=positions_w.dtype,
    )

    query_xy = positions_w[:, None, :2] + delta[:, None, :2] + offsets[None, :, :]
    query_z = torch.full(
        (positions_w.shape[0], offsets.shape[0], 1),
        ray_start_height,
        device=positions_w.device,
        dtype=positions_w.dtype,
    ) + delta[:, None, 2:3]

    ray_starts = torch.cat((query_xy, query_z), dim=-1).reshape(-1, 3)
    ray_directions = torch.zeros_like(ray_starts)
    ray_directions[:, 2] = -1.0

    ray_hits = raycast_mesh(ray_starts, ray_directions, wp_mesh)[0].view(
        positions_w.shape[0], offsets.shape[0], 3
    )
    heights_w = ray_hits[..., 2] - delta[:, None, 2]
    heights_w = torch.where(torch.isfinite(heights_w), heights_w, positions_w[:, None, 2])

    return heights_w.max(dim=1).values