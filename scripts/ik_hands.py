#!/usr/bin/env python3
"""IK hands v3: wrist ring matching + target validation + normal-based anti-penetration.

Optimizes shoulder/elbow/wrist rotations (body_pose joints 15-20) so that:
  1. A ring of SMPL-X wrist vertices matches the depth-corrected MANO wrist ring
     (constrains both wrist position AND orientation)
  2. Arm/hand vertices don't penetrate the torso (surface-normal check)
  3. Wrist targets that are inside the torso are pushed to the surface pre-IK

Hand pose (finger articulation) is NOT modified — kept from HaMeR merge step.

Memory-safe: chunks FK passes and IK gradient accumulation to handle 10K+ frame videos.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import smplx
import torch

from hand_utils import load_hand_meshes, place_hands_dc
from utils import load_mano_joint_regressor, one_euro_filter_np, smplx_forward_chunked


# ---------------------------------------------------------------------------
# Wrist ring utilities
# ---------------------------------------------------------------------------

def compute_wrist_ring_ids(mano_J: torch.Tensor, n_ring: int = 15) -> torch.Tensor:
    """Find MANO vertex indices closest to wrist joint (J[0]) by regressor weight.

    Returns (n_ring,) int64 tensor of MANO vertex indices.
    """
    _, ids = torch.topk(mano_J[0], n_ring)
    return ids.long()


def compute_knuckle_ids(mano_J: torch.Tensor, n_per_knuckle: int = 1) -> torch.Tensor:
    """Find MANO vertex indices at the metacarpal knuckles (J[1], J[4], J[7], J[10], J[13])."""
    mcp_joints = [1, 4, 7, 10, 13]
    ids = []
    for j in mcp_joints:
        _, idx = torch.topk(mano_J[j], n_per_knuckle)
        ids.append(idx.long())
    return torch.cat(ids, dim=0)


def compute_fingertip_ids(mano_J: torch.Tensor, n_per_tip: int = 3) -> torch.Tensor:
    """Find MANO vertex indices at the fingertips (J[3], J[6], J[9], J[12], J[15]).

    Fingertips are the points furthest from the wrist — maximum leverage for
    constraining wrist orientation. Especially good for palm roll.

    Returns (5 * n_per_tip,) int64 tensor of MANO vertex indices.
    """
    # MANO joint indices: 3=index_tip, 6=middle_tip, 9=ring_tip, 12=pinky_tip, 15=thumb_tip
    tip_joints = [3, 6, 9, 12, 15]
    ids = []
    for j in tip_joints:
        _, idx = torch.topk(mano_J[j], n_per_tip)
        ids.append(idx.long())
    return torch.cat(ids, dim=0)


def compute_anchor_ids(mano_J: torch.Tensor, n_ring: int = 15,
                       n_per_knuckle: int = 1, n_per_tip: int = 3) -> torch.Tensor:
    """Combined wrist ring + knuckle + fingertip anchors for full hand orientation constraint.

    Ring constrains wrist position + flexion/abduction.
    Knuckles constrain palm scale + nearby roll.
    Fingertips have maximum leverage on wrist roll (since they're furthest from wrist).

    Returns (n_ring + 5*n_per_knuckle + 5*n_per_tip,) int64 tensor of MANO vertex indices.
    Default: 15 + 5 + 15 = 35 anchors.
    """
    ring = compute_wrist_ring_ids(mano_J, n_ring)
    knuckles = compute_knuckle_ids(mano_J, n_per_knuckle=n_per_knuckle)
    tips = compute_fingertip_ids(mano_J, n_per_tip=n_per_tip)
    return torch.cat([ring, knuckles, tips], dim=0)


def load_mano_smplx_mapping(repo_dir: Path, smplx_dir: str | None = None) -> dict[str, np.ndarray]:
    """Load MANO→SMPL-X vertex ID mapping."""
    candidates = [
        repo_dir / "models" / "smplx" / "MANO_SMPLX_vertex_ids.pkl",
        repo_dir.parent / "models" / "smplx" / "MANO_SMPLX_vertex_ids.pkl",
    ]
    if smplx_dir:
        candidates.insert(0, Path(smplx_dir) / "smplx" / "MANO_SMPLX_vertex_ids.pkl")
        candidates.insert(0, Path(smplx_dir) / "MANO_SMPLX_vertex_ids.pkl")
    pkl_path = next((p for p in candidates if p.exists()), None)
    if pkl_path is None:
        raise FileNotFoundError(f"MANO_SMPLX_vertex_ids.pkl not found in: {[str(c) for c in candidates]}")
    with open(pkl_path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def get_smplx_ring_ids(mano_ring_ids: torch.Tensor, mano_smplx_map: dict[str, np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
    """Map MANO wrist ring indices to SMPL-X vertex indices for left and right.

    Returns (left_ring_ids, right_ring_ids), each (n_ring,) int64.
    """
    idx = mano_ring_ids.numpy()
    left = torch.from_numpy(np.array(mano_smplx_map["left_hand"])[idx]).long()
    right = torch.from_numpy(np.array(mano_smplx_map["right_hand"])[idx]).long()
    return left, right


def build_ring_targets(hands_incam_dc: dict[int, list[dict]], mano_ring_ids: torch.Tensor,
                       num_frames: int, n_ring: int = 15) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract wrist ring vertex positions from DC MANO meshes.

    Returns:
        ring_target_l: (L, n_ring, 3) target positions
        ring_target_r: (L, n_ring, 3) target positions
        ring_w_l: (L, 1) per-frame weight (0 or 1)
        ring_w_r: (L, 1) per-frame weight (0 or 1)
    """
    ring_target_l = torch.zeros(num_frames, n_ring, 3)
    ring_target_r = torch.zeros(num_frames, n_ring, 3)
    ring_w_l = torch.zeros(num_frames, 1)
    ring_w_r = torch.zeros(num_frames, 1)

    for fi, dets in hands_incam_dc.items():
        if fi >= num_frames:
            continue
        for det in dets:
            ring_verts = det["verts"][mano_ring_ids]  # (n_ring, 3)
            if det["is_right"]:
                ring_target_r[fi] = ring_verts
                ring_w_r[fi] = 1.0
            else:
                ring_target_l[fi] = ring_verts
                ring_w_l[fi] = 1.0

    return ring_target_l, ring_target_r, ring_w_l, ring_w_r


# ---------------------------------------------------------------------------
# Body part vertex IDs (shared by collision + target validation)
# ---------------------------------------------------------------------------

def compute_body_part_vert_ids(model: torch.nn.Module, device: str = "cuda") -> dict[str, torch.Tensor]:
    """Compute vertex IDs for arm/hand and torso from rest-pose skinning weights.

    Returns dict with:
        coll_src: subsampled arm+hand+elbow vert IDs (for collision source)
        coll_tgt: subsampled torso vert IDs (for collision target)
        torso_full: un-subsampled torso vert IDs (for target validation)
    """
    with torch.no_grad():
        old_bs = model.batch_size
        model.batch_size = 1
        rest_out = model()
        model.batch_size = old_bs
        v0, j0 = rest_out.vertices[0], rest_out.joints[0]
        dists_to_j = torch.cdist(v0.unsqueeze(0), j0[:55].unsqueeze(0)).squeeze(0)
        closest_j = dists_to_j.argmin(dim=1)

        arm_hand_joints = {18, 19, 20, 21} | set(range(25, 55))
        torso_joints = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15}

        arm_ids = [i for i in range(len(v0)) if closest_j[i].item() in arm_hand_joints]
        torso_ids = [i for i in range(len(v0)) if closest_j[i].item() in torso_joints]

    return {
        "coll_src": torch.tensor(arm_ids[::3], device=device),
        "coll_tgt": torch.tensor(torso_ids[::5], device=device),
        "torso_full": torch.tensor(torso_ids, device=device),
    }


# ---------------------------------------------------------------------------
# Torso vertex normals (for anti-penetration + target validation)
# ---------------------------------------------------------------------------

def compute_vertex_normals(verts: torch.Tensor, faces: torch.Tensor,
                           vert_ids: torch.Tensor,
                           subset_faces: torch.Tensor | None = None) -> torch.Tensor:
    """Compute outward-facing vertex normals for a subset of vertices.

    Memory-efficient: only processes faces that touch vert_ids.

    Args:
        verts: (bs, V, 3) or (V, 3) mesh vertices
        faces: (F, 3) int64 face indices
        vert_ids: (N,) which vertices to return normals for
        subset_faces: optional precomputed subset of faces that touch vert_ids.
                      If None, computed on the fly.

    Returns:
        (bs, N, 3) or (N, 3) unit normals
    """
    squeeze = False
    if verts.dim() == 2:
        verts = verts.unsqueeze(0)
        squeeze = True

    bs, V, _ = verts.shape

    # Restrict to faces that touch vert_ids (much fewer than all faces)
    if subset_faces is None:
        vert_mask = torch.zeros(V, dtype=torch.bool, device=faces.device)
        vert_mask[vert_ids] = True
        face_touches = vert_mask[faces].any(dim=-1)
        subset_faces = faces[face_touches]

    F_sub = subset_faces.shape[0]

    v0 = verts[:, subset_faces[:, 0]]
    v1 = verts[:, subset_faces[:, 1]]
    v2 = verts[:, subset_faces[:, 2]]
    fn = torch.cross(v1 - v0, v2 - v0, dim=-1)  # (bs, F_sub, 3)

    vn = torch.zeros(bs, V, 3, device=verts.device, dtype=verts.dtype)
    for k in range(3):
        idx = subset_faces[:, k].unsqueeze(0).expand(bs, -1).unsqueeze(-1).expand(-1, -1, 3)
        vn.scatter_add_(1, idx, fn)

    vn_sub = vn[:, vert_ids]
    norms = vn_sub.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    vn_sub = vn_sub / norms

    if squeeze:
        vn_sub = vn_sub.squeeze(0)
    return vn_sub


def get_subset_faces(faces: torch.Tensor, vert_ids: torch.Tensor, V: int) -> torch.Tensor:
    """Return the subset of faces that touch any vertex in vert_ids."""
    vert_mask = torch.zeros(V, dtype=torch.bool, device=faces.device)
    vert_mask[vert_ids] = True
    face_touches = vert_mask[faces].any(dim=-1)
    return faces[face_touches]


# ---------------------------------------------------------------------------
# Pre-IK wrist target validation
# ---------------------------------------------------------------------------

def validate_wrist_targets(hands_incam_dc: dict[int, list[dict]],
                           smplx_verts_incam: torch.Tensor,
                           faces: torch.Tensor,
                           torso_vert_ids: torch.Tensor,
                           mano_J: torch.Tensor,
                           offset: float = 0.02) -> int:
    """Push wrist targets that are inside the torso to the surface + offset.

    Modifies hands_incam_dc in-place (shifts all 778 verts by correction vector).
    Returns number of corrected detections.
    """
    n_corrected = 0

    for fi, dets in hands_incam_dc.items():
        if fi >= smplx_verts_incam.shape[0]:
            continue

        torso_v = smplx_verts_incam[fi, torso_vert_ids]  # (N_torso, 3)
        torso_normals = compute_vertex_normals(
            smplx_verts_incam[fi], faces, torso_vert_ids
        )  # (N_torso, 3)

        for det in dets:
            wrist_pos = (mano_J[0] @ det["verts"]).detach()  # (3,)

            # Find nearest torso vertex
            dists = (torso_v - wrist_pos.unsqueeze(0)).norm(dim=-1)  # (N_torso,)
            nn_idx = dists.argmin()
            nn_pos = torso_v[nn_idx]
            nn_normal = torso_normals[nn_idx]

            # Inside test: dot(wrist - nearest, normal) < 0
            dot = ((wrist_pos - nn_pos) * nn_normal).sum()
            if dot < offset:  # inside or within margin
                # Push wrist to surface + offset along normal
                target_pos = nn_pos + offset * nn_normal
                correction = target_pos - wrist_pos
                det["verts"] = det["verts"] + correction.unsqueeze(0)
                n_corrected += 1

    return n_corrected


# ---------------------------------------------------------------------------
# IK solver
# ---------------------------------------------------------------------------

def solve_ik(model: torch.nn.Module, smplx_params: dict[str, torch.Tensor],
             ring_target_l: torch.Tensor, ring_target_r: torch.Tensor,
             ring_w_l: torch.Tensor, ring_w_r: torch.Tensor,
             smplx_ring_left: torch.Tensor, smplx_ring_right: torch.Tensor,
             faces: torch.Tensor,
             coll_src: torch.Tensor | None = None,
             coll_tgt: torch.Tensor | None = None,
             num_iters: int = 300, lr: float = 0.02,
             collision_weight: float = 1.0,
             penetration_tol: float = 0.005,
             ring_weight: float = 20.0,
             temporal_weight: float = 0.1,
             ik_chunk: int = 2048) -> tuple[dict[str, torch.Tensor], dict[str, np.ndarray] | None]:
    """Optimize arm rotations for wrist ring matching + anti-penetration.

    Wrist ring loss constrains both position and orientation.
    Normal-based anti-penetration prevents arm/hand mesh from going through torso.
    Hand pose is NOT modified.
    """
    device = "cuda"
    L = smplx_params["body_pose"].shape[0]
    IK_CHUNK = ik_chunk

    # Move ring targets to device
    ring_target_l = ring_target_l.to(device)
    ring_target_r = ring_target_r.to(device)
    ring_w_l = ring_w_l.to(device)
    ring_w_r = ring_w_r.to(device)
    smplx_ring_left = smplx_ring_left.to(device)
    smplx_ring_right = smplx_ring_right.to(device)
    faces_dev = faces.to(device)

    n_targets = int(ring_w_l.sum() + ring_w_r.sum())
    if n_targets == 0:
        print("  [IK] No ring targets — skipping")
        return smplx_params, None

    # Only optimize frames with targets
    has_target = ((ring_w_l.squeeze(-1) > 0) | (ring_w_r.squeeze(-1) > 0)).cpu()
    target_frames = torch.where(has_target)[0].tolist()
    idx = torch.tensor(target_frames, dtype=torch.long)
    L_ik = len(idx)
    n_ring = ring_target_l.shape[1]

    print(f"  [IK] {n_targets} ring targets ({n_ring} verts/ring), {L_ik}/{L} frames")

    # Subset to target frames
    frozen = {}
    for k, v in smplx_params.items():
        if k != "body_pose":
            frozen[k] = v[idx].to(device).detach()

    body_orig_full = smplx_params["body_pose"].reshape(L, 21, 3).to(device).detach()
    body_orig = body_orig_full[idx]

    arm_params = body_orig[:, 15:21].clone().requires_grad_(True)

    rt_l, rt_r = ring_target_l[idx], ring_target_r[idx]
    rw_l, rw_r = ring_w_l[idx], ring_w_r[idx]

    if collision_weight > 0 and coll_src is not None:
        from pytorch3d.ops import knn_points
        # Faces touching the torso subset — needed for per-frame outward normals,
        # which turn the unsigned KNN distance into a signed one.
        faces_dev = faces.to(device)
        coll_tgt = coll_tgt.to(device)
        coll_tgt_faces = get_subset_faces(faces_dev, coll_tgt, model.v_template.shape[0])
        print(f"  [IK] Anti-penetration: {len(coll_src)} arm+hand vs {len(coll_tgt)} torso verts "
              f"(signed, tol={penetration_tol * 1000:.0f}mm)")
    else:
        knn_points = None
        coll_tgt_faces = None

    optimizer = torch.optim.Adam([arm_params], lr=lr)

    for step in range(num_iters):
        optimizer.zero_grad()

        total_ring = torch.tensor(0.0, device=device)
        total_coll = torch.tensor(0.0, device=device)

        for ci in range(0, L_ik, IK_CHUNK):
            ce = min(ci + IK_CHUNK, L_ik)
            bs = ce - ci
            model.batch_size = bs

            bp_chunk = torch.cat([body_orig[ci:ce, :15], arm_params[ci:ce]], dim=1)
            chunk_params = {k: v[ci:ce] for k, v in frozen.items()}
            chunk_params["body_pose"] = bp_chunk.reshape(bs, -1)

            out = model(**chunk_params)
            verts = out.vertices

            # Wrist ring loss: match ring of vertices for position + orientation
            pred_ring_l = verts[:, smplx_ring_left]   # (bs, n_ring, 3)
            pred_ring_r = verts[:, smplx_ring_right]   # (bs, n_ring, 3)

            lr_loss = (rw_l[ci:ce].unsqueeze(-1) * (pred_ring_l - rt_l[ci:ce]).pow(2)).sum(dim=-1).mean(dim=-1).sum()
            rr_loss = (rw_r[ci:ce].unsqueeze(-1) * (pred_ring_r - rt_r[ci:ce]).pow(2)).sum(dim=-1).mean(dim=-1).sum()
            ring_loss = ring_weight * (lr_loss + rr_loss)

            # Signed anti-penetration. Contact is legitimate (hands rest on the body),
            # so only penalize arm/hand verts that are actually INSIDE the torso, and
            # allow `penetration_tol` of slack for soft tissue our rigid mesh can't model.
            # An unsigned distance cannot express this: it penalizes resting contact, and
            # a deeply buried vertex has a LARGE nearest-surface distance so it escapes
            # entirely. The sign comes from the torso outward normal at the nearest vertex.
            coll_loss = torch.tensor(0.0, device=device)
            if collision_weight > 0 and coll_src is not None and knn_points is not None:
                src_v = verts[:, coll_src]
                tgt_v = verts[:, coll_tgt].detach()
                tgt_n = compute_vertex_normals(
                    verts.detach(), faces_dev, coll_tgt, coll_tgt_faces
                )

                _, nn_idx, _ = knn_points(src_v, tgt_v, K=1)
                idx3 = nn_idx.expand(-1, -1, 3)
                nn_pts = torch.gather(tgt_v, 1, idx3)
                nn_nrm = torch.gather(tgt_n, 1, idx3)

                signed = ((src_v - nn_pts) * nn_nrm).sum(dim=-1)  # >0 outside, <0 inside
                penetration = torch.relu(-signed - penetration_tol)
                coll_loss = collision_weight * penetration.pow(2).sum()

            (ring_loss + coll_loss).backward()
            total_ring = total_ring + ring_loss.detach()
            total_coll = total_coll + coll_loss.detach()

        # Regularization — tight on shoulder/collar (keep posture stable),
        # loose on elbow/wrist (let them rotate freely to match HaMeR wrist)
        reg = 0.001 * (arm_params[:, :4] - body_orig[:, 15:19]).pow(2).sum() + \
              0.00002 * (arm_params[:, 4:] - body_orig[:, 19:21]).pow(2).sum()

        # Temporal smoothness (lowered; HaMeR targets may have high-freq motion)
        if L_ik > 2:
            temporal = temporal_weight * (arm_params[2:] - 2 * arm_params[1:-1] + arm_params[:-2]).pow(2).sum()
        else:
            temporal = torch.tensor(0.0, device=device)

        (reg + temporal).backward()
        optimizer.step()

        if step % 50 == 0 or step == num_iters - 1:
            print(f"    Step {step}: ring={total_ring.item():.6f} "
                  f"coll={total_coll.item():.4f} reg={reg.item():.4f} temporal={temporal.item():.4f}")

    # Write back optimized arm joints
    arm_full = body_orig_full[:, 15:21].clone()
    arm_full[idx] = arm_params.detach()

    arm_smoothed = torch.from_numpy(one_euro_filter_np(arm_full.cpu().numpy(), min_cutoff=1.0, beta=0.5)).float()
    print(f"  [IK] Applied One-Euro filter on arms (1.0/0.5)")

    result = {k: v.cpu() for k, v in smplx_params.items()}
    bp_final = torch.cat([body_orig_full[:, :15].cpu(), arm_smoothed], dim=1)
    result["body_pose"] = bp_final.reshape(L, -1)

    # Per-frame ring loss
    print("  [IK] Computing per-frame ring loss...")
    ik_ring_parts = []
    with torch.no_grad():
        for ci in range(0, L_ik, IK_CHUNK):
            ce = min(ci + IK_CHUNK, L_ik)
            model.batch_size = ce - ci
            cp = {k: v[ci:ce].to(device) for k, v in frozen.items()}
            cp["body_pose"] = result["body_pose"][idx[ci:ce]].to(device)

            v = model(**cp).vertices
            rl = (rw_l[ci:ce].unsqueeze(-1) * (v[:, smplx_ring_left] - rt_l[ci:ce]).pow(2)).sum(dim=-1).mean(dim=-1)
            rr = (rw_r[ci:ce].unsqueeze(-1) * (v[:, smplx_ring_right] - rt_r[ci:ce]).pow(2)).sum(dim=-1).mean(dim=-1)
            ik_ring_parts.append((rl + rr).cpu().numpy())

    ik_wrist_loss = np.zeros(L, dtype=np.float32)
    ik_wrist_loss[idx.numpy()] = np.concatenate(ik_ring_parts)

    per_frame_loss = {"ik_wrist_loss": ik_wrist_loss}

    valid = ik_wrist_loss[ik_wrist_loss > 0]
    if len(valid) > 0:
        print(f"  [IK] Ring loss: median={np.median(valid):.6f}, "
              f"max={valid.max():.6f}, p95={np.percentile(valid, 95):.6f}")

    return result, per_frame_loss


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smplx_params", required=True)
    parser.add_argument("--hamer_params", required=True)
    parser.add_argument("--gvhmr_result", required=True)
    parser.add_argument("--smplx_dir", required=True)
    parser.add_argument("--num_iters", type=int, default=150,
                        help="Adam steps. 150 measured identical to 300 (ring loss median "
                             "0.001529 vs 0.001512, p95 slightly better) at half the time.")
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--collision_weight", type=float, default=1.0,
                        help="Anti-penetration weight (0=off, 1.0=default). Requires pytorch3d.")
    parser.add_argument("--penetration_tol", type=float, default=0.005,
                        help="Allowed penetration in metres before penalizing (default 5mm). "
                             "Real bodies compress; a rigid mesh cannot, so contact and slight "
                             "sinking are natural. Raise if poses look repelled, lower if arms "
                             "still sink into the torso.")
    parser.add_argument("--ik_chunk", type=int, default=2048,
                        help="IK chunk size. Reduce to lower GPU memory (e.g. 512 for 8GB GPU).")
    parser.add_argument("--ring_weight", type=float, default=20.0,
                        help="Weight on wrist ring loss (higher = tighter MANO overfit).")
    parser.add_argument("--temporal_weight", type=float, default=0.1,
                        help="Temporal smoothness weight (lower = let wrist follow fast motion).")
    args = parser.parse_args()

    npz_path = Path(args.smplx_params)
    repo_dir = Path(__file__).resolve().parent.parent
    print(f"\n==== IK Hands v3 ====")
    print(f"SMPL-X params: {npz_path}")
    print(f"HaMeR params:  {args.hamer_params}")
    print(f"GVHMR result:  {args.gvhmr_result}")
    print(f"Iterations:    {args.num_iters}")
    print(f"Collision:     {args.collision_weight}")

    data = dict(np.load(npz_path, allow_pickle=True))
    num_frames = int(data.get("num_frames", len(data["body_pose"])))
    coord = str(data.get("coord_system", "global"))

    # Build incam params
    incam_params = {}
    for key in ["body_pose", "global_orient", "betas", "transl"]:
        incam_key = key + "_incam"
        if coord == "global" and incam_key in data:
            incam_params[key] = torch.from_numpy(data[incam_key][:num_frames]).float()
        elif key in data:
            incam_params[key] = torch.from_numpy(data[key][:num_frames]).float()

    for key in ["left_hand_pose", "right_hand_pose"]:
        if key in data:
            incam_params[key] = torch.from_numpy(data[key][:num_frames]).float()

    for key in ["jaw_pose", "expression", "leye_pose", "reye_pose"]:
        if key in data and np.any(data[key] != 0):
            incam_params[key] = torch.from_numpy(data[key][:num_frames]).float()

    gvhmr_data = torch.load(args.gvhmr_result, map_location="cpu", weights_only=False)
    K = gvhmr_data["K_fullimg"][0]
    if not isinstance(K, torch.Tensor):
        K = torch.tensor(K).float()
    else:
        K = K.clone().float()

    width = int(K[0, 2].item() * 2)
    height = int(K[1, 2].item() * 2)
    print(f"  Camera: fx={K[0,0]:.1f}, width~={width}, height~={height}")

    body_valid = data.get("body_valid", np.ones(num_frames, dtype=bool))
    n_invalid = (~body_valid).sum()
    if n_invalid > 0:
        print(f"  [Depth] {n_invalid}/{num_frames} frames marked invalid by merge")

    print("  Loading SMPL-X model...")
    smplx_dir = str(Path(args.smplx_dir).resolve())
    if (Path(smplx_dir) / "SMPLX_NEUTRAL.npz").exists():
        smplx_dir = str(Path(smplx_dir).parent)

    FK_BATCH = min(args.ik_chunk, num_frames)
    model = smplx.create(
        smplx_dir, model_type="smplx", gender="neutral",
        use_face_contour=False, num_betas=10, num_expression_coeffs=100,
        use_pca=False, flat_hand_mean=True, batch_size=FK_BATCH,
    ).cuda()

    faces = torch.from_numpy(model.faces.astype(np.int64))

    # Forward passes
    print(f"  SMPL-X forward (incam, {num_frames} frames)...")
    smplx_verts_incam, smplx_joints_incam = smplx_forward_chunked(model, incam_params, chunk_size=FK_BATCH)

    if coord == "global":
        global_params = {}
        for key in ["body_pose", "global_orient", "betas", "transl"]:
            if key in data:
                global_params[key] = torch.from_numpy(data[key][:num_frames]).float()
        for key in ["left_hand_pose", "right_hand_pose", "jaw_pose", "expression", "leye_pose", "reye_pose"]:
            if key in data and np.any(data[key] != 0):
                global_params[key] = torch.from_numpy(data[key][:num_frames]).float()
        print("  SMPL-X forward (global)...")
        smplx_verts_global, _ = smplx_forward_chunked(model, global_params, chunk_size=FK_BATCH)
    else:
        smplx_verts_global = smplx_verts_incam

    # Load HaMeR hands
    print("  Loading HaMeR hand meshes...")
    hand_meshes = load_hand_meshes(args.hamer_params, num_frames)
    print(f"  HaMeR: {len(hand_meshes)} frames with detections")
    if len(hand_meshes) == 0:
        print("  [IK] No HaMeR detections — nothing to do")
        return

    mano_J = load_mano_joint_regressor()

    # Depth-correct hands
    dc_wrist_targets, hands_incam_dc = place_hands_dc(
        hand_meshes, K, smplx_joints_incam,
        smplx_verts_incam, smplx_verts_global,
        width, height, num_frames, mano_J=mano_J,
    )

    # Remove depth-invalid frames
    n_before = len(dc_wrist_targets)
    dc_wrist_targets = {fi: t for fi, t in dc_wrist_targets.items() if body_valid[fi]}
    hands_incam_dc = {fi: t for fi, t in hands_incam_dc.items() if body_valid[fi]}
    n_removed = n_before - len(dc_wrist_targets)
    if n_removed > 0:
        print(f"  [Depth] Removed {n_removed} targets on invalid frames")

    # IK coverage check
    n_hand_frames = len(hand_meshes)
    n_targets = len(dc_wrist_targets)
    ik_coverage = n_targets / max(n_hand_frames, 1)
    MIN_IK_COVERAGE = 0.20
    if ik_coverage < MIN_IK_COVERAGE:
        msg = (f"  [IK FAIL] DC coverage too low: {n_targets}/{n_hand_frames} "
               f"({ik_coverage:.1%}) < {MIN_IK_COVERAGE:.0%} threshold")
        print(msg, file=sys.stderr)
        print(msg)
        sys.exit(77)

    # Phase B: Validate wrist targets — push inside-torso targets to surface
    print("  [IK] Computing body part vertex IDs...")
    body_parts = compute_body_part_vert_ids(model)

    n_corrected = validate_wrist_targets(
        hands_incam_dc, smplx_verts_incam, faces,
        body_parts["torso_full"].cpu(), mano_J, offset=0.02,
    )
    if n_corrected > 0:
        print(f"  [IK] Pushed {n_corrected} wrist targets out of torso (+2cm offset)")

    # Phase A: Build wrist ring + knuckle anchor targets
    print("  [IK] Setting up wrist ring + knuckle matching...")
    mano_ring_ids = compute_anchor_ids(mano_J, n_ring=15)  # 15 ring + 5 knuckles = 20 anchors
    mano_smplx_map = load_mano_smplx_mapping(repo_dir, smplx_dir=args.smplx_dir)
    smplx_ring_left, smplx_ring_right = get_smplx_ring_ids(mano_ring_ids, mano_smplx_map)

    ring_target_l, ring_target_r, ring_w_l, ring_w_r = build_ring_targets(
        hands_incam_dc, mano_ring_ids, num_frames, n_ring=len(mano_ring_ids),
    )
    print(f"  [IK] Anchor targets ({len(mano_ring_ids)} verts/hand): L={int(ring_w_l.sum())}, R={int(ring_w_r.sum())} frames")

    # Phase A+C: Run IK
    ik_result, per_frame_loss = solve_ik(
        model, incam_params,
        ring_target_l, ring_target_r, ring_w_l, ring_w_r,
        smplx_ring_left, smplx_ring_right,
        faces,
        coll_src=body_parts["coll_src"],
        coll_tgt=body_parts["coll_tgt"],
        num_iters=args.num_iters, lr=args.lr,
        collision_weight=args.collision_weight,
        penetration_tol=args.penetration_tol,
        ring_weight=args.ring_weight,
        temporal_weight=args.temporal_weight,
        ik_chunk=args.ik_chunk,
    )

    if per_frame_loss is None:
        print("  [IK] No optimization performed")
        return

    # Save
    save_data = dict(data)
    ik_body_pose = ik_result["body_pose"].numpy()
    save_data["body_pose"] = ik_body_pose
    if "body_pose_incam" in data:
        save_data["body_pose_incam"] = ik_body_pose

    save_data["ik_wrist_loss"] = per_frame_loss["ik_wrist_loss"]
    save_data["ik_coverage"] = np.float32(ik_coverage)
    save_data["ik_n_targets"] = np.int64(n_targets)

    np.savez(npz_path, **save_data)
    print(f"  [IK] Saved → {npz_path}")
    print("  [IK] Done!")


if __name__ == "__main__":
    main()
