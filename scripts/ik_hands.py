#!/usr/bin/env python3
"""IK hands: optimize SMPL-X arm/hand pose to match HaMeR MANO targets.

Solves for shoulder/elbow/wrist rotations and hand pose to minimize
wrist position error + finger joint error against depth-corrected MANO targets.

Memory-safe: chunks FK passes and IK gradient accumulation to handle 10K+ frame videos.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import smplx
import torch

from hand_utils import load_hand_meshes, place_hands_dc
from utils import load_mano_joint_regressor, one_euro_filter_np, smplx_forward_chunked


def solve_ik(model: torch.nn.Module, smplx_params: dict[str, torch.Tensor], dc_wrist_targets: dict[int, dict[str, torch.Tensor]], mano_finger_targets: dict[int, dict[str, torch.Tensor]],
             num_iters: int = 300, lr: float = 0.02, collision_weight: float = 0.0) -> tuple[dict[str, torch.Tensor], dict[str, np.ndarray] | None]:
    """Optimize arm rotations + hand pose so SMPL-X matches MANO targets.

    Runs only on frames with targets and chunks the forward pass to avoid OOM.
    """
    device = "cuda"
    L = smplx_params["body_pose"].shape[0]
    # ponytail: measured ~3 MB/frame of activations per chunk (docs/benchmarks.md); size chunks to free VRAM.
    free_mb = torch.cuda.mem_get_info()[0] // 2**20 if torch.cuda.is_available() else 0
    IK_CHUNK = int(min(2048, max(64, free_mb // 6))) if free_mb else 2048

    # Build target tensors (full L)
    target_wrist_l = torch.zeros(L, 3, device=device)
    target_wrist_r = torch.zeros(L, 3, device=device)
    w_wrist_l = torch.zeros(L, 1, device=device)
    w_wrist_r = torch.zeros(L, 1, device=device)

    for fi, t in dc_wrist_targets.items():
        if fi >= L:
            continue
        if "left" in t:
            target_wrist_l[fi] = t["left"].to(device)
            w_wrist_l[fi] = 1.0
        if "right" in t:
            target_wrist_r[fi] = t["right"].to(device)
            w_wrist_r[fi] = 1.0

    target_fingers_l = torch.zeros(L, 15, 3, device=device)
    target_fingers_r = torch.zeros(L, 15, 3, device=device)
    w_fingers_l = torch.zeros(L, 1, 1, device=device)
    w_fingers_r = torch.zeros(L, 1, 1, device=device)

    for fi, t in mano_finger_targets.items():
        if fi >= L:
            continue
        if "left" in t:
            target_fingers_l[fi] = t["left"].to(device)
            w_fingers_l[fi] = 1.0
        if "right" in t:
            target_fingers_r[fi] = t["right"].to(device)
            w_fingers_r[fi] = 1.0

    n_wrist = int(w_wrist_l.sum() + w_wrist_r.sum())
    n_finger = int(w_fingers_l.sum() + w_fingers_r.sum())
    if n_wrist == 0 and n_finger == 0:
        print("  [IK] No hand targets — skipping")
        return smplx_params, None

    print(f"  [IK] {n_wrist} wrist targets + {n_finger} finger-hand targets")

    # Only optimize frames with targets
    target_frames = sorted(set(dc_wrist_targets.keys()) | set(mano_finger_targets.keys()))
    target_frames = [f for f in target_frames if f < L]
    idx = torch.tensor(target_frames, dtype=torch.long)
    L_ik = len(idx)
    print(f"  [IK] Optimizing {L_ik}/{L} frames with targets (chunks of {min(IK_CHUNK, L_ik)})")

    # Subset to target frames
    frozen = {}
    for k, v in smplx_params.items():
        if k not in ("body_pose", "left_hand_pose", "right_hand_pose"):
            frozen[k] = v[idx].to(device).detach()

    body_orig_full = smplx_params["body_pose"].reshape(L, 21, 3).to(device).detach()
    lhand_orig_full = smplx_params["left_hand_pose"].to(device).detach()
    rhand_orig_full = smplx_params["right_hand_pose"].to(device).detach()

    body_orig = body_orig_full[idx]
    lhand_orig = lhand_orig_full[idx]
    rhand_orig = rhand_orig_full[idx]

    arm_params = body_orig[:, 15:21].clone().requires_grad_(True)
    lhand_params = lhand_orig.clone().requires_grad_(True)
    rhand_params = rhand_orig.clone().requires_grad_(True)

    tw_l, tw_r = target_wrist_l[idx], target_wrist_r[idx]
    ww_l, ww_r = w_wrist_l[idx], w_wrist_r[idx]
    tf_l, tf_r = target_fingers_l[idx], target_fingers_r[idx]
    wf_l, wf_r = w_fingers_l[idx], w_fingers_r[idx]

    # Precompute collision vertex IDs if collision penalty is enabled
    coll_src = coll_tgt = None
    if collision_weight > 0:
        from pytorch3d.ops import knn_points
        with torch.no_grad():
            old_bs = model.batch_size
            model.batch_size = 1
            rest_out = model()
            model.batch_size = old_bs
            v0, j0 = rest_out.vertices[0], rest_out.joints[0]
            dists_to_j = torch.cdist(v0.unsqueeze(0), j0[:55].unsqueeze(0)).squeeze(0)
            closest_j = dists_to_j.argmin(dim=1)
            forearm_hand_joints = {18, 19, 20, 21} | set(range(25, 55))
            torso_joints_set = {0, 3, 6, 9, 12, 13, 14}
            coll_src = torch.tensor([i for i in range(len(v0))
                                     if closest_j[i].item() in forearm_hand_joints][::5], device=device)
            coll_tgt = torch.tensor([i for i in range(len(v0))
                                     if closest_j[i].item() in torso_joints_set][::10], device=device)
        print(f"  [IK] Collision check: {len(coll_src)} forearm+hand vs {len(coll_tgt)} torso verts")

    optimizer = torch.optim.Adam([arm_params, lhand_params, rhand_params], lr=lr)

    for step in range(num_iters):
        optimizer.zero_grad()

        total_wrist = torch.tensor(0.0, device=device)
        total_fingers = torch.tensor(0.0, device=device)
        total_collision = torch.tensor(0.0, device=device)

        for ci in range(0, L_ik, IK_CHUNK):
            ce = min(ci + IK_CHUNK, L_ik)
            bs = ce - ci
            model.batch_size = bs

            bp_chunk = torch.cat([body_orig[ci:ce, :15], arm_params[ci:ce]], dim=1)
            chunk_params = {k: v[ci:ce] for k, v in frozen.items()}
            chunk_params["body_pose"] = bp_chunk.reshape(bs, -1)
            chunk_params["left_hand_pose"] = lhand_params[ci:ce]
            chunk_params["right_hand_pose"] = rhand_params[ci:ce]

            out = model(**chunk_params)
            joints = out.joints
            verts = out.vertices

            lw = (ww_l[ci:ce] * (joints[:, 20] - tw_l[ci:ce]).pow(2)).sum() + \
                 (ww_r[ci:ce] * (joints[:, 21] - tw_r[ci:ce]).pow(2)).sum()
            lf = (wf_l[ci:ce] * (joints[:, 25:40] - tf_l[ci:ce]).pow(2)).sum() + \
                 (wf_r[ci:ce] * (joints[:, 40:55] - tf_r[ci:ce]).pow(2)).sum()

            coll_loss = torch.tensor(0.0, device=device)
            if collision_weight > 0 and coll_src is not None:
                src_v = verts[:, coll_src]
                tgt_v = verts[:, coll_tgt].detach()
                _, nn_idx, _ = knn_points(src_v, tgt_v, K=1)
                nn_pts = torch.gather(tgt_v, 1, nn_idx.expand(-1, -1, 3))
                dist = (src_v - nn_pts).norm(dim=-1, keepdim=True).clamp(min=1e-6)
                torso_center = tgt_v.mean(dim=1, keepdim=True)
                inside = (src_v - torso_center).norm(dim=-1) < (nn_pts - torso_center).norm(dim=-1)
                penetration = torch.where(inside, dist.squeeze(-1), torch.zeros_like(inside.float()))
                penetration = torch.relu(penetration - 0.03)
                coll_loss = collision_weight * penetration.pow(2).sum()

            (lw + 5.0 * lf + coll_loss).backward()
            total_wrist = total_wrist + lw.detach()
            total_fingers = total_fingers + lf.detach()
            total_collision = total_collision + coll_loss.detach()

        reg = 0.001 * (arm_params[:, :4] - body_orig[:, 15:19]).pow(2).sum() + \
              0.0002 * (arm_params[:, 4:] - body_orig[:, 19:21]).pow(2).sum() + \
              0.00003 * (lhand_params - lhand_orig).pow(2).sum() + \
              0.00003 * (rhand_params - rhand_orig).pow(2).sum()

        if L_ik > 2:
            temporal = 0.5 * (arm_params[2:] - 2*arm_params[1:-1] + arm_params[:-2]).pow(2).sum() + \
                       0.1 * (lhand_params[2:] - 2*lhand_params[1:-1] + lhand_params[:-2]).pow(2).sum() + \
                       0.1 * (rhand_params[2:] - 2*rhand_params[1:-1] + rhand_params[:-2]).pow(2).sum()
        else:
            temporal = torch.tensor(0.0, device=device)

        (reg + temporal).backward()
        optimizer.step()

        if step % 50 == 0 or step == num_iters - 1:
            print(f"    Step {step}: wrist={total_wrist.item():.6f} fingers={total_fingers.item():.6f} "
                  f"coll={total_collision.item():.4f} reg={reg.item():.4f} temporal={temporal.item():.4f}")

    arm_full = body_orig_full[:, 15:21].clone()
    arm_full[idx] = arm_params.detach()
    lhand_full = lhand_orig_full.clone()
    lhand_full[idx] = lhand_params.detach()
    rhand_full = rhand_orig_full.clone()
    rhand_full[idx] = rhand_params.detach()

    # Smooth
    arm_smoothed = torch.from_numpy(one_euro_filter_np(arm_full.cpu().numpy(), min_cutoff=1.0, beta=0.5)).float()
    lhand_smoothed = torch.from_numpy(one_euro_filter_np(lhand_full.cpu().numpy(), min_cutoff=0.8, beta=0.3)).float()
    rhand_smoothed = torch.from_numpy(one_euro_filter_np(rhand_full.cpu().numpy(), min_cutoff=0.8, beta=0.3)).float()
    print(f"  [IK] Applied One-Euro filter (arms: 1.0/0.5, hands: 0.8/0.3)")

    result = {k: v.cpu() for k, v in smplx_params.items()}
    bp_final = torch.cat([body_orig_full[:, :15].cpu(), arm_smoothed], dim=1)
    result["body_pose"] = bp_final.reshape(L, -1)
    result["left_hand_pose"] = lhand_smoothed
    result["right_hand_pose"] = rhand_smoothed

    # Chunked per-frame loss
    print("  [IK] Computing per-frame loss...")
    ik_wrist_parts, ik_finger_parts = [], []
    with torch.no_grad():
        for ci in range(0, L_ik, IK_CHUNK):
            ce = min(ci + IK_CHUNK, L_ik)
            model.batch_size = ce - ci
            cp = {k: v[ci:ce].to(device) for k, v in frozen.items()}
            cp["body_pose"] = result["body_pose"][idx[ci:ce]].to(device)
            cp["left_hand_pose"] = result["left_hand_pose"][idx[ci:ce]].to(device)
            cp["right_hand_pose"] = result["right_hand_pose"][idx[ci:ce]].to(device)

            j = model(**cp).joints
            wl = (ww_l[ci:ce].squeeze(-1) * (j[:, 20] - tw_l[ci:ce]).pow(2).sum(dim=-1))
            wr = (ww_r[ci:ce].squeeze(-1) * (j[:, 21] - tw_r[ci:ce]).pow(2).sum(dim=-1))
            ik_wrist_parts.append((wl + wr).cpu().numpy())

            fl = (wf_l[ci:ce].squeeze(-1).squeeze(-1) * (j[:, 25:40] - tf_l[ci:ce]).pow(2).sum(dim=-1).sum(dim=-1))
            fr = (wf_r[ci:ce].squeeze(-1).squeeze(-1) * (j[:, 40:55] - tf_r[ci:ce]).pow(2).sum(dim=-1).sum(dim=-1))
            ik_finger_parts.append((fl + fr).cpu().numpy())

    ik_wrist_loss = np.zeros(L, dtype=np.float32)
    ik_finger_loss = np.zeros(L, dtype=np.float32)
    ik_wrist_loss[idx.numpy()] = np.concatenate(ik_wrist_parts)
    ik_finger_loss[idx.numpy()] = np.concatenate(ik_finger_parts)

    per_frame_loss = {
        "ik_wrist_loss": ik_wrist_loss,
        "ik_finger_loss": ik_finger_loss,
    }

    valid_wrist = ik_wrist_loss[ik_wrist_loss > 0]
    valid_finger = ik_finger_loss[ik_finger_loss > 0]
    if len(valid_wrist) > 0:
        print(f"  [IK] Wrist loss: median={np.median(valid_wrist):.6f}, "
              f"max={valid_wrist.max():.6f}, p95={np.percentile(valid_wrist, 95):.6f}")
    if len(valid_finger) > 0:
        print(f"  [IK] Finger loss: median={np.median(valid_finger):.6f}, "
              f"max={valid_finger.max():.6f}, p95={np.percentile(valid_finger, 95):.6f}")

    return result, per_frame_loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smplx_params", required=True)
    parser.add_argument("--hamer_params", required=True)
    parser.add_argument("--gvhmr_result", required=True)
    parser.add_argument("--smplx_dir", required=True)
    parser.add_argument("--num_iters", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--collision_weight", type=float, default=0.0,
                        help="Collision penalty weight (0=off, 1.0=moderate). Requires pytorch3d.")
    args = parser.parse_args()

    npz_path = Path(args.smplx_params)
    print(f"\n==== IK Hands ====")
    print(f"SMPL-X params: {npz_path}")
    print(f"HaMeR params:  {args.hamer_params}")
    print(f"GVHMR result:  {args.gvhmr_result}")
    print(f"Iterations:    {args.num_iters}")

    data = dict(np.load(npz_path, allow_pickle=True))
    num_frames = int(data.get("num_frames", len(data["body_pose"])))
    coord = str(data.get("coord_system", "global"))

    # Build incam params: prefer {param}_incam variants
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

    # Read body_valid from merge (depth validation done there)
    body_valid = data.get("body_valid", np.ones(num_frames, dtype=bool))
    n_invalid = (~body_valid).sum()
    if n_invalid > 0:
        print(f"  [Depth] {n_invalid}/{num_frames} frames marked invalid by merge")

    print("  Loading SMPL-X model...")
    smplx_dir = str(Path(args.smplx_dir).resolve())
    if (Path(smplx_dir) / "SMPLX_NEUTRAL.npz").exists():
        smplx_dir = str(Path(smplx_dir).parent)

    FK_BATCH = min(512, num_frames)
    model = smplx.create(
        smplx_dir, model_type="smplx", gender="neutral",
        use_face_contour=False, num_betas=10, num_expression_coeffs=100,
        use_pca=False, flat_hand_mean=True, batch_size=FK_BATCH,
    ).cuda()

    print(f"  Running SMPL-X forward pass (incam, {num_frames} frames, chunks of {FK_BATCH})...")
    smplx_verts_incam, smplx_joints_incam = smplx_forward_chunked(model, incam_params, chunk_size=FK_BATCH)

    if coord == "global":
        global_params = {}
        for key in ["body_pose", "global_orient", "betas", "transl"]:
            if key in data:
                global_params[key] = torch.from_numpy(data[key][:num_frames]).float()
        for key in ["left_hand_pose", "right_hand_pose", "jaw_pose", "expression", "leye_pose", "reye_pose"]:
            if key in data and np.any(data[key] != 0):
                global_params[key] = torch.from_numpy(data[key][:num_frames]).float()

        print("  Running SMPL-X forward pass (global)...")
        smplx_verts_global, _ = smplx_forward_chunked(model, global_params, chunk_size=FK_BATCH)
    else:
        smplx_verts_global = smplx_verts_incam

    print(f"  Loading HaMeR hand meshes...")
    hand_meshes = load_hand_meshes(args.hamer_params, num_frames)
    print(f"  HaMeR: {len(hand_meshes)} frames with detections")

    if len(hand_meshes) == 0:
        print("  [IK] No HaMeR detections — nothing to do")
        return

    mano_J = load_mano_joint_regressor()

    dc_wrist_targets, hands_incam_dc = place_hands_dc(
        hand_meshes, K, smplx_joints_incam,
        smplx_verts_incam, smplx_verts_global,
        width, height, num_frames,
        mano_J=mano_J,
    )

    # Remove IK targets for depth-invalid frames
    n_before = len(dc_wrist_targets)
    dc_wrist_targets = {fi: t for fi, t in dc_wrist_targets.items() if body_valid[fi]}
    hands_incam_dc = {fi: t for fi, t in hands_incam_dc.items() if body_valid[fi]}
    n_removed = n_before - len(dc_wrist_targets)
    if n_removed > 0:
        print(f"  [Depth] Removed {n_removed} IK targets on invalid frames")
    print(f"  DC wrist targets: {len(dc_wrist_targets)} frames")

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
    mano_finger_targets = {}
    for fi, dets in hands_incam_dc.items():
        mano_finger_targets[fi] = {}
        for d in dets:
            joints_mano = mano_J @ d["verts"]
            finger_joints = joints_mano[1:]
            key = "right" if d["is_right"] else "left"
            mano_finger_targets[fi][key] = finger_joints

    ik_result, per_frame_loss = solve_ik(
        model, incam_params, dc_wrist_targets, mano_finger_targets,
        num_iters=args.num_iters, lr=args.lr,
        collision_weight=args.collision_weight,
    )

    if per_frame_loss is None:
        print("  [IK] No optimization performed")
        return

    save_data = dict(data)
    ik_body_pose = ik_result["body_pose"].numpy()
    ik_lhand = ik_result["left_hand_pose"].numpy()
    ik_rhand = ik_result["right_hand_pose"].numpy()

    # IK modifies body_pose in incam space — save with matching convention
    if coord == "global" and "transl_incam" in data:
        save_data["body_pose_incam"] = ik_body_pose
    else:
        save_data["body_pose"] = ik_body_pose

    save_data["left_hand_pose"] = ik_lhand
    save_data["right_hand_pose"] = ik_rhand
    save_data["ik_wrist_loss"] = per_frame_loss["ik_wrist_loss"]
    save_data["ik_finger_loss"] = per_frame_loss["ik_finger_loss"]
    save_data["ik_coverage"] = np.float32(ik_coverage)
    save_data["ik_n_targets"] = np.int64(n_targets)

    np.savez(npz_path, **save_data)
    print(f"  [IK] Saved IK-corrected params → {npz_path}")
    print("  [IK] Done!")


if __name__ == "__main__":
    main()
