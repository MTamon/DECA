#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extract DECA expression and head motion (position/angle) parameters from a video.

Outputs a compressed NPZ containing per-frame arrays:
- frame_idx: [N] indices processed
- exp: [N, n_exp] expression coefficients
- pose_axis_angle: [N, 6] (global head axis-angle[3] + jaw axis-angle[3])
- head_euler_deg: [N, 3] Euler angles [pitch(x), yaw(y), roll(z)] in degrees
- cam: [N, 3] weak-perspective camera [s, tx, ty]

Cropping uses the same FAN-based detector as other demos. GPU/CPU controlled by --device.
Rendering is not used; only encode is invoked for speed.
"""

import os
import sys
import argparse
import cv2
import numpy as np
import torch

# make repo importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decalib.deca import DECA
from decalib.utils.config import cfg as deca_cfg
from decalib.datasets.detectors import FAN
from decalib.utils.rotation_converter import batch_axis2matrix


def bbox2point(left, right, top, bottom, type='kpt68'):
    if type == 'kpt68':
        old_size = (right - left + bottom - top) / 2 * 1.1
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
    elif type == 'bbox':
        old_size = (right - left + bottom - top) / 2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 + old_size * 0.12])
    else:
        raise NotImplementedError
    return old_size, center


def crop_face_rgb(image_rgb, detector, crop_size=224, scale=1.25):
    h, w, _ = image_rgb.shape
    bbox, bbox_type = detector.run(image_rgb)
    if len(bbox) < 4:
        left, right, top, bottom = 0, w - 1, 0, h - 1
        bbox_type = 'bbox'
    else:
        left, top, right, bottom = bbox[0], bbox[1], bbox[2], bbox[3]
    old_size, center = bbox2point(left, right, top, bottom, type=bbox_type)
    size = int(old_size * scale)

    src = np.float32([
        [center[0] - size / 2, center[1] - size / 2],
        [center[0] - size / 2, center[1] + size / 2],
        [center[0] + size / 2, center[1] - size / 2],
    ])
    dst = np.float32([
        [0, 0],
        [0, crop_size - 1],
        [crop_size - 1, 0],
    ])
    M = cv2.getAffineTransform(src, dst)
    cropped = cv2.warpAffine(image_rgb, M, (crop_size, crop_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return cropped


def default_output_path(src_path: str) -> str:
    root, _ = os.path.splitext(src_path)
    return f"{root}_deca_params.npz"


def rotmat_to_euler_xyz(R: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrices to XYZ intrinsic Euler angles [pitch(x), yaw(y), roll(z)].
    R: [N, 3, 3]
    Returns: [N, 3] angles in radians.
    """
    # robust to gimbal lock
    sy = torch.sqrt(R[:, 0, 0] * R[:, 0, 0] + R[:, 1, 0] * R[:, 1, 0])
    singular = sy < 1e-6

    x = torch.where(
        ~singular,
        torch.atan2(R[:, 2, 1], R[:, 2, 2]),
        torch.atan2(-R[:, 1, 2], R[:, 1, 1]),
    )
    y = torch.where(
        ~singular,
        torch.atan2(-R[:, 2, 0], sy),
        torch.atan2(-R[:, 2, 0], sy),  # same formula in singular case
    )
    z = torch.where(
        ~singular,
        torch.atan2(R[:, 1, 0], R[:, 0, 0]),
        torch.zeros_like(x),
    )
    return torch.stack([x, y, z], dim=1)


def main():
    parser = argparse.ArgumentParser(description='Extract DECA params from a video into NPZ')
    parser.add_argument('-i', '--input', required=True, type=str, help='input video path')
    parser.add_argument('-o', '--output', default=None, type=str, help='output npz path (default: <input>_deca_params.npz)')
    parser.add_argument('--device', default='cuda', type=str, help='cuda or cpu')
    parser.add_argument('--rasterizer_type', default='standard', type=str, help='standard or pytorch3d')
    parser.add_argument('--crop_size', default=224, type=int)
    parser.add_argument('--scale', default=1.25, type=float)
    parser.add_argument('--sample_step', default=1, type=int, help='process every Nth frame')
    parser.add_argument('--limit', default=0, type=int, help='optional max number of frames to process')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print('Input not found:', args.input)
        sys.exit(1)

    out_path = args.output or default_output_path(args.input)
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)

    # Configure DECA
    deca_cfg.rasterizer_type = args.rasterizer_type
    device = args.device
    deca = DECA(config=deca_cfg, device=device)
    detector = FAN()

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print('Failed to open video:', args.input)
        sys.exit(1)

    frame_idx = 0
    stored = 0
    idx_list = []
    exp_list = []
    pose_axis_list = []
    head_euler_list = []
    cam_list = []

    with torch.no_grad():
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if frame_idx % max(1, args.sample_step) != 0:
                frame_idx += 1
                continue

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            crop_rgb = crop_face_rgb(frame_rgb, detector, crop_size=args.crop_size, scale=args.scale)
            inp = torch.from_numpy(crop_rgb.astype(np.float32) / 255.0).permute(2, 0, 1)[None, ...].to(device)

            # use_detail=False to skip detail encoder for speed
            codedict = deca.encode(inp, use_detail=False)

            # gather params
            exp = codedict['exp'].detach().cpu().numpy()[0]
            pose_axis = codedict['pose'].detach().cpu().numpy()[0]  # [6] global(3)+jaw(3)
            cam = codedict['cam'].detach().cpu().numpy()[0]         # [3] s, tx, ty

            # head euler from global axis-angle (first 3)
            head_axis = codedict['pose'][:, :3]
            R = batch_axis2matrix(head_axis)  # [1,3,3]
            euler_xyz = rotmat_to_euler_xyz(R).detach().cpu().numpy()[0]  # radians
            euler_deg = euler_xyz * (180.0 / np.pi)

            # store
            idx_list.append(frame_idx)
            exp_list.append(exp.astype(np.float32))
            pose_axis_list.append(pose_axis.astype(np.float32))
            head_euler_list.append(euler_deg.astype(np.float32))
            cam_list.append(cam.astype(np.float32))

            stored += 1
            frame_idx += 1
            if args.limit > 0 and stored >= args.limit:
                break

    cap.release()

    if stored == 0:
        print('No frames processed. Nothing to save.')
        sys.exit(2)

    # Stack and save
    idx_arr = np.asarray(idx_list, dtype=np.int32)
    exp_arr = np.stack(exp_list, axis=0)
    pose_axis_arr = np.stack(pose_axis_list, axis=0)
    head_euler_arr = np.stack(head_euler_list, axis=0)
    cam_arr = np.stack(cam_list, axis=0)

    np.savez_compressed(
        out_path,
        frame_idx=idx_arr,
        exp=exp_arr,
        pose_axis_angle=pose_axis_arr,
        head_euler_deg=head_euler_arr,
        cam=cam_arr,
    )
    print('Saved:', out_path)


if __name__ == '__main__':
    main()

