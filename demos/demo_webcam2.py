#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import argparse
import cv2
import numpy as np
import torch

# make repo importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decalib.deca import DECA
from decalib.utils.config import cfg as deca_cfg
from decalib.datasets.detectors import FAN


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
        # fallback to full image if no face detected
        left, right, top, bottom = 0, w - 1, 0, h - 1
        bbox_type = 'bbox'
    else:
        left, top, right, bottom = bbox[0], bbox[1], bbox[2], bbox[3]
    old_size, center = bbox2point(left, right, top, bottom, type=bbox_type)
    size = int(old_size * scale)

    # three-point similarity transform (OpenCV)
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


def is_camera_source(source: str) -> bool:
    return source.isdigit()


def default_output_path(src_path: str) -> str:
    root, ext = os.path.splitext(src_path)
    if not ext:
        ext = '.mp4'
    return f"{root}_deca_vis.mp4"


def main():
    parser = argparse.ArgumentParser(description='DECA webcam/video demo with video output & FPS')
    parser.add_argument('--source', default='0', type=str, help='camera index or video file path (default: 0)')
    parser.add_argument('--output', default=None, type=str, help='output video path when source is a file')
    parser.add_argument('--device', default='cuda', type=str, help='cuda or cpu')
    parser.add_argument('--rasterizer_type', default='standard', type=str, help='standard or pytorch3d')
    parser.add_argument('--useTex', default='False', type=str, help='use FLAME texture model')
    parser.add_argument('--extractTex', default='True', type=str, help='extract texture from input')
    parser.add_argument('--crop_size', default=224, type=int)
    parser.add_argument('--scale', default=1.25, type=float)
    args = parser.parse_args()

    # parse booleans from str
    str2bool = lambda x: x.lower() in ['true', '1', 'yes', 'y']
    useTex = str2bool(args.useTex)
    extractTex = str2bool(args.extractTex)

    # open source
    cam_mode = is_camera_source(args.source)
    cap = cv2.VideoCapture(int(args.source) if cam_mode else args.source)
    if not cap.isOpened():
        print('Failed to open source:', args.source)
        return

    # configure DECA
    deca_cfg.model.use_tex = useTex
    deca_cfg.model.extract_tex = extractTex
    deca_cfg.rasterizer_type = args.rasterizer_type
    device = args.device

    deca = DECA(config=deca_cfg, device=device)
    detector = FAN()

    # Prepare writer if processing a video file
    writer = None
    output_path = None
    if not cam_mode:
        output_path = args.output or default_output_path(args.source)
        # We will initialize writer after first visualization frame to know its size
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    # UI for webcam display
    win_name = 'DECA Live'
    if cam_mode:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    # FPS tracking (detection only, exclude writer.write)
    total_detect_time = 0.0
    total_frames = 0

    # target FPS from source for video writing
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0
    if src_fps <= 0 or np.isnan(src_fps):
        src_fps = 25.0

    with torch.no_grad():
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # detection timing starts (exclude I/O & writing)
            t0 = time.time()

            crop_rgb = crop_face_rgb(frame_rgb, detector, crop_size=args.crop_size, scale=args.scale)
            inp = torch.from_numpy(crop_rgb.astype(np.float32) / 255.0).permute(2, 0, 1)[None, ...].to(device)

            codedict = deca.encode(inp)
            _, visdict = deca.decode(codedict, render_orig=False)
            vis = deca.visualize(visdict, size=args.crop_size)
            # ensure array is C-contiguous for OpenCV ops
            if not vis.flags['C_CONTIGUOUS']:
                vis = np.ascontiguousarray(vis)

            # detection timing ends
            t1 = time.time()
            dt = t1 - t0
            total_detect_time += max(dt, 1e-8)
            total_frames += 1

            # draw instantaneous detection FPS on the visualization
            inst_fps = 1.0 / max(dt, 1e-8)
            cv2.putText(vis, f"Detect FPS: {inst_fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if cam_mode:
                # Show live visualization for webcam
                cv2.imshow(win_name, vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q')):
                    break
            else:
                # Lazy-init writer based on first vis frame size
                if writer is None:
                    h, w = vis.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # widely supported
                    writer = cv2.VideoWriter(output_path, fourcc, float(src_fps), (w, h))
                    if not writer.isOpened():
                        print('Failed to open writer for:', output_path)
                        break
                # Write frame (excluded from FPS timing)
                writer.write(vis)

    cap.release()
    if writer is not None:
        writer.release()
    if cam_mode:
        cv2.destroyAllWindows()

    # Report detection-only FPS for video files
    if not cam_mode and total_detect_time > 0 and total_frames > 0:
        avg_fps = total_frames / total_detect_time
        print(f"Detection-only FPS (excluding video writing): {avg_fps:.2f} ({total_frames} frames)")
        print(f"Video saved to: {output_path}")


if __name__ == '__main__':
    main()
