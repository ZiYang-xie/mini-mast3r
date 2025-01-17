from pathlib import Path
from argparse import ArgumentParser
import torch
import json
import numpy as np
import imageio
import open3d as o3d
import os

import torch.nn.functional as F

from mini_mast3r.api import OptimizedResult, inferece_mast3r, log_optimized_result
from mini_mast3r.model import AsymmetricMASt3R

def export_results(optimized_results: OptimizedResult, output_path: str):
    depth_hw_list = optimized_results.depth_hw_list
    H, W = depth_hw_list[0].shape

    images = optimized_results.rgb_hw3_list
    masks = optimized_results.masks_list
    depths = optimized_results.depth_hw_list
    
    image_output_path = os.path.join(output_path, "images")
    os.makedirs(image_output_path, exist_ok=True)
    for i, image in enumerate(images):
        image = (image * 255).astype(np.uint8)
        image_path = os.path.join(image_output_path, f"{i:04d}.png")
        imageio.imwrite(image_path, image)

    mask_output_path = os.path.join(output_path, "masks")
    os.makedirs(mask_output_path, exist_ok=True)
    for i, mask in enumerate(masks):
        mask = (mask * 255).astype(np.uint8)
        mask_path = os.path.join(mask_output_path, f"{i:04d}.png")
        imageio.imwrite(mask_path, mask)

    depth_output_path = os.path.join(output_path, "depth")
    os.makedirs(depth_output_path, exist_ok=True)
    for i, depth in enumerate(depths):
        depth_path = os.path.join(depth_output_path, f"{i:04d}.npy")
        np.save(depth_path, depth)
    

    w2c_pose = optimized_results.world_T_cam_b44
    intrinsic = optimized_results.K_b33
    intr = intrinsic.mean(0)
    results = {
        "camera_type": "perspective",
        "render_height": int(H), 
        "render_width": int(W),
        "camera_path": [],
        "fps": 24,
        "smoothness_value": 0,
        "is_cycle": True
    }
    for i, w2c in enumerate(w2c_pose):
        c2w = np.linalg.inv(w2c)
        cam_dict = {
            "camera_to_world": c2w.flatten().tolist(),
            "fov": 2 * np.arctan(int(W) / (2 * float(intr[0, 0]))) * 180 / np.pi,
            "aspect": 1
        }
        results["camera_path"].append(cam_dict)
    
    # Save pointcloud
    pts = optimized_results.point_cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)


    # Export Mesh
    mesh = optimized_results.mesh
    mesh.export(f"{output_path}/mesh.obj")

    # o3d.io.write_point_cloud(f"{output_path}/pointcloud.ply", pcd)
    confs = optimized_results.conf_hw_list
    confs = np.stack(confs, axis=0)
    confs = confs[masks]

    colors = np.array(pts.colors)[:,:3] / 255.0
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd.normals = o3d.utility.Vector3dVector(confs[..., None].repeat(3, axis=-1))

    pcd = pcd.voxel_down_sample(voxel_size=0.0025)
    confs = np.asarray(pcd.normals)[:,0]
    np.save(f"{output_path}/conf.npy", confs)

    o3d.io.write_point_cloud(f"{output_path}/pointcloud.ply", pcd)
    results['ply_file_path'] = "pointcloud.ply"

    with open(f"{output_path}/transforms.json", "w") as f:
        json.dump(results, f, indent=4)


def main(image_dir: Path):
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    model = AsymmetricMASt3R.from_pretrained(
        "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    ).to(device)

    optimized_results = inferece_mast3r(
        image_dir_or_list=image_dir,
        model=model,
        device=device,
        batch_size=1,
    )

    output_path = str(image_dir.parent)
    export_results(optimized_results, output_path)

if __name__ == "__main__":
    parser = ArgumentParser("mini-dust3r rerun demo script")
    parser.add_argument(
        "--image-dir",
        type=Path,
        help="Directory containing images to process",
        required=True,
    )
    args = parser.parse_args()
    main(args.image_dir)
