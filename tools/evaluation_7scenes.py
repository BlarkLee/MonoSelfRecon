# This file is derived from [Atlas](https://github.com/magicleap/Atlas).
# Originating Author: Zak Murez (zak.murez.com)
# Modified for [NeuralRecon](https://github.com/zju3dv/NeuralRecon) by Yiming Xie.

# Original header:
# Copyright 2020 Magic Leap, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys

sys.path.append('.')
import argparse
import json
import os

import numpy as np
import pyrender
import torch
import trimesh
from tools.simple_loader import *

from tools.evaluation_utils import eval_depth, eval_mesh
from tools.visualize_metrics import visualize
import open3d as o3d
import ray

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

torch.multiprocessing.set_sharing_strategy('file_system')


def parse_args():
    parser = argparse.ArgumentParser(description="NeuralRecon ScanNet Testing")
    parser.add_argument("--model", required=True, metavar="FILE",
                        help="path to checkpoint")
    parser.add_argument('--max_depth', default=10., type=float,
                        help='mask out large depth values since they are noisy')
    parser.add_argument("--data_path", metavar="DIR",
                        help="path to dataset", default='/hdd1/7scene/stairs/test/')
    parser.add_argument("--gt_path", metavar="DIR",
                        help="path to raw dataset", default='/hdd1/7scene/stairs/processed_test/')

    # ray config
    parser.add_argument('--n_proc', type=int, default=1, help='#processes launched to process scenes.')
    parser.add_argument('--n_gpu', type=int, default=1, help='#number of gpus')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--loader_num_workers', type=int, default=8)
    return parser.parse_args()


args = parse_args()


class Renderer():
    """OpenGL mesh renderer

    Used to render depthmaps from a mesh for 2d evaluation
    """

    def __init__(self, height=480, width=640):
        self.renderer = pyrender.OffscreenRenderer(width, height)
        self.scene = pyrender.Scene()
        # self.render_flags = pyrender.RenderFlags.SKIP_CULL_FACES

    def __call__(self, height, width, intrinsics, pose, mesh):
        self.renderer.viewport_height = height
        self.renderer.viewport_width = width
        self.scene.clear()
        self.scene.add(mesh)
        cam = pyrender.IntrinsicsCamera(cx=intrinsics[0, 2], cy=intrinsics[1, 2],
                                        fx=intrinsics[0, 0], fy=intrinsics[1, 1])
        self.scene.add(cam, pose=self.fix_pose(pose))
        return self.renderer.render(self.scene)  # , self.render_flags)

    def fix_pose(self, pose):
        # 3D Rotation about the x-axis.
        t = np.pi
        c = np.cos(t)
        s = np.sin(t)
        R = np.array([[1, 0, 0],
                      [0, c, -s],
                      [0, s, c]])
        axis_transform = np.eye(4)
        axis_transform[:3, :3] = R
        return pose @ axis_transform

    def mesh_opengl(self, mesh):
        return pyrender.Mesh.from_trimesh(mesh)

    def delete(self):
        self.renderer.delete()


def process(scene, total_scenes_index, total_scenes_count):
    if not (os.path.isfile(os.path.join(args.model, '%s_metrics.json' % scene))):
        #if True:
        print("no this scene")
        #print(os.path.join(args.model, '%s_metrics.json' % scene))
        save_path = args.model
        width, height = 640, 480
    
        #test_framid = os.listdir(os.path.join(args.data_path, scene, 'color'))
        test_framid = [i for i in os.listdir(os.path.join(args.data_path, scene)) if i[-9:-4] == 'color']
        n_imgs = len(test_framid)
        #intrinsic_dir = os.path.join(args.data_path, scene, 'intrinsic', 'intrinsic_depth.txt')
        #cam_intr = np.loadtxt(intrinsic_dir, delimiter=' ')[:3, :3]
        cam_intr = np.array([[585., 0., 320., 0.],
                                 [0., 585., 240., 0.],
                                 [0., 0., 1., 0.],
                                 [0., 0., 0., 1.]]).astype(np.float32)[:3, :3]
        #dataset = ScanNetDataset(n_imgs, scene, args.data_path, args.max_depth)
        dataset = SevenscenesDataset(n_imgs, scene, args.data_path, args.max_depth)
    
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, collate_fn=collate_fn,
                                             batch_sampler=None, num_workers=args.loader_num_workers)

        voxel_size = 4

        # re-fuse to remove hole filling since filled holes are penalized in
        # mesh metrics
        # tsdf_fusion = TSDFFusion(vol_dim, float(voxel_size)/100, origin, color=False)
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=float(voxel_size) / 100,
        sdf_trunc=3 * float(voxel_size) / 100,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

        mesh_file = os.path.join(save_path, '%s.ply' % scene.replace('/', '-'))
        #print("mesh_file", mesh_file)
        #print(stop)
        mesh = trimesh.load(mesh_file, process=False)
        # mesh renderer
        renderer = Renderer()
        mesh_opengl = renderer.mesh_opengl(mesh)
    
        for i, (cam_pose, depth_trgt, _) in enumerate(dataloader):
            print(total_scenes_index, total_scenes_count, scene, i, len(dataloader))
            '''
            if cam_pose[0][0] == np.inf or cam_pose[0][0] == -np.inf or cam_pose[0][0] == np.nan:
                continue'''
            '''    
            if i%10!=0:
                continue'''
            _, depth_pred = renderer(height, width, cam_intr, cam_pose, mesh_opengl)
            #print("depth_pred_range", depth_pred.shape, np.unique(depth_pred))
            #print("depth_trgt_range", depth_trgt.shape, np.unique(depth_trgt))
            
            '''
            #print("cam_intr", cam_intr.shape)
            #print("cam_pose", cam_pose.shape)
            depth_path = '//hdd1/scannet_est_depth/'#os.path.join(save_path, "depth")
            #np.save(os.path.join(depth_path, "depth_pred_%s_%d.npy"%(scene, i)), depth_pred)
            #np.save(os.path.join(depth_path, "depth_gt_%s_%d.npy"%(scene, i)), depth_gt)
            
            # Choose a colormap (replace 'viridis' with the desired colormap)
            cmap_name = 'viridis'
            cmap = plt.get_cmap(cmap_name)
            
            # Normalize the depth values to [0, 1] for colormap mapping
            #depth_pred[depth_pred>3.0] = 3.0
            normalized_depth_pred = (depth_pred - np.min(depth_pred)) / (np.max(depth_pred) - np.min(depth_pred))
            normalized_depth_gt = (depth_trgt - np.min(depth_trgt)) / (np.max(depth_trgt) - np.min(depth_trgt))

            # Apply the colormap to the normalized depth values
            colored_depth_pred = cmap(normalized_depth_pred)
            colored_depth_gt = cmap(normalized_depth_gt)

            # Save the colored depth map as an image
            output_file_pred = os.path.join(depth_path, "depth_pred_%s_%d.png"%(scene, i))  # Output file name
            output_file_gt = os.path.join(depth_path, "depth_gt_%s_%d.png"%(scene, i))
            plt.imsave(output_file_pred, colored_depth_pred)
            plt.imsave(output_file_gt, colored_depth_gt)'''
            
            
            temp = eval_depth(depth_pred, depth_trgt)
            if i == 0:
                metrics_depth = temp
            else:
                metrics_depth = {key: value + temp[key]
                                 for key, value in metrics_depth.items()}

            # placeholder
            color_im = np.repeat(depth_pred[:, :, np.newaxis] * 255, 3, axis=2).astype(np.uint8)
            depth_pred = o3d.geometry.Image(depth_pred)
            color_im = o3d.geometry.Image(color_im)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color_im, depth_pred, depth_scale=1.0,
                                                                  depth_trunc=5.0,
                                                                  convert_rgb_to_intensity=False)

            volume.integrate(
            rgbd,
            o3d.camera.PinholeCameraIntrinsic(width=width, height=height, fx=cam_intr[0, 0], fy=cam_intr[1, 1],
                                              cx=cam_intr[0, 2],
                                              cy=cam_intr[1, 2]), np.linalg.inv(cam_pose))

        metrics_depth = {key: value / len(dataloader)
                         for key, value in metrics_depth.items()}
        


        # save trimed mesh
        file_mesh_trim = os.path.join(save_path, '%s_trim_single.ply' % scene.replace('/', '-'))
        o3d.io.write_triangle_mesh(file_mesh_trim, volume.extract_triangle_mesh())

        # eval trimed mesh
        #file_mesh_trgt = os.path.join(args.gt_path, scene, scene + '_vh_clean_2.ply')
        file_mesh_trgt = os.path.join(args.gt_path, scene, 'mesh_layer0.ply') 
        metrics_mesh = eval_mesh(file_mesh_trim, file_mesh_trgt)

        metrics = {**metrics_depth, **metrics_mesh}

        rslt_file = os.path.join(save_path, '%s_metrics.json' % scene.replace('/', '-'))
        json.dump(metrics, open(rslt_file, 'w'))
       
    if(os.path.isfile(os.path.join(args.model, '%s_metrics.json' % scene))):
        print("have this scene")
        metrics = json.load(open(os.path.join(args.model, '%s_metrics.json' % scene), 'r'))

    return scene, metrics


@ray.remote(num_cpus=args.num_workers + 1, num_gpus=(1 / args.n_proc))
def process_with_single_worker(info_files):
    metrics = {}
    for i, info_file in enumerate(info_files):
        '''
        #haha = os.path.join(args.model, '%s_metrics.json' % info_file)
        if(os.path.isfile(os.path.join(args.model, '%s_metrics.json' % info_file))):
            continue'''
        scene, temp = process(info_file, i, len(info_files))
        #process(info_file, i, len(info_files))
        
        if temp is not None:
            metrics[scene] = temp
    return metrics


def split_list(_list, n):
    assert len(_list) >= n
    ret = [[] for _ in range(n)]
    for idx, item in enumerate(_list):
        ret[idx % n].append(item)
    return ret


def main():
    all_proc = args.n_proc * args.n_gpu

    ray.init(num_cpus=all_proc * (args.num_workers + 1), num_gpus=args.n_gpu)

    info_files = sorted(os.listdir(args.data_path))

    info_files = split_list(info_files, all_proc)
    #print("args.data_path", args.data_path)
    print("info_files", info_files[0])
    #info_files[0] = info_files[0][:-2]

    ray_worker_ids = []
    for w_idx in range(all_proc):
        ray_worker_ids.append(process_with_single_worker.remote(info_files[w_idx]))

    results = ray.get(ray_worker_ids)

    metrics = {}
    for r in results:
        metrics.update(r)

    rslt_file = os.path.join(args.model, 'metrics.json')
    json.dump(metrics, open(rslt_file, 'w'))

    # display results
    visualize(rslt_file)


if __name__ == "__main__":
    main()
