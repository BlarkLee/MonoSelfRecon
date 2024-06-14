import os
import numpy as np
import pickle
import cv2
from PIL import Image
from torch.utils.data import Dataset
import torch

from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import slic, watershed, felzenszwalb
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float


class ScanNetDataset(Dataset):
    def __init__(self, datapath, mode, transforms, nviews, n_scales):
        super(ScanNetDataset, self).__init__()
        self.datapath = datapath
        self.mode = mode
        self.n_views = nviews
        self.transforms = transforms
        if mode == 'test':
            self.idx_file = os.path.join(self.datapath, 'scannetv2_test.txt')
            self.tsdf_file = 'scans_test_processed'
        if mode == 'train':
            self.idx_file = os.path.join(self.datapath, 'scannetv2_train.txt')
            self.tsdf_file = 'scans_train_processed'

        assert self.mode in ["train", "val", "test"]
        self.metas = self.build_list()
        if mode == 'test':
            self.source_path = 'scans_test'
        else:
            self.source_path = 'scans'

        self.n_scales = n_scales
        self.epoch = None
        self.tsdf_cashe = {}
        self.max_cashe = 100
        
        self.metas_list = []
        for i, meta in enumerate(self.metas):
            for j, frag in enumerate(meta):
                info = {'meta_idx': i, 'frag_idx': j}
                self.metas_list.append(info)

    def build_list(self):
        metas = []
        with open(self.idx_file) as f:
            lines = f.readlines()
            for i, item in enumerate (lines):
                scan_id = item[:-1]
                with open(os.path.join(self.datapath, self.tsdf_file, scan_id, 'fragments.pkl'), 'rb') as f:

                    meta = pickle.load(f)
                    metas.append(meta)
        return metas

    def __len__(self):
        return len(self.metas_list)

    def read_cam_file(self, filepath, vid):
        intrinsics = np.loadtxt(os.path.join(filepath, 'intrinsic', 'intrinsic_color.txt'), delimiter=' ')[:3, :3]
        intrinsics = intrinsics.astype(np.float32)
        extrinsics = np.loadtxt(os.path.join(filepath, 'pose', '{}.txt'.format(str(vid))))
        return intrinsics, extrinsics

    def read_img(self, filepath):
        img = Image.open(filepath)
        return img

    def read_depth(self, filepath):
        # Read depth image and camera pose
        depth_im = cv2.imread(filepath, -1).astype(
            np.float32)
        depth_im /= 1000.  # depth is saved in 16-bit PNG in millimeters
        depth_im[depth_im > 3.0] = 0
        return depth_im

    def read_scene_volumes(self, data_path, scene):
        if scene not in self.tsdf_cashe.keys():
            if len(self.tsdf_cashe) > self.max_cashe:
                self.tsdf_cashe = {}
            full_tsdf_list = []
            for l in range(self.n_scales + 1):
                # load full tsdf volume
                full_tsdf = np.load(os.path.join(data_path, scene, 'full_tsdf_layer{}.npz'.format(l)),
                                    allow_pickle=True)
                full_tsdf_list.append(full_tsdf.f.arr_0)
            self.tsdf_cashe[scene] = full_tsdf_list
        return self.tsdf_cashe[scene]
    
    def gen_rays(self, H, W, c2w, intrinsic):
        c2w = c2w.numpy()
        intrinsic = intrinsic.numpy()
        rays_o = c2w[:3, 3]
        X, Y = np.meshgrid(np.arange(W), np.arange(H))
        XYZ = np.concatenate((X[:, :, None], Y[:, :, None], np.ones_like(X[:, :, None])), axis=-1)
        
        XYZ = XYZ @ np.linalg.inv(intrinsic).T
        XYZ = XYZ @ c2w[:3, :3].T
        rays_d = XYZ.reshape(-1, 3)
        rays = np.concatenate([rays_o[None].repeat(H*W, axis=0), rays_d], axis=-1)
        return rays.astype(np.float32)

    def __getitem__(self, idx):
        # if resume from a checkpoint with a specific epoch and step, two gpus
        '''
        if self.epoch==24 and ((idx>=0 and idx<=5400) or (idx>=9950 and idx<=(9950+5400))):
            return torch.tensor([0])'''
        meta = self.metas[self.metas_list[idx]['meta_idx']][self.metas_list[idx]['frag_idx']]

        imgs = []
        depth = []
        extrinsics_list = []
        intrinsics_list = []
        dso_points = []
        segments = {1:[], 2:[], 4:[]}
        segments_1 = []
        segments_2 = []
        segments_4 = []
        
        
        tsdf_list = self.read_scene_volumes(os.path.join(self.datapath, self.tsdf_file), meta['scene'])

        for i, vid in enumerate(meta['image_ids']):
            # load images
            imgs.append(
                self.read_img(
                    os.path.join(self.datapath, self.source_path, meta['scene'], 'color', '{}.jpg'.format(vid))))

            depth.append(
                self.read_depth(
                    os.path.join(self.datapath, self.source_path, meta['scene'], 'depth', '{}.png'.format(vid)))
            )

            # load intrinsics and extrinsics
            intrinsics, extrinsics = self.read_cam_file(os.path.join(self.datapath, self.source_path, meta['scene']),
                                                        vid)

            intrinsics_list.append(intrinsics)
            extrinsics_list.append(extrinsics)
            
            # ---------------------- segments ---------------------
            if self.mode == 'train':
                scales = [1, 2, 4]
                markers = [400, 100, 25]
                
                
                # use this part to generate and save the plane segmentation when running for the first time
                '''
                image = cv2.imread(os.path.join(self.datapath, self.source_path, meta['scene'], 'color', '{}.jpg'.format(vid)))
                for s, m in zip(scales, markers):
                    image = cv2.resize(image, (640//s, 480//s))
                    image = img_as_float(image)
                    
                    gradient = sobel(rgb2gray(image))
                    # segment = watershed(gradient, markers=m, compactness=0.001)
                    segment = felzenszwalb(image, scale=100, sigma=0.5, min_size=50)
                    segments[s].append(segment.astype(np.int16))
                #print(segments)
                
                seg_save_path = os.path.join('//hdd1/ScanNet/segmentation_all_train', meta['scene'])
                if not os.path.exists(seg_save_path):
                    os.makedirs(seg_save_path)
                np.savez(os.path.join(seg_save_path, "seg_%d.npz"%(vid)), segments)
                
                #np.save(os.path.join('//hdd1/ScanNet/segmentation_haha', "seg_%d_%d.npy"%(idx, vid)), segments)'''
                
               # load the segments if you save it before
                segments_load = np.load(os.path.join('//hdd1/ScanNet/segmentation_all_train', meta['scene'], "seg_%d.npz"%(vid)), allow_pickle=True)['arr_0'].item()
                
                for s, m in zip(scales, markers):
                    segment = segments_load[s][0].astype(np.int64)
                    segments[s].append(segment)

        intrinsics = np.stack(intrinsics_list)
        extrinsics = np.stack(extrinsics_list)
        
        items = {
            'imgs': imgs,
            'depth': depth,
            'intrinsics': intrinsics,
            'extrinsics': extrinsics,
            'tsdf_list_full': tsdf_list,
            'vol_origin': meta['vol_origin'],
            'scene': meta['scene'],
            'fragment': meta['scene'] + '_' + str(meta['fragment_id']),
            'epoch': [self.epoch],
            'image_ids': meta['image_ids'],
            'frag_idx': self.metas_list[idx]['frag_idx']
        }
        
        if self.mode == 'train':
            items['segments']= segments

        if self.transforms is not None:
            items = self.transforms(items)
            
               
        rays = []
        imgs_nerf = []
        h, w = items['imgs'].shape[-2], items['imgs'].shape[-1]
        for i in range (items['imgs'].shape[0]):
            ray = self.gen_rays(h, w, items['extrinsics'][i], items['intrinsics'][i])
            ids = np.random.choice(len(ray), 1024, replace=False)
            ray = ray[ids] #(1024, 6)
            img = items['imgs'][i].reshape(3, -1).transpose(1, 0)[ids]  # (1024, 3)
            img = img.type(torch.float32) / 255
            rays.append(ray)
            imgs_nerf.append(img)
        items['rays'] = rays #(9, 1024, 6)
        items['imgs_nerf'] = imgs_nerf #(9, 1024, 3)   
        
        return items

