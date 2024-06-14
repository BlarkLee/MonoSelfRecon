import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsparse.tensor import PointTensor
from loguru import logger
import random

from models.modules import SPVCNN
from utils import apply_log_transform
from .gru_fusion import GRUFusion
from ops.back_project_sfm_att import back_project, back_project_gru
from ops.generate_grids import generate_grid
from ops.rgb_consistency_loss_seg_selectsuper import rgb_consistency_loss
from .transformer import Transformer

class NeuConNet(nn.Module):
    '''
    Coarse-to-fine network.
    '''

    def __init__(self, cfg):
        super(NeuConNet, self).__init__()
        self.cfg = cfg
        self.n_scales = len(cfg.THRESHOLDS) - 1

        alpha = int(self.cfg.BACKBONE2D.ARC.split('-')[-1])
        ch_in = [64 * alpha + 1, 96 + 64 * alpha + 2 + 1, 48 + 32 * alpha + 2 + 1, 24 + 24 + 2 + 1]
        channels = [96, 48, 24]
        channels_att = [64, 64, 32]

        if self.cfg.FUSION.FUSION_ON:
            # GRU Fusion
            self.gru_fusion = GRUFusion(cfg, channels)
        # sparse conv
        self.sp_convs = nn.ModuleList()
        # MLPs that predict tsdf and occupancy.
        self.tsdf_preds = nn.ModuleList()
        self.occ_preds = nn.ModuleList()
        self.transformer = nn.ModuleList()
        self.transformer_mlp = nn.ModuleList()
        for i in range(len(cfg.THRESHOLDS)):
            self.sp_convs.append(
                SPVCNN(num_classes=1, in_channels=ch_in[i],
                       pres=1,
                       cr=1 / 2 ** i,
                       vres=self.cfg.VOXEL_SIZE * 2 ** (self.n_scales - i),
                       dropout=self.cfg.SPARSEREG.DROPOUT)
            )
            self.tsdf_preds.append(nn.Linear(channels[i], 1))
            self.occ_preds.append(nn.Linear(channels[i], 1))
            
            self.transformer_temp = Transformer(
            channels_att[i],
            channels_att[i] * 2,
            num_layers=2, #self.cfg.n_layers_att,
            num_heads=2, #self.cfg.n_attn_heads,
        )
            self.transformer.append(self.transformer_temp)
            self.transformer_mlp_temp = torch.nn.Linear(channels_att[i], 1, bias=True)
            
            torch.nn.init.kaiming_normal_(self.transformer_mlp_temp.weight)
            torch.nn.init.zeros_(self.transformer_mlp_temp.bias)
            
            self.transformer_mlp.append(self.transformer_mlp_temp)


    def get_target(self, coords, inputs, scale):
        '''
        Won't be used when 'fusion_on' flag is turned on
        :param coords: (Tensor), coordinates of voxels, (N, 4) (4 : Batch ind, x, y, z)
        :param inputs: (List), inputs['tsdf_list' / 'occ_list']: ground truth volume list, [(B, DIM_X, DIM_Y, DIM_Z)]
        :param scale:
        :return: tsdf_target: (Tensor), tsdf ground truth for each predicted voxels, (N,)
        :return: occ_target: (Tensor), occupancy ground truth for each predicted voxels, (N,)
        '''
        with torch.no_grad():
            tsdf_target = inputs['tsdf_list'][scale]
            occ_target = inputs['occ_list'][scale]
            coords_down = coords.detach().clone().long()
            coords_down[:, 1:] = (coords[:, 1:] // 2 ** scale)
            tsdf_target = tsdf_target[coords_down[:, 0], coords_down[:, 1], coords_down[:, 2], coords_down[:, 3]]
            occ_target = occ_target[coords_down[:, 0], coords_down[:, 1], coords_down[:, 2], coords_down[:, 3]]
            return tsdf_target, occ_target

    def upsample(self, pre_feat, pre_coords, interval, num=8):
        '''

        :param pre_feat: (Tensor), features from last level, (N, C)
        :param pre_coords: (Tensor), coordinates from last level, (N, 4) (4 : Batch ind, x, y, z)
        :param interval: interval of voxels, interval = scale ** 2
        :param num: 1 -> 8
        :return: up_feat : (Tensor), upsampled features, (N*8, C)
        :return: up_coords: (N*8, 4), upsampled coordinates, (4 : Batch ind, x, y, z)
        '''
        with torch.no_grad():
            pos_list = [1, 2, 3, [1, 2], [1, 3], [2, 3], [1, 2, 3]]
            n, c = pre_feat.shape
            up_feat = pre_feat.unsqueeze(1).expand(-1, num, -1).contiguous()
            up_coords = pre_coords.unsqueeze(1).repeat(1, num, 1).contiguous()
            for i in range(num - 1):
                up_coords[:, i + 1, pos_list[i]] += interval

            up_feat = up_feat.view(-1, c)
            up_coords = up_coords.view(-1, 4)

        return up_feat, up_coords

    def forward(self, features, inputs, outputs, extrinsics_last, proj_matrices_last, imgs_last, segments_last):
        
        
        '''

        :param features: list: features for each image: eg. list[0] : pyramid features for image0 : [(B, C0, H, W), (B, C1, H/2, W/2), (B, C2, H/2, W/2)]
        :param inputs: meta data from dataloader
        :param outputs: {}
        :return: outputs: dict: {
            'coords':                  (Tensor), coordinates of voxels,
                                    (number of voxels, 4) (4 : batch ind, x, y, z)
            'tsdf':                    (Tensor), TSDF of voxels,
                                    (number of voxels, 1)
        }
        :return: loss_dict: dict: {
            'tsdf_occ_loss_X':         (Tensor), multi level loss
        }
        '''
        
        bs = features[0][0].shape[0]
        pre_feat = None
        pre_coords = None
        loss_dict = {}
        rgb_loss_dict = {}
        planar_loss_dict = {}
        src_depth = {}
        tgt1_depth = {}
        tgt2_depth = {}
        src_img_xy = {}
        tgt1_img_xy = {}
        tgt2_img_xy = {}
        src_idx = {}
        tgt1_idx = {}
        tgt2_idx = {}
        depth_cam0_dict = {}
        depth_cam1_dict = {}
        depth_cam0_dict[0] = {}
        depth_cam1_dict[0] = {}
        depth_cam0_dict[1] = {}
        depth_cam1_dict[1] = {}
        depth_cam0_dict[2] = {}
        depth_cam1_dict[2] = {}
        #depth_cam0_proj2cam1_dict = {}
        #depth_cam1_proj2cam0_dict = {}
        im_xy_0_dict = {}
        im_xy_1_dict = {}
        im_xy_0_dict[0] = {}
        im_xy_1_dict[0] = {}
        im_xy_0_dict[1] = {}
        im_xy_1_dict[1] = {}
        im_xy_0_dict[2] = {}
        im_xy_1_dict[2] = {}
        #im_xy_recon0_in_cam1_dict = {}
        #im_xy_recon1_in_cam0_dict = {}
        # ----coarse to fine----
        for i in range(self.cfg.N_LAYER):
            interval = 2 ** (self.n_scales - i)
            scale = self.n_scales - i

            if i == 0:
                # ----generate new coords----
                coords = generate_grid(self.cfg.N_VOX, interval)[0]
                up_coords = []
                for b in range(bs):
                    up_coords.append(torch.cat([torch.ones(1, coords.shape[-1]).to(coords.device) * b, coords]))
                up_coords = torch.cat(up_coords, dim=1).permute(1, 0).contiguous()
            else:
                # ----upsample coords----
                up_feat, up_coords = self.upsample(pre_feat, pre_coords, interval)

            # ----back project----
            feats = torch.stack([feat[scale] for feat in features]) #(frames=9, bs=1, c=80, h=30, w=40)
            KRcam = inputs['proj_matrices'][:, :, scale].permute(1, 0, 2, 3).contiguous()
            #if self.training:
            volume, count, img_xy_all, mask_all, rs_grid = back_project(up_coords, inputs['vol_origin_partial'], self.cfg.VOXEL_SIZE, feats,
                                         KRcam, self.transformer[i],  self.transformer_mlp[i], True)
            
            grid_mask = count > 1

            # ----concat feature from last stage----
            if i != 0:
                feat = torch.cat([volume, up_feat], dim=1)
            else:
                feat = volume

            if not self.cfg.FUSION.FUSION_ON:
                tsdf_target, occ_target = self.get_target(up_coords, inputs, scale)

            # ----convert to aligned camera coordinate----
            r_coords = up_coords.detach().clone().float()
            for b in range(bs):
                batch_ind = torch.nonzero(up_coords[:, 0] == b).squeeze(1)
                coords_batch = up_coords[batch_ind][:, 1:].float()
                coords_batch = coords_batch * self.cfg.VOXEL_SIZE + inputs['vol_origin_partial'][b].float()

                coords_batch = torch.cat((coords_batch, torch.ones_like(coords_batch[:, :1])), dim=1)
                coords_batch = coords_batch @ inputs['world_to_aligned_camera'][b, :3, :].permute(1, 0).contiguous()
                r_coords[batch_ind, 1:] = coords_batch

            # batch index is in the last position
            r_coords = r_coords[:, [1, 2, 3, 0]]
            if i == self.cfg.N_LAYER-1 :
                coords_aligned_camera = r_coords[:, :3]

            # ----sparse conv 3d backbone----
            point_feat = PointTensor(feat, r_coords)
            feat = self.sp_convs[i](point_feat)

            # ----gru fusion----
            do_frag = False
            if self.cfg.FUSION.FUSION_ON:
                up_coords_old = up_coords.clone()
                up_coords, feat, tsdf_target, occ_target, gru_mask = self.gru_fusion(up_coords, feat, inputs, i)
                if (up_coords.shape != up_coords_old.shape):
                    KRcam_last = proj_matrices_last[:, :, scale].permute(1, 0, 2, 3).contiguous()
                    KRcam_all = torch.cat((KRcam_last, KRcam), 0)
                    count, img_xy_all, mask_all, rs_grid = back_project_gru(up_coords, inputs['vol_origin_partial'], self.cfg.VOXEL_SIZE, feats, KRcam_all, True) #self.training)
                    grid_mask = torch.ones_like(feat[:, 0]).bool()
                    do_frag = True

            tsdf = self.tsdf_preds[i](feat) #[13824,1]
            tsdf_mask = torch.zeros_like(tsdf).bool()
            tsdf_mask[(tsdf < 0.3) & (tsdf > -0.3)] = True #[13824,1]
            grid_mask = (tsdf_mask.squeeze(1).long() * grid_mask.long()).bool() #[13824,1]

            # ------define the sparsity for the next stage-----
            occupancy = grid_mask
            occ = occupancy.unsqueeze(1)

            num = int(occupancy.sum().data.cpu())

            if num == 0:
                logger.warning('no valid points: scale {}'.format(i))
                outputs['pix_coords_all'] = None
                rgb_loss = None
                planar_loss = None
                total_loss = None
                return outputs, rgb_loss, planar_loss, total_loss, None, None, None, None, None, None, None #, loss_dict

            # ------avoid out of memory: sample points if num of points is too large-----
            if self.training and num > self.cfg.TRAIN_NUM_SAMPLE[i] * bs:
                choice = np.random.choice(num, num - self.cfg.TRAIN_NUM_SAMPLE[i] * bs,
                                          replace=False)
                ind = torch.nonzero(occupancy)
                occupancy[ind[choice]] = False
            
            pre_coords = up_coords[occupancy]
            for b in range(bs):
                batch_ind = torch.nonzero(pre_coords[:, 0] == b).squeeze(1)
                if len(batch_ind) == 0:
                    logger.warning('no valid points: scale {}, batch {}'.format(i, b))
                    outputs['pix_coords_all'] = None
                    rgb_loss = None
                    planar_loss = None
                    total_loss = None
                    return outputs, rgb_loss, planar_loss, total_loss, None, None, None, None, None, None, None #, loss_dict
            pre_feat = feat[occupancy]
            pre_tsdf = tsdf[occupancy]
            pre_occ = occ[occupancy]

            pre_feat = torch.cat([pre_feat, pre_tsdf, pre_occ], dim=1)

            if i == self.cfg.N_LAYER - 1:
                outputs['coords'] = pre_coords
                outputs['tsdf'] = pre_tsdf
                outputs['feat'] = pre_feat #feat[occupancy]
                outputs['occupancy'] = occupancy
                
            #if self.training:
            if True: 
                SDF_pairs = {}
                
                rgb_loss = torch.tensor([]).cuda()
                planar_loss = 0
                if not do_frag:
                    rs_grid_cam = torch.inverse(inputs['extrinsics'][0]) @ rs_grid # rs_grid_cam [9,4,13824]
                    
                else:
                    inputs_imgs = torch.cat((imgs_last, inputs['imgs']), 1)
                    inputs_proj_matrices = torch.cat((proj_matrices_last, inputs['proj_matrices']), 1)
                    inputs_extrinsics = torch.cat((extrinsics_last, inputs['extrinsics']), 1)
                    inputs_segments = inputs['segments']
                    if self.training:
                        for key in inputs_segments.keys():
                            inputs_segments[key] = segments_last[key] + inputs_segments[key]
                    rs_grid_cam = torch.inverse(inputs_extrinsics[0]) @ rs_grid
                
                # mask_all 52 keys (52 is subset of 8*9=72), each key has (1, 13824)
                for (i,j) in mask_all.keys():
                    if not self.cfg.FUSION.FUSION_ON and (abs(i-j)>5):
                        continue
                    if self.cfg.FUSION.FUSION_ON and not do_frag and (abs(i-j)>3):
                        continue
                    if do_frag and (i<7 or i>10 or j<7 or j>10 or abs(i-j)>3 or i==j):
                        continue

                    SDF_pairs[i,j] = tsdf[mask_all[i,j][0]] # mask_all[i,j] = [1,13824], SDF_pairs[i,j] = [n,1], img_xy_all[i,j] = [2,n,2]
                    rs_grid_cam_0 = rs_grid_cam[i].transpose(1,0)[mask_all[i,j][0]] # (n, 4)
                    rs_grid_cam_1 = rs_grid_cam[j].transpose(1,0)[mask_all[i,j][0]] # (n, 4)
                   
                    # ------------------------
                    #surf_pairs[i,j] = [2,n,3], rs_grid_cam_0/1 = [n, 4]
                    if do_frag:
                        rgb_loss_ij, planar_loss_ij, depth_cam0, im_xy_0 = rgb_consistency_loss(SDF_pairs[i,j], rs_grid_cam_0[:, :3], rs_grid_cam_1[:, :3], inputs_extrinsics, inputs_proj_matrices, inputs_imgs, inputs_segments, img_xy_all[i,j], i, j, scale, save_depth=True, is_training=self.training)
                    else:
                        if not self.training:
                            inputs['segments'] = None
                        rgb_loss_ij, planar_loss_ij, depth_cam0, im_xy_0 = rgb_consistency_loss(SDF_pairs[i,j], rs_grid_cam_0[:, :3], rs_grid_cam_1[:, :3], inputs['extrinsics'], inputs['proj_matrices'], inputs['imgs'], inputs['segments'], img_xy_all[i,j], i, j, scale, save_depth=True, is_training=self.training)
                    if rgb_loss_ij is None:
                        continue
                    rgb_loss = torch.cat((rgb_loss, rgb_loss_ij), dim=1)
                    planar_loss += planar_loss_ij
                    if do_frag:
                        if i >= 9:
                            depth_cam0_dict[scale][i] = depth_cam0
                            im_xy_0_dict[scale][i] = im_xy_0
                    else:
                        depth_cam0_dict[scale][i] = depth_cam0
                        im_xy_0_dict[scale][i] = im_xy_0
                
                rgb_loss = rgb_loss.mean()
                rgb_loss_dict[scale] = rgb_loss
                planar_loss_dict[scale] = planar_loss
                
                if self.cfg.FUSION.FUSION_ON and scale!=0:
                    continue
                haha = [depth_cam0_dict[scale][key].shape for key in depth_cam0_dict[scale].keys()]
                if do_frag and len(haha)!=2:
                    continue
                if depth_cam0 is None or len(haha)==0:
                    src_depth[scale] = None
                    tgt1_depth[scale] = None
                    tgt2_depth[scale] = None
                    src_img_xy[scale] = None
                    tgt1_img_xy[scale] = None
                    tgt2_img_xy[scale] = None
                    src_idx[scale] = None
                    tgt1_idx[scale] = None
                    tgt2_idx[scale] = None
                elif do_frag and depth_cam0.shape==0:
                    src_depth[scale] = None
                    tgt1_depth[scale] = None
                    tgt2_depth[scale] = None
                    src_img_xy[scale] = None
                    tgt1_img_xy[scale] = None
                    tgt2_img_xy[scale] = None
                    src_idx[scale] = None
                    tgt1_idx[scale] = None
                    tgt2_idx[scale] = None
                else:
                    hehe = [key for key in depth_cam0_dict[scale].keys()]
                    
                    max_index = hehe[haha.index(max(haha))]
                    if do_frag:
                        max_index -= 9
                    
                    second_max_index = hehe[haha.index(max(haha[:max_index] + haha[max_index+1:]))]
                    if do_frag:
                        second_max_index -= 9

                    if do_frag:
                        max_index += 9
                        second_max_index += 9
            
                    src_depth[scale] = depth_cam0_dict[scale][max_index]
                    tgt1_depth[scale] = depth_cam0_dict[scale][second_max_index]
                    #tgt2_depth[scale] = depth_cam0_dict[scale][third_max_index]
                    src_img_xy[scale] = im_xy_0_dict[scale][max_index]
                    tgt1_img_xy[scale] = im_xy_0_dict[scale][second_max_index]
                    #tgt2_img_xy[scale] = im_xy_0_dict[scale][third_max_index]
                    if do_frag:
                        max_index -= 9
                        second_max_index -= 9
                    src_idx[scale] = max_index
                    tgt1_idx[scale] = second_max_index
                    #tgt2_idx[scale] = third_max_index

        #if self.training:
        if True:
            rgb_loss = 0.26*rgb_loss_dict[0] + 0.33*rgb_loss_dict[1] + 0.41*rgb_loss_dict[2]
            planar_loss = 0.26*planar_loss_dict[0] + 0.33*planar_loss_dict[1] + 0.41*planar_loss_dict[2]
            gt_loss = torch.tensor(0.)
            total_loss = rgb_loss + 0.05*planar_loss
            
        return outputs, rgb_loss, planar_loss, gt_loss, total_loss, src_depth, tgt1_depth, src_img_xy, tgt1_img_xy, src_idx, tgt1_idx #depth_cam_dict, im_xy_dict #, loss_dict
    
    @staticmethod
    def pts2pix(pts, intrinsics, extrinsics, h, w):
        proj_mat = intrinsics @ extrinsics[:, :3, :4]
    
        n_views = proj_mat.shape[0]
        rs_grid = pts.unsqueeze(0).expand(n_views, -1, -1)
        rs_grid = rs_grid.permute(0, 2, 1).contiguous()
        nV = rs_grid.shape[-1]
        rs_grid = torch.cat([rs_grid, torch.ones([n_views, 1, nV]).cuda()], dim=1)
        
        # Project grid
        im_p = proj_mat @ rs_grid
        im_x, im_y, im_z = im_p[:, 0], im_p[:, 1], im_p[:, 2]
        im_x = im_x / im_z
        im_y = im_y / im_z

        im_grid = torch.stack([im_x, im_y], dim=-1)
        im_grid = im_grid.view(n_views, 1, -1, 2)

        p_cam = (extrinsics @ rs_grid)[:, :3, :]
        z_cam = p_cam[:, 2]

        return im_grid, z_cam, p_cam
    
    @staticmethod
    def compute_loss(tsdf, occ, tsdf_target, occ_target, loss_weight=(1, 1),
                     mask=None, pos_weight=1.0):
        '''

        :param tsdf: (Tensor), predicted tsdf, (N, 1)
        :param occ: (Tensor), predicted occupancy, (N, 1)
        :param tsdf_target: (Tensor),ground truth tsdf, (N, 1)
        :param occ_target: (Tensor), ground truth occupancy, (N, 1)
        :param loss_weight: (Tuple)
        :param mask: (Tensor), mask voxels which cannot be seen by all views
        :param pos_weight: (float)
        :return: loss: (Tensor)
        '''
        # compute occupancy/tsdf loss
        tsdf = tsdf.view(-1)
        occ = occ.view(-1)
        tsdf_target = tsdf_target.view(-1)
        occ_target = occ_target.view(-1)
        if mask is not None:
            mask = mask.view(-1)
            tsdf = tsdf[mask]
            occ = occ[mask]
            tsdf_target = tsdf_target[mask]
            occ_target = occ_target[mask]

        n_all = occ_target.shape[0]
        n_p = occ_target.sum()
        if n_p == 0:
            logger.warning('target: no valid voxel when computing loss')
            return torch.Tensor([0.0]).cuda()[0] * tsdf.sum()
        w_for_1 = (n_all - n_p).float() / n_p
        w_for_1 *= pos_weight

        # compute occ bce loss
        occ_loss = F.binary_cross_entropy_with_logits(occ, occ_target.float(), pos_weight=w_for_1)

        # compute tsdf l1 loss
        tsdf = apply_log_transform(tsdf[occ_target])
        tsdf_target = apply_log_transform(tsdf_target[occ_target])
        tsdf_loss = torch.mean(torch.abs(tsdf - tsdf_target))

        # compute final loss
        loss = loss_weight[0] * occ_loss + loss_weight[1] * tsdf_loss
        return loss
