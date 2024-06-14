import torch
import torch.nn as nn

from .backbone_resnet import ResnetEncoder
from .NeRF.monodepth2.depth_decoder_light import DepthDecoder
from .neucon_network_sfm_frag_attention_gt import NeuConNet #training for weakly-supervision
#from .neucon_network_sfm_frag_attention import NeuConNet #training for purely self-supervision
from .NeRF.operations import rendering_utils, mpi_rendering
from .NeRF.operations.homography_sampler import HomographySample
from .NeRF.utils import restore_model, run_shell_cmd, get_embedder, AverageMeter, inverse, disparity_normalization_vis
from .NeRF.ssim import SSIM
from .NeRF.layers import edge_aware_loss, edge_aware_loss_v2, psnr
from .gru_fusion import GRUFusion
from utils import tocuda
import numpy as np
import torch.nn.functional as F
from skimage import measure
import trimesh
import pyrender
import lpips


def _get_disparity_list(cfg, B, device=torch.device("cuda")):
    S_coarse, S_fine = cfg.num_bins_coarse, cfg.num_bins_fine
    disparity_start, disparity_end = cfg.disparity_start, cfg.disparity_end

    disparity_coarse_src = rendering_utils.uniformly_sample_disparity_from_linspace_bins(
                batch_size=B,
                num_bins=S_coarse,
                start=disparity_start,
                end=disparity_end,
                device=device
            )
    return disparity_coarse_src



class SelfRecon(nn.Module):
    '''
    SelfRecon main class.
    '''

    def __init__(self, cfg):
        super(SelfRecon, self).__init__()
        self.cfg = cfg.MODEL
        alpha = float(self.cfg.BACKBONE2D.ARC.split('-')[-1])

        # other hparams
        self.pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1)
        self.pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1)
        self.n_scales = len(self.cfg.THRESHOLDS) - 1

        # networks
        self.backbone2d = ResnetEncoder(num_layers=18, pretrained=True)
        self.neucon_net = NeuConNet(cfg.MODEL)
        self.embedder, out_dim = get_embedder(10)
        self.nerf = DepthDecoder(
            # Common params
            #num_ch_enc=[32,64,64], # for not using same decoder feat for sdf and nerf uncomment this 
            num_ch_enc = self.backbone2d.num_ch_enc[:3],
            use_alpha=False,
            num_output_channels=4,
            scales=range(2),
            use_skips=True,
            # DepthDecoder params (ignored in BatchDecoder impl)
            embedder=self.embedder,
            embedder_out_dim=out_dim,
        )
        
        # for fusing to global volume
        self.fuse_to_global = GRUFusion(cfg.MODEL, direct_substitute=True)
        
        H_tgt, W_tgt = 480, 640
        self.homography_sampler_list = \
            [HomographySample(H_tgt, W_tgt, device=torch.device("cuda")),
             HomographySample(int(H_tgt / 2), int(W_tgt / 2), device=torch.device("cuda")),
             HomographySample(int(H_tgt / 4), int(W_tgt / 4), device=torch.device("cuda"))]
        self.upsample_list = \
            [nn.Identity(),
             nn.Upsample(size=(int(H_tgt / 2), int(W_tgt / 2))),
             nn.Upsample(size=(int(H_tgt / 4), int(W_tgt / 4)))]
        
        self.ssim = SSIM(size_average=True).cuda()
        self.lpips = lpips.LPIPS(net="vgg").cuda()
        self.lpips.requires_grad = False
        
        self.scene_name = None
        self.extrinsics_last = None 
        self.proj_matrices_last = None 
        self.imgs_last = None 
        self.segments_last = None

    def normalizer(self, x):
        """ Normalizes the RGB images to the input range"""
        #return (x - self.pixel_mean.type_as(x)) / self.pixel_std.type_as(x)
        return x/255.0
    '''
    def compute_scale_factor(self, src_depth_syn, src_depth):
        # 1. calibrate the scale between the src image/depth and our synthesized image/depth
        
        scale_factor = torch.exp(torch.mean(
            torch.log(disparity_syn_pt3dsrc) - torch.log(pt3d_disp_src),
            dim=2, keepdim=False)).squeeze(1)  # B
        #scale_factor = torch.mean(src_depth)/torch.mean(src_depth_syn)
        return scale_factor'''
    
    def compute_scale_factor(self, disparity_syn_pt3dsrc, pt3d_disp_src):
        # 1. calibrate the scale between the src image/depth and our synthesized image/depth
        scale_factor = torch.exp(torch.mean(
            torch.log(disparity_syn_pt3dsrc) - torch.log(pt3d_disp_src),
            dim=2, keepdim=False)).squeeze(1)  # B
        return scale_factor
    
    def render_novel_view(self, mpi_all_rgb_src, mpi_all_sigma_src,
                          disparity_all_src, G_tgt_src,
                          K_src_inv, K_tgt, scale=0, scale_factor=None):
        # Apply scale factor
        if scale_factor is not None:
            with torch.no_grad():
                G_tgt_src = torch.clone(G_tgt_src)
                G_tgt_src[:, 0:3, 3] = G_tgt_src[:, 0:3, 3] / scale_factor.view(-1, 1)

        xyz_src_BS3HW = mpi_rendering.get_src_xyz_from_plane_disparity(
            self.homography_sampler_list[scale].meshgrid,
            disparity_all_src,
            K_src_inv
        )

        xyz_tgt_BS3HW = mpi_rendering.get_tgt_xyz_from_plane_disparity(
            xyz_src_BS3HW,
            G_tgt_src
        )

        # Bx1xHxW, Bx3xHxW, Bx1xHxW
        tgt_imgs_syn, tgt_depth_syn, tgt_mask_syn = mpi_rendering.render_tgt_rgb_depth(
            self.homography_sampler_list[scale],
            mpi_all_rgb_src,
            mpi_all_sigma_src,
            disparity_all_src,
            xyz_tgt_BS3HW,
            G_tgt_src,
            K_src_inv,
            K_tgt,
            use_alpha= False,
            is_bg_depth_inf= True
        )
        #print("tgt_depth_syn", torch.unique(tgt_depth_syn))
        tgt_disparity_syn = torch.reciprocal(tgt_depth_syn)

        return {
            "tgt_imgs_syn": tgt_imgs_syn,
            "tgt_disparity_syn": tgt_disparity_syn,
            #"tgt_depth_syn": tgt_depth_syn,
            "tgt_mask_syn": tgt_mask_syn
        }
    
    def loss_fcn_per_scale(self, inputs, scale,
                           mpi_all_src, disparity_all_src,
                           src_depth, tgt1_depth, src_img_xy, tgt1_img_xy, nerf_src_idx, tgt1_idx,
                           scale_factor=None,
                           is_val=False):
        
        src_imgs = inputs['imgs'][:, nerf_src_idx, :, :, :]
        tgt1_imgs = inputs['imgs'][:, tgt1_idx[scale], :, :, :]
        #tgt2_imgs = inputs['imgs'][:, tgt2_idx[scale], :, :, :]
        src_imgs = self.normalizer(src_imgs)
        tgt1_imgs = self.normalizer(tgt1_imgs)
        #tgt2_imgs = self.normalizer(tgt2_imgs)
        src_depth = src_depth[scale] #depth_cam_dict[scale][max_index]
        src_pxpy = src_img_xy[scale] #im_xy_dict[scale][max_index]
        tgt1_depth = tgt1_depth[scale] #depth_cam_dict[scale][second_max_index]
        #tgt2_depth = tgt2_depth[scale] #depth_cam_dict[scale][second_max_index]
        tgt1_pxpy = tgt1_img_xy[scale] #im_xy_dict[scale][second_max_index]
        #tgt2_pxpy = tgt2_img_xy[scale] #im_xy_dict[scale][second_max_index]
        src_disp = torch.reciprocal(src_depth)
        tgt1_disp = torch.reciprocal(tgt1_depth)
        #tgt2_disp = torch.reciprocal(tgt2_depth)
        
        src_extrinsics = inputs['extrinsics'][:, nerf_src_idx, :, :]
        tgt1_extrinsics = inputs['extrinsics'][:, tgt1_idx[scale], :, :]
        #tgt2_extrinsics = inputs['extrinsics'][:, tgt2_idx[scale], :, :]
        G_tgt1_src = torch.inverse(tgt1_extrinsics) @ src_extrinsics # transformation from source to target view
        #G_tgt2_src = torch.inverse(tgt2_extrinsics) @ src_extrinsics # transformation from source to target view
            

        
        src_imgs_scaled = self.upsample_list[scale](src_imgs)
        tgt1_imgs_scaled = self.upsample_list[scale](tgt1_imgs)
        #tgt2_imgs_scaled = self.upsample_list[scale](tgt2_imgs)
        B, _, H_img_scaled, W_img_scaled = src_imgs_scaled.size()
        
        K_src = inputs['intrinsics'][:, nerf_src_idx, :, :]
        K_tgt = inputs['intrinsics'][:, tgt1_idx[scale], :, :]

        K_src_scaled = K_src / (2 ** scale)
        K_src_scaled[:, 2, 2] = 1
        K_tgt_scaled = K_tgt / (2 ** scale)
        K_tgt_scaled[:, 2, 2] = 1
        # TODO: sometimes it returns identity, unless there is CUDA_LAUNCH_BLOCKING=1
        torch.cuda.synchronize()
        K_src_scaled_inv = torch.inverse(K_src_scaled)

        # compute xyz for src and tgt
        # here we need to ensure mpi resolution == image resolution
        assert mpi_all_src.size(3) == H_img_scaled, mpi_all_src.size(4) == W_img_scaled
        xyz_src_BS3HW = mpi_rendering.get_src_xyz_from_plane_disparity(
            self.homography_sampler_list[scale].meshgrid,
            disparity_all_src,
            K_src_scaled_inv
        )

        # compose depth_src
        # here is blend_weights means how much this plane is visible from the camera, BxSx1xHxW
        # e.g, blend_weights = 0 means it is invisible from the camera
        mpi_all_rgb_src = mpi_all_src[:, :, 0:3, :, :]  # BxSx3xHxW
        mpi_all_sigma_src = mpi_all_src[:, :, 3:, :, :]  # BxSx1xHxW
        src_imgs_syn, src_depth_syn, blend_weights, weights = mpi_rendering.render(
            mpi_all_rgb_src,
            mpi_all_sigma_src,
            xyz_src_BS3HW,
            use_alpha=False,
            is_bg_depth_inf=False
        )
        src_rgb_blending = True

        '''
        mpi_all_rgb_src = mpi_all_rgb_src.reshape(mpi_all_rgb_src.shape[0], mpi_all_rgb_src.shape[1], mpi_all_rgb_src.shape[2], -1)
        mpi_all_sigma_src = mpi_all_sigma_src.reshape(mpi_all_sigma_src.shape[0], mpi_all_sigma_src.shape[1], mpi_all_sigma_src.shape[2], -1)
        xyz_src_BS3HW = xyz_src_BS3HW.reshape(xyz_src_BS3HW.shape[0], xyz_src_BS3HW.shape[1], xyz_src_BS3HW.shape[2], -1)
        print("mpi_all_rgb_src", mpi_all_rgb_src.shape)
        print("mpi_all_sigma_src", mpi_all_sigma_src.shape)
        print("xyz_src_BS3HW", xyz_src_BS3HW.shape)
        src_imgs_syn, src_depth_syn, blend_weights, weights = mpi_rendering.render_haha(
            mpi_all_rgb_src,
            mpi_all_sigma_src,
            xyz_src_BS3HW,
            use_alpha=False,
            is_bg_depth_inf=False
        )
        
        print("src_imgs_syn", src_imgs_syn.shape)
        print("src_depth_syn", src_depth_syn.shape)
        print("blend_weights", blend_weights.shape)
        print("weights", weights.shape)'''

        if src_rgb_blending == True:
            #print("blend_weights", blend_weights)
            mpi_all_rgb_src = blend_weights * src_imgs_scaled.unsqueeze(1) + (1 - blend_weights) * mpi_all_rgb_src
            src_imgs_syn, src_depth_syn = mpi_rendering.weighted_sum_mpi(
                mpi_all_rgb_src,
                xyz_src_BS3HW,
                weights,
                is_bg_depth_inf=False
            )
        #print("src_depth_syn", torch.unique(src_depth_syn), src_depth_syn.shape)
        src_disp_syn = torch.reciprocal(src_depth_syn) # Bx1xN_pt
        #print("src_disp_syn", src_disp_syn.shape)

        # compute scale factor
        #src_pt3d_disp = torch.reciprocal(self.pt3d_src[:, 2:, :])  # Bx1xN_pt
        #src_pt3d_pxpy = torch.matmul(K_src_scaled, self.pt3d_src)  # Bx3x3 * Bx3xN_pt -> Bx3xN_pt
        #src_pt3d_pxpy = src_pt3d_pxpy[:, 0:2, :] / src_pt3d_pxpy[:, 2:, :]  # Bx2xN_pt
        #print("src_pxpy", src_pxpy.shape)
        #src_depth_syn_original = src_depth_syn.clone()
        src_disp_syn_original = src_disp_syn.clone()
        #src_depth_syn = rendering_utils.gather_pixel_by_pxpy(src_depth_syn, src_pxpy.T.unsqueeze(0))  # Bx1xN_pt
        src_disp_syn = rendering_utils.gather_pixel_by_pxpy(src_disp_syn, src_pxpy.T.unsqueeze(0))  # Bx1xN_pt
        
        if scale_factor is None:
            #scale_factor = self.compute_scale_factor(src_depth_syn, src_depth)  # B
            scale_factor = self.compute_scale_factor(src_disp_syn, src_disp) 
        #print('scale_factor', scale_factor)
        
        # Render target view
        render_results1 = self.render_novel_view(mpi_all_rgb_src, mpi_all_sigma_src,
                                                disparity_all_src, G_tgt1_src,
                                                K_src_scaled_inv, K_tgt_scaled,
                                                scale=scale,
                                                scale_factor=scale_factor)
        tgt1_imgs_syn = render_results1["tgt_imgs_syn"]
        tgt1_disp_syn = render_results1["tgt_disparity_syn"]
        #tgt_depth_syn = render_results["tgt_depth_syn"]
        tgt1_mask_syn = render_results1["tgt_mask_syn"]
        '''
        render_results2 = self.render_novel_view(mpi_all_rgb_src, mpi_all_sigma_src,
                                                disparity_all_src, G_tgt2_src,
                                                K_src_scaled_inv, K_tgt_scaled,
                                                scale=scale,
                                                scale_factor=scale_factor)'''
        #tgt2_imgs_syn = render_results2["tgt_imgs_syn"]
        #tgt2_disp_syn = render_results2["tgt_disparity_syn"]
        #tgt_depth_syn = render_results["tgt_depth_syn"]
        #tgt2_mask_syn = render_results2["tgt_mask_syn"]

        # build loss
        # Read lambdas
        smoothness_lambda_v1 = 0.5
        smoothness_lambda_v2 = 1.0

        #with torch.no_grad():
        smoothness_gmin = 0.8
        smoothness_grad_ratio = 0.1
        loss_rgb_src = torch.mean(torch.abs(src_imgs_syn - src_imgs_scaled))
        loss_ssim_src = 1 - self.ssim(src_imgs_syn, src_imgs_scaled)
        '''
        loss_smooth_src = edge_aware_loss(src_imgs_scaled, src_disp_syn_original,
                                              gmin=smoothness_gmin,
                                              grad_ratio=smoothness_grad_ratio)'''

        # 1. disparity at src frame
        # compute pixel coordinates of gt points
        src_disp_syn_scaled = src_disp_syn/scale_factor.view(B, 1, 1)
        loss_disp_src = torch.mean(torch.abs(torch.log(src_disp_syn_scaled) - torch.log(src_disp)))
        #loss_depth_src = torch.mean(torch.abs(src_depth_syn_scaled - src_depth))

        # disparity at tgt frame
        #tgt_pt3d_disp = torch.reciprocal(self.pt3d_tgt[:, 2:, :])  # Bx1xN_pt
        #tgt_pt3d_pxpy = torch.matmul(K_tgt_scaled, self.pt3d_tgt)  # Bx3x3 * Bx3xN_pt -> Bx3xN_pt
        #tgt_pt3d_pxpy = tgt_pt3d_pxpy[:, 0:2, :] / tgt_pt3d_pxpy[:, 2:, :]  # Bx2xN_pt
        #tgt_pt3d_disp_syn = rendering_utils.gather_pixel_by_pxpy(tgt_disparity_syn, tgt_pt3d_pxpy)  # Bx1xN_pt
        #tgt_depth_syn_original = tgt_depth_syn.clone()
        #tgt_depth_syn = rendering_utils.gather_pixel_by_pxpy(tgt_depth_syn, tgt_pxpy.T.unsqueeze(0))  # Bx1xN_pt
        tgt1_disp_syn_original = tgt1_disp_syn.clone()
        tgt1_disp_syn = rendering_utils.gather_pixel_by_pxpy(tgt1_disp_syn, tgt1_pxpy.T.unsqueeze(0))  # Bx1xN_pt
        #tgt_pt3d_disp_syn_scaled = tgt_pt3d_disp_syn / scale_factor.view(B, 1, 1)
        tgt1_disp_syn_scaled = tgt1_disp_syn/scale_factor.view(B, 1, 1)
        #tgt2_disp_syn_original = tgt2_disp_syn.clone()
        #tgt2_disp_syn = rendering_utils.gather_pixel_by_pxpy(tgt2_disp_syn, tgt2_pxpy.T.unsqueeze(0))  # Bx1xN_pt
        #tgt2_disp_syn_scaled = tgt2_disp_syn/scale_factor.view(B, 1, 1)
        '''
        loss_disp_pt3dtgt = disp_lambda * torch.mean(torch.abs(
            torch.log(tgt_pt3d_disp_syn_scaled) - torch.log(tgt_pt3d_disp)
        ))'''
        #loss_depth_tgt = torch.mean(torch.abs(tgt_depth_syn_scaled - tgt_depth))
        loss_disp_tgt1 = torch.mean(torch.abs(torch.log(tgt1_disp_syn_scaled) - torch.log(tgt1_disp)))
        #loss_disp_tgt2 = torch.mean(torch.abs(torch.log(tgt2_disp_syn_scaled) - torch.log(tgt2_disp)))

        # 2. rgb loss at tgt frame
        # some pixels in tgt frame is outside src FoV, here we can detect and ignore those pixels
        rgb_tgt1_valid_mask = torch.ge(tgt1_mask_syn, self.cfg.valid_mask_threshold).to(torch.float32)
        loss_map1 = torch.abs(tgt1_imgs_syn - tgt1_imgs_scaled) * rgb_tgt1_valid_mask
        loss_rgb_tgt1 = loss_map1.mean()
        #rgb_tgt2_valid_mask = torch.ge(tgt2_mask_syn, self.cfg.valid_mask_threshold).to(torch.float32)
        #loss_map2 = torch.abs(tgt2_imgs_syn - tgt2_imgs_scaled) * rgb_tgt2_valid_mask
        #loss_rgb_tgt2 = loss_map2.mean()

        # Edge aware smoothless losses
        loss_smooth_tgt1 = smoothness_lambda_v1 * edge_aware_loss(
            tgt1_imgs_scaled,
            tgt1_disp_syn_original,
            gmin=smoothness_gmin,
            grad_ratio=smoothness_grad_ratio)
        '''
        loss_smooth_tgt2 = smoothness_lambda_v1 * edge_aware_loss(
            tgt2_imgs_scaled,
            tgt2_disp_syn_original,
            gmin=smoothness_gmin,
            grad_ratio=smoothness_grad_ratio)'''
        loss_smooth_tgt1_v2 = smoothness_lambda_v2 * edge_aware_loss_v2(tgt1_imgs_scaled, tgt1_disp_syn_original)
        #loss_smooth_tgt2_v2 = smoothness_lambda_v2 * edge_aware_loss_v2(tgt2_imgs_scaled, tgt2_disp_syn_original)
        loss_smooth_src_v2 = smoothness_lambda_v2 * edge_aware_loss_v2(src_imgs_scaled, src_disp_syn_original)
        loss_ssim_tgt1 = 1 - self.ssim(tgt1_imgs_syn, tgt1_imgs_scaled)
        #loss_ssim_tgt2 = 1 - self.ssim(tgt2_imgs_syn, tgt2_imgs_scaled)


        # LPIPS and PSNR loss (for eval only):
        with torch.no_grad():
            lpips_tgt1 = self.lpips(tgt1_imgs_syn, tgt1_imgs_scaled).mean() #if (is_val and scale == 0) else torch.tensor(0.0)
            psnr_tgt1 = psnr(tgt1_imgs_syn, tgt1_imgs_scaled).mean()
            #lpips_tgt2 = self.lpips(tgt2_imgs_syn, tgt2_imgs_scaled).mean() #if (is_val and scale == 0) else torch.tensor(0.0)
            #psnr_tgt2 = psnr(tgt2_imgs_syn, tgt2_imgs_scaled).mean()
            
            lpips_src = self.lpips(src_imgs_syn, src_imgs_scaled).mean() #if (is_val and scale == 0) else torch.tensor(0.0)
            psnr_src = psnr(src_imgs_syn, src_imgs_scaled).mean()

         
        '''
        loss = loss_depth_tgt + loss_depth_src \
            + loss_rgb_tgt + loss_ssim_tgt \
            + loss_smooth_tgt \
            + loss_smooth_src_v2 + loss_smooth_tgt_v2'''
        
        loss = loss_disp_tgt1 + loss_disp_src \
            + loss_rgb_tgt1 + loss_ssim_tgt1 + loss_rgb_src + loss_ssim_src\
            + loss_smooth_tgt1 \
            + loss_smooth_src_v2 + loss_smooth_tgt1_v2
        '''
        loss = loss_disp_tgt1 + loss_disp_src \
            + loss_rgb_tgt1 + loss_rgb_src'''
        '''
        loss = loss_disp_tgt + loss_rgb_tgt + loss_ssim_tgt + loss_disp_src + loss_rgb_src + loss_ssim_src + loss_smooth_src_v2 + loss_smooth_tgt_v2'''

        '''
        loss_smooth_tgt = (loss_smooth_tgt1+loss_smooth_tgt2)/2
        loss_rgb_tgt = (loss_rgb_tgt1+loss_rgb_tgt2)/2
        psnr_tgt = (psnr_tgt1+psnr_tgt2)/2
        lpips_tgt = (lpips_tgt1+lpips_tgt2)/2
        loss_smooth_tgt_v2 = (loss_smooth_tgt1_v2+loss_smooth_tgt2_v2)/2
        loss_ssim_tgt = (loss_ssim_tgt1+loss_ssim_tgt2)/2
        loss_disp_tgt = (loss_disp_tgt1+loss_disp_tgt2)/2'''

        loss_dict = {"loss": loss,
                     "loss_rgb_src": loss_rgb_src,
                     "loss_ssim_src": loss_ssim_src,
                     "loss_depth_src": loss_disp_src,
                     #"loss_smooth_src": torch.tensor(0), #loss_smooth_src,
                     "loss_smooth_tgt": loss_smooth_tgt1,
                     "loss_smooth_src_v2": loss_smooth_src_v2,
                     "loss_smooth_tgt_v2": loss_smooth_tgt1_v2,
                     "loss_rgb_tgt": loss_rgb_tgt1,
                     "loss_ssim_tgt": loss_ssim_tgt1,
                     "lpips_tgt": lpips_tgt1,
                     "psnr_tgt": psnr_tgt1,
                     "lpips_src": lpips_src,
                     "psnr_src":  psnr_src,
                     "loss_depth_tgt": loss_disp_tgt1}
        
        '''
        visualization_dict = {"src_disparity_syn": src_disparity_syn,
                              "tgt_disparity_syn": tgt_disparity_syn,
                              "tgt_imgs_syn": tgt_imgs_syn,
                              "tgt_mask_syn": tgt_mask_syn,
                              "src_imgs_syn": src_imgs_syn}'''

        return loss_dict, scale_factor

    def forward(self, inputs, save_mesh=False):

        inputs = tocuda(inputs)
        outputs = {}
        imgs = torch.unbind(inputs['imgs'], 1)

        # image feature extraction
        # in: images; out: feature maps
        features = [self.backbone2d(img) for img in imgs]
        features_sdf = [feat[-3:] for feat in features]
        features_nerf = [feat[:-3] for feat in features]
        
        scene = inputs['scene'][0]
        if self.scene_name is None or scene != self.scene_name:
            self.scene_name = scene
            self.extrinsics_last = inputs['extrinsics']
            self.proj_matrices_last = inputs['proj_matrices']
            self.imgs_last = inputs['imgs']
            if self.training:
                self.segments_last = inputs['segments']
        
        # coarse-to-fine decoder: SparseConv and GRU Fusion.
        # in: image feature; out: sparse coords and tsdf
        outputs, rgb_loss, planar_loss, gt_loss, total_loss, src_depth, tgt1_depth, src_img_xy, tgt1_img_xy, src_idx, tgt1_idx = self.neucon_net(features_sdf, inputs, outputs, self.extrinsics_last, self.proj_matrices_last, self.imgs_last, self.segments_last)
        if rgb_loss is None:
            return None, None, None, None, None, None
        
        # NeRF, uncomment if train with nerf
                
        '''
        #print("inputs['imgs']", inputs['imgs'].shape)
        B = inputs['imgs'].shape[0]
        disparity_coarse_src = _get_disparity_list(self.cfg, B, device=inputs['imgs'].device)
        #print("disparity_coarse_src", disparity_coarse_src.shape, disparity_coarse_src)
        #stop
        nerf_src_idx = src_idx[0]
        outputs_nerf = self.nerf(features_nerf[nerf_src_idx], disparity_coarse_src)
        #output_nerf_list = [outputs_nerf[("disp", 0)], outputs_nerf[("disp", 1)], outputs_nerf[("disp", 2)], outputs_nerf[("disp", 3)]]
        #output_nerf_list = [outputs_nerf[("disp", 0)], outputs_nerf[("disp", 1)]]
        output_nerf_list = [outputs_nerf[("disp", 0)]]
        
        scale_factor = None
        scale_list = [0] #[0,1,2] #list(range(3))
        nerf_loss_dict_list = []
        for scale in scale_list:
            nerf_loss_dict_tmp, scale_factor = self.loss_fcn_per_scale(
                inputs,
                scale,
                output_nerf_list[scale],
                disparity_coarse_src,
                src_depth, tgt1_depth, src_img_xy, tgt1_img_xy, nerf_src_idx, tgt1_idx, 
                scale_factor,
                is_val= not self.training
            )
            nerf_loss_dict_list.append(nerf_loss_dict_tmp)
            #visualization_dict_list.append(visualization_dict_tmp)
        nerf_loss_dict = nerf_loss_dict_list[0]
        
        for scale in scale_list[1:]:
            #if self.config.get("training.use_multi_scale", True):
            
            #nerf_loss_dict["loss"] += (nerf_loss_dict_list[scale]["loss_rgb_tgt"] + nerf_loss_dict_list[scale]["loss_ssim_tgt"])
            #nerf_loss_dict["loss"] += (nerf_loss_dict_list[scale]["loss_depth_src"] + nerf_loss_dict_list[scale]["loss_depth_tgt"])
            #nerf_loss_dict["loss"] += (nerf_loss_dict_list[scale]["loss_smooth_src_v2"] + nerf_loss_dict_list[scale]["loss_smooth_tgt_v2"])
            nerf_loss_dict["loss"] += nerf_loss_dict_list[scale]["loss"]
        total_loss += nerf_loss_dict["loss"]
        #total_loss = nerf_loss_dict["loss"]'''
        
        # comment this if traininig with NeRF
        nerf_loss_dict = {"loss": torch.tensor(0),
                     "loss_rgb_src": torch.tensor(0),
                     "loss_ssim_src": torch.tensor(0),
                     "loss_depth_src": torch.tensor(0),
                     "loss_smooth_src": torch.tensor(0),
                     "loss_smooth_tgt": torch.tensor(0),
                     "loss_smooth_src_v2": torch.tensor(0),
                     "loss_smooth_tgt_v2": torch.tensor(0),
                     "loss_rgb_tgt": torch.tensor(0),
                     "loss_ssim_tgt": torch.tensor(0),
                     "lpips_tgt": torch.tensor(0),
                     "psnr_tgt": torch.tensor(0),
                     "lpips_src": torch.tensor(0),
                     "psnr_src": torch.tensor(0),
                     "loss_depth_tgt": torch.tensor(0)}
        
        
        # fuse to global volume.
        if not self.training and 'coords' in outputs.keys():
            outputs, mask = self.fuse_to_global(outputs['coords'], outputs['tsdf'], inputs, self.n_scales, outputs, save_mesh)
        #stop

       
        return outputs, rgb_loss, planar_loss, gt_loss, total_loss, nerf_loss_dict
