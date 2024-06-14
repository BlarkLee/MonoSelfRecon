from .base import RadianceNet #SDFNet, SemanticNet
from .ray_sampler import sdf_to_sigma, fine_sample

import copy
import functools
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sdfnerfnet_utils import batchify_query
from .vox2ray_sparse_coarse import vox2ray
from .ray_sample_coarse import ray_sample_coarse

class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.speed_factor = cfg.MODEL.speed_factor
        ln_beta_init = np.log(cfg.MODEL.beta_init) / self.speed_factor
        self.ln_beta = nn.Parameter(data=torch.Tensor([ln_beta_init]), requires_grad=True)

        #self.sdf_net = SDFNet(cfg)
        self.radiance_net = RadianceNet(cfg)
        #self.semantic_net = SemanticNet(cfg)
    
    def forward_ab(self):
        beta = torch.exp(self.ln_beta * self.speed_factor)
        return 1./beta, beta

    def forward_surface(self, x: torch.Tensor):
        sdf = self.sdf_net.forward(x)
        return sdf        

    def forward_surface_with_nablas(self, x: torch.Tensor):
        sdf, nablas, h = self.sdf_net.forward_with_nablas(x)
        return sdf, nablas, h

    def set_feat(self, feat: torch.Tensor):
        self.geometry_feature = feat
        
    def forward(self, x:torch. Tensor, view_dirs: torch.Tensor):
        #sdf, nablas, geometry_feature = self.forward_surface_with_nablas(x)
        radiances = self.radiance_net.forward(x, view_dirs, self.geometry_feature)
        #semantics = self.semantic_net.forward(x, geometry_feature)
        return radiances#, semantics, sdf, nablas
    
    def forward_semantic(self, x:torch. Tensor):
        sdf, nablas, geometry_feature = self.forward_surface_with_nablas(x)
        semantics = self.semantic_net.forward(x, geometry_feature)
        return semantics


def volume_render(
    valid_masked_pts,
    valid_ray_pix,
    rays_o, 
    rays_d,
    model: MLP,
    cfg,
    inputs,
    outputs,
    view_idx,
    near=0.0,
    far=2.0,
    perturb = True
    ):

    device = rays_o.device
    rayschunk = cfg.sample.rayschunk
    netchunk = cfg.sample.netchunk
    N_samples = cfg.sample.N_samples
    N_importance = cfg.sample.N_importance
    max_upsample_steps = cfg.sample.max_upsample_steps
    max_bisection_steps = cfg.sample.max_bisection_steps
    epsilon = cfg.sample.epsilon

    DIM_BATCHIFY = 1
    B = rays_d.shape[0]  # batch_size
    flat_vec_shape = [B, -1, 3]

    rays_o = torch.reshape(rays_o, flat_vec_shape).float()
    rays_d = torch.reshape(rays_d, flat_vec_shape).float()

    depth_ratio = rays_d.norm(dim=-1)
    rays_d = F.normalize(rays_d, dim=-1)
    
    batchify_query_ = functools.partial(batchify_query, chunk=netchunk, dim_batchify=DIM_BATCHIFY)

    def render_rayschunk(valid_ray_pix, rays_o: torch.Tensor, rays_d: torch.Tensor, inputs, outputs, view_idx):
        
        
        # rays_o : (1,1024,3), 3 vector of translation of the extrinsics, same for 1024 pixels cuz from the same frame
        # rays_d: (1024, 3), xyz of the corresponding pixel in world coordinate
        view_dirs = rays_d
        #print("view_dirs", view_dirs)
        #print("view_dirs", view_dirs.shape)
        
        prefix_batch = [B]
        N_rays = rays_o.shape[-2]
        #print("rays_o.shape", rays_o.shape)
        #print("rays_d.shape", rays_d.shape)
        
        #print("prefix_batch", prefix_batch)
        #print("device", device)
        #print("near", near)
        #nears = near * torch.ones([*prefix_batch, N_rays, 1]).to(device)
        #nears = near.unsqueeze(1).unsqueeze(2)
        #print("nears", nears.shape)
        #stop
        #print("far", far)
        #fars = far * torch.ones([*prefix_batch, N_rays, 1]).to(device)
        #fars = far.unsqueeze(1).unsqueeze(2)
        #print("fars", fars.shape)
        #print("fars", fars)
        
        #print("rays_o", rays_o.shape)
        #print("rays_o", rays_o)
        #print("valid_masked_pts", valid_masked_pts.shape)
        #print("rays_d", rays_d.shape)
        #print("valid_masked_pts", valid_masked_pts)
        #print("valid_ray_pix", valid_ray_pix.shape)
        #stop
        #print("rays_d", rays_d)
        t_center = (valid_masked_pts[:,-1]-rays_o.squeeze(1)[:, -1])/rays_d.squeeze(1)[:, -1]
        #print("t_center", t_center.shape)
        #print("t_center_range", torch.unique(t_center))
        #stop
        nears = (t_center - 0.1*0.04).unsqueeze(1).unsqueeze(1)
        fars = (t_center + 0.1*0.04).unsqueeze(1).unsqueeze(1)
        #print("nears", nears.shape)
        
        

        _t = torch.linspace(0, 1, N_samples).float().to(device)
        d_coarse = nears * (1 - _t) + fars * _t
        #d_coarse = t_center.unsqueeze(1).unsqueeze(1) * (1 - _t) + t_center.unsqueeze(1).unsqueeze(1) * _t
        #print("d_coarse", d_coarse)
        #print("d_coarse", d_coarse.shape)
        #print(stop)
        
        
        
        
        alpha, beta = model.forward_ab()
        '''
        with torch.no_grad():
            _t = torch.linspace(0, 1, N_samples*4).float().to(device)
            d_init = nears * (1 - _t) + fars * _t
            
            d_fine, beta_map, iter_usage = fine_sample(
                model.forward_surface, d_init, rays_o, rays_d, 
                alpha_net=alpha, beta_net=beta, far=fars, 
                eps=epsilon, max_iter=max_upsample_steps, max_bisection=max_bisection_steps, 
                final_N_importance=N_importance, perturb=perturb, 
                N_up=N_samples*4
            )
        print("d_fine", d_fine)
        print("d_fine", d_fine.shape)
        d_all = torch.cat([d_coarse, d_fine], dim=-1)
        '''
        
        d_all = d_coarse
        #d_all, _ = torch.sort(d_all, dim=-1)
        #print("d_all", d_all.shape)
        #print("d_all", d_all)
        #print("rays_o[..., None, :]", rays_o[..., None, :].shape)
        #print("rays_o", rays_o)
        #print("rays_d", rays_d)
        #print("rays_o", rays_o.shape)
        #print("rays_d", rays_d.shape)
        #print("d_all", d_all)
        #print("rays_0_grad", rays_o.requires_grad)
        #print("rays_d_grad", rays_d.requires_grad)
        #print("d_all_grad", d_all.requires_grad)
        pts = (rays_o[..., None, :] + rays_d[..., None, :] * d_all[..., :, None]).squeeze(1)
        #print("pts", pts.shape)
        #print("pts", pts)
        pts_64 = pts[:, 64, :]
        #print("pts[64]", pts_64)
        dist = (valid_masked_pts - pts_64).pow(2).sum(1).sqrt()
        #print("dist", dist.shape)
        #print("dist", dist)
        #print('dist_max', dist.max())
        dist_mask = dist<0.004
        #print("dist_mask", dist_mask.shape)
        pts = pts[dist_mask]
        #print("pts", pts.shape)
        
        ids = np.random.choice(pts.shape[0], 1024, replace=False)
        pts = pts[ids]
        valid_ray_pix = torch.round(valid_ray_pix[dist_mask][ids]).type(torch.long)
        #valid_ray_pix = valid_ray_pix[dist_mask][ids]
        #print("pts", pts.shape)
        #print("valid_ray_pix", valid_ray_pix.shape)
        #stop
        
        
        
        #stop
        #dist = dist[dist<0.04]
        #print("dist_over", dist_over.shape)
        #stop
        #print("d_all[..., :, None]", d_all[..., :, None].shape)
        #print(stop)
        #print("inputs['world'][0]", inputs['world'][0])
        #stop
        sdf, ray_feat = vox2ray(pts, inputs['vol_origin_partial'], outputs['coords'][:, 1:], outputs['sdf'], outputs['feat'])
        #pts = pts.reshape(1, -1, 3)
        #print("pts", pts.shape)
        #print("sdf", sdf.shape)
        
        '''
        has_grad = True
        print("sdf_grad", sdf.requires_grad)
        print("pts_grad", pts.requires_grad)
        with torch.enable_grad():
            pts = pts.requires_grad_(True)
            nabla = torch.autograd.grad(
                sdf,
                pts,
                torch.ones_like(sdf, device=pts.device),
                create_graph=has_grad,
                retain_graph=has_grad,
                only_inputs=True
            )[0]
        print("nabla", nabla)
        print("nabla", nabla.shape)
        stop'''
        sigma = sdf_to_sigma(sdf, alpha, beta)
        d_all = d_all[dist_mask][ids].squeeze(1).unsqueeze(0)
        #print("d_all", d_all.shape)
        delta_i = d_all[..., 1:] - d_all[..., :-1]
        p_i = torch.exp(-F.relu_(sigma[..., :-1] * delta_i))
        #print("p_i", p_i.shape)

        tau_i = (1 - p_i + 1e-10) * (
            torch.cumprod(
                torch.cat(
                    [torch.ones([*p_i.shape[:-1], 1], device=device), p_i], dim=-1), 
                dim=-1)[..., :-1]
            )
        #print("ray_feat", ray_feat.shape)
        #stop
        #print("ray_feat.reshape(1, -1, ray_feat.shape[-1])", ray_feat.reshape(1, -1, ray_feat.shape[-1]).shape)
        model.set_feat(ray_feat.reshape(1, -1, ray_feat.shape[-1]))
        #tau_i = tau_i[dist_mask][ids]
        view_dirs = view_dirs[dist_mask][ids].squeeze(1).unsqueeze(0)
        pts = pts.unsqueeze(0)
        #print("pts", pts.shape)
        #print("view_dirs", view_dirs.shape)
        radiances = batchify_query_(model.forward, pts, view_dirs.unsqueeze(-2).expand_as(pts))
        #print("radiances", radiances.shape)
        #print("tau_i", tau_i.shape)
        #print("tau_i", np.unique(tau_i.detach().cpu().numpy()))
        rgb_map = torch.sum(tau_i[..., None] * radiances[..., :-1, :], dim=-2)
        #print("rgb_map", rgb_map.shape)
        #print("valid_ray_pix", valid_ray_pix.shape)
        #stop
        #print("rgb_map", np.unique(rgb_map.detach().cpu().numpy()))
        #semantic_map = torch.sum(tau_i[..., None] * semantics[..., :-1, :], dim=-2)
        
        #distance_map = torch.sum(tau_i / (tau_i.sum(-1, keepdim=True)+1e-10) * d_all[..., :-1], dim=-1)
        #depth_map = distance_map / depth_ratio
        #acc_map = torch.sum(tau_i, -1)
        #print("rgb_map", rgb_map.shape)
        #print("radiances", radiances.shape)
        #print("depth_map", depth_map.shape)
        #print("semantic_map", semantic_map.shape)
        
        ret_i = OrderedDict([
            ('rgb', rgb_map),
            #('semantic', semantic_map),
            #('distance', distance_map),
            #('depth', depth_map),
            #('mask_volume', acc_map)
        ])
        '''
        surface_points = rays_o + rays_d * distance_map[..., None]
        _, surface_normals, _ = model.sdf_net.forward_with_nablas(surface_points.detach())
        ret_i['surface_normals'] = surface_normals'''

        # normals_map = F.normalize(nablas, dim=-1)
        # N_pts = min(tau_i.shape[-1], normals_map.shape[-2])
        # normals_map = (normals_map[..., :N_pts, :] * tau_i[..., :N_pts, None]).sum(dim=-2)
        # ret_i['normals_volume'] = normals_map

        #ret_i['sdf'] = sdf
        #ret_i['nablas'] = nablas
        #ret_i['radiance'] = radiances
        #ret_i['alpha'] = 1.0 - p_i
        #ret_i['p_i'] = p_i
        #ret_i['visibility_weights'] = tau_i
        #ret_i['d_vals'] = d_all
        #ret_i['sigma'] = sigma
        #ret_i['beta_map'] = beta_map
        #ret_i['iter_usage'] = iter_usage
        #print(stop)
        #print("pts", pts.shape)
        #stop
        return ret_i, pts, valid_ray_pix
        
    ret = {}
    
    for i in range(0, rays_o.shape[DIM_BATCHIFY], rayschunk):
        ret_i, pts, valid_ray_pix = render_rayschunk(valid_ray_pix, rays_o[:, i:i+rayschunk], rays_d[:, i:i+rayschunk], inputs, outputs, view_idx)
        for k, v in ret_i.items():
            if k not in ret:
                ret[k] = []
            ret[k].append(v)
    for k, v in ret.items():
        ret[k] = torch.cat(v, DIM_BATCHIFY)
    #stop
    
    alpha, beta = model.forward_ab()
    alpha, beta = alpha.data, beta.data
    ret['scalars'] = {'alpha': alpha, 'beta': beta}
    return ret, pts, valid_ray_pix

def pts2pix(pts, intrinsics, extrinsics, h, w):
    
    proj_mat = torch.inverse(extrinsics)
    #print("intrinsics", intrinsics.shape)
    #print("proj_mat", proj_mat.shape)
    proj_mat[:, :3, :4] = intrinsics @ proj_mat[:, :3, :4]
    #print("proj_mat", proj_mat.shape)
    #stop
    
    n_views = proj_mat.shape[0]
    rs_grid = pts.unsqueeze(0).expand(n_views, -1, -1)
    rs_grid = rs_grid.permute(0, 2, 1).contiguous()
    nV = rs_grid.shape[-1]
    rs_grid = torch.cat([rs_grid, torch.ones([n_views, 1, nV]).cuda()], dim=1)
    #print("rs_grid", rs_grid)
    #print("rs_grid", rs_grid.shape)
    
    # Project grid
    im_p = proj_mat @ rs_grid
    #print("im_p", im_p.shape)
    im_x, im_y, im_z = im_p[:, 0], im_p[:, 1], im_p[:, 2]
    im_x = im_x / im_z
    im_y = im_y / im_z

    #im_grid = torch.stack([2 * im_x / (w - 1) - 1, 2 * im_y / (h - 1) - 1], dim=-1)
    im_grid = torch.stack([im_x, im_y], dim=-1)
    im_grid = im_grid.view(n_views, 1, -1, 2)
    #im_grid = torch.round(im_grid).type(torch.long)
    #print("im_grid", im_grid.shape)
    #print("im_grid", im_grid)
    
    #print("range", torch.unique(im_grid[0,0,:,0]))
    #stop
    return im_grid

def gen_rays(pix, c2w, intrinsic):
    #c2w = c2w.numpy()
    #intrinsic = intrinsic.numpy()
    rays_o = c2w[:3, 3]
    #print("pix", pix.shape)
    #print("torch.ones((pix.shape[0], 1)", torch.ones((pix.shape[0], 1)).shape)
    pix = torch.cat((pix, torch.ones((pix.shape[0], 1)).to(pix.device)), dim=-1)
    #print("pix", pix.shape)
        
    pix = pix @ torch.linalg.inv(intrinsic).T
    #print("pix", pix.shape)
    #print("c2w", c2w.shape)
    #stop
    rays_d = pix @ c2w[:3, :3].T
    #print("pix", pix.shape)
    #rays_d = pix.reshape(-1, 3)
    #print("rays_o", rays_o.shape)
    #print("rays_d", rays_d.shape)
    rays = torch.cat([rays_o[None].repeat(pix.shape[0], 1), rays_d], dim=-1)
    #print("rays", rays.shape)
    #print("rays", rays)
    return rays.type(torch.float32)#astype(np.float32)

class SDFNerfNet(nn.Module):
    def __init__(self, cfg):
        super(SDFNerfNet, self).__init__()
        self.model = MLP(cfg)
        self.cfg = cfg
        
        self.theta = nn.Parameter(torch.Tensor([0.]), requires_grad=True)
        # <cos(theta), sin(tehta), 0> is $\mathbf{n}_w$ in equation (9)
    
    def forward(self, batch, neucon_outputs):
        
        masked_idx = ray_sample_coarse(neucon_outputs)
        masked_pts = masked_idx*0.04 + batch['world'][0][0,:,0]
        #print("masked_pts", masked_pts.shape)
        #print("batch['proj_matrices']", batch['proj_matrices'].shape)
        #print("batch['proj_matrices']", batch['proj_matrices'][0,0,:,:,:].shape)
        #print("batch['proj_matrices']", batch['proj_matrices'][0,0,:,:,:])
        #print("batch['intrinsics']", batch['intrinsics'].shape)
        #print("batch['extrinsics']", batch['extrinsics'].shape)
        #print("batch['imgs'][0]", batch['imgs'][0].shape)
        h, w = batch['imgs'][0].shape[-2], batch['imgs'][0].shape[-1]
        #print(h, w)
        ray_pixs= pts2pix(masked_pts, batch['intrinsics'][0], batch['extrinsics'][0], h, w)
        
        rendered_imgs = []
        ray_pts = []
        gt_rgb_coords = []
        gt_imgs = []
        #print("batch['rays']", batch['rays'][0].shape)
        #stop
        #for view_idx, rays in enumerate(batch['rays']):
        for view_idx, ray_pix in enumerate(ray_pixs):
            #print("view_idx", view_idx)
            ray_pix = ray_pix[0]
            #print("ray_pix", ray_pix.detach().cpu().numpy())
            mask = ((ray_pix[:,0]>0).type(torch.long)*(ray_pix[:, 1]>0).type(torch.long)*(ray_pix[:,0]<h).type(torch.long)*(ray_pix[:, 1]<w).type(torch.long)).type(torch.bool)
            #print("mask", mask.shape)
            valid_ray_pix = ray_pix[mask]
            valid_masked_pts = masked_pts[mask]
            #print("valid_ray_pix", valid_ray_pix.shape)
            #stop
            #print("valid_masked_pts", valid_masked_pts.shape)
            
            if valid_ray_pix.shape[0]<1500:
                continue
            else:
                ids = np.random.choice(valid_ray_pix.shape[0], valid_ray_pix.shape[0], replace=False)
                valid_ray_pix = valid_ray_pix[ids]
                valid_masked_pts = valid_masked_pts[ids]
                #print("valid_ray_pix", valid_ray_pix.shape)
                #print("valid_masked_pts", valid_masked_pts.shape)
                rays = gen_rays(valid_ray_pix, batch['extrinsics'][0][view_idx], batch['intrinsics'][0][view_idx])
                #print("valid_ray_pix", valid_ray_pix.shape)
                #print("valid_masked_pts", valid_masked_pts.shape)
                #print("ray", ray.shape)
                #stop
                
                '''
                ray = self.gen_rays(h, w, items['extrinsics'][i], items['intrinsics'][i])
                ids = np.random.choice(len(ray), 1024, replace=False)
                ray = ray[ids] #(1024, 6)
                
                
                #print("view_idx", view_idx)'''
                rays_o, rays_d = rays[:, :3], rays[:, 3:6]
                rays_d[rays_d.abs() < 1e-6] = 1e-6
                '''
                if self.training:
                    near = self.cfg.TRAIN.near
                    far = self.cfg.TRAIN.far
                    pertube = True
                

                else:
                    near = self.cfg.TEST.near
                    far = self.cfg.TEST.far
                    pertube = False'''
                
                if self.training:
                    pertube = True
                
                else:
                    pertube = False
                
                rendered_rgb, pts, gt_rgb_coord = volume_render(
                    valid_masked_pts,
                    valid_ray_pix,
                    rays_o,
                    rays_d,
                    self.model,
                    self.cfg,
                    batch,
                    neucon_outputs,
                    view_idx,
                    perturb=pertube
                )
                rendered_rgb = rendered_rgb['rgb']
                rendered_imgs.append(rendered_rgb)
                ray_pts.append(pts)
                #print("w", w)
                gt_rgb_idx = w*gt_rgb_coord[:, 0] + gt_rgb_coord[:, 1]
                #print("gt_rgb_idx_range", torch.unique(gt_rgb_idx))
                #print("gt_rgb_coord[:, 1]_min", torch.min(gt_rgb_coord[:, 1]))
                #print("gt_rgb_coord[:, 1]_max", torch.max(gt_rgb_coord[:, 1]))
                #print("gt_rgb_coord[:, 0]_min", torch.min(gt_rgb_coord[:, 0]))
                #print("gt_rgb_coord[:, 0]_max", torch.max(gt_rgb_coord[:, 0]))
                rendered_gt_img = batch['imgs'][0][view_idx].reshape(3, -1).transpose(1, 0)[gt_rgb_idx].unsqueeze(0).type(torch.float32)/255
                #print("rendered_gt_img", rendered_gt_img)
                gt_imgs.append(rendered_gt_img)
                
                
                #print("rendered_gt_img", rendered_gt_img.shape)
                #stop
        return rendered_imgs, ray_pts, gt_imgs
            
