import torch
import torch.nn.functional as F
from torch.autograd import Variable

def mat_3x3_det(mat):
    '''
    calculate the determinant of a 3x3 matrix, support batch.
    '''
    if len(mat.shape) < 3:
        mat = mat[None]
    assert mat.shape[1:] == (3, 3)

    det = mat[:, 0, 0] * (mat[:, 1, 1] * mat[:, 2, 2] - mat[:, 2, 1] * mat[:, 1, 2]) \
        - mat[:, 0, 1] * (mat[:, 1, 0] * mat[:, 2, 2] - mat[:, 1, 2] * mat[:, 2, 0]) \
        + mat[:, 0, 2] * (mat[:, 1, 0] * mat[:, 2, 1] - mat[:, 1, 1] * mat[:, 2, 0])
    return det

def mat_3X3_inv(mat):
    '''
    calculate the inverse of a 3x3 matrix, support batch.
    :param mat: torch.Tensor -- [input matrix, shape: (B, 3, 3)]
    :return: mat_inv: torch.Tensor -- [inversed matrix shape: (B, 3, 3)]
    '''
    if len(mat.shape) < 3:
        mat = mat[None]
    assert mat.shape[1:] == (3, 3)

    # Divide the matrix with it's maximum element
    max_vals = mat.max(1)[0].max(1)[0].view((-1, 1, 1))
    mat = mat / max_vals
    
    det = mat_3x3_det(mat)
    inv_det = 1.0 / det
    
    mat_inv = torch.zeros(mat.shape, device=mat.device)
    mat_inv[:, 0, 0] = (mat[:, 1, 1] * mat[:, 2, 2] - mat[:, 2, 1] * mat[:, 1, 2]) * inv_det
    mat_inv[:, 0, 1] = (mat[:, 0, 2] * mat[:, 2, 1] - mat[:, 0, 1] * mat[:, 2, 2]) * inv_det
    mat_inv[:, 0, 2] = (mat[:, 0, 1] * mat[:, 1, 2] - mat[:, 0, 2] * mat[:, 1, 1]) * inv_det
    mat_inv[:, 1, 0] = (mat[:, 1, 2] * mat[:, 2, 0] - mat[:, 1, 0] * mat[:, 2, 2]) * inv_det
    mat_inv[:, 1, 1] = (mat[:, 0, 0] * mat[:, 2, 2] - mat[:, 0, 2] * mat[:, 2, 0]) * inv_det
    mat_inv[:, 1, 2] = (mat[:, 1, 0] * mat[:, 0, 2] - mat[:, 0, 0] * mat[:, 1, 2]) * inv_det
    mat_inv[:, 2, 0] = (mat[:, 1, 0] * mat[:, 2, 1] - mat[:, 2, 0] * mat[:, 1, 1]) * inv_det
    mat_inv[:, 2, 1] = (mat[:, 2, 0] * mat[:, 0, 1] - mat[:, 0, 0] * mat[:, 2, 1]) * inv_det
    mat_inv[:, 2, 2] = (mat[:, 0, 0] * mat[:, 1, 1] - mat[:, 1, 0] * mat[:, 0, 1]) * inv_det
    
    # Divide the maximum value once more
    mat_inv = mat_inv / max_vals
    return mat_inv 


def planar_depth_loss(segments_im, surf3d_cam, im_xy_real, plane_thresh):
    
    #print("segments_im", segments_im.shape)
    #print("surf3d_cam", surf3d_cam.shape)
    #print("im_xy_real", im_xy_real.shape)
    #print("im_xy_real", im_xy_real)
    #print("plane_thresh", plane_thresh)

    # get segment ID for points
    #print("im_xy_real", im_xy_real)
    #print("torch.max(im_xy_real[:, 0])", torch.max(im_xy_real[:, 0]))
    #print("torch.max(im_xy_real[:, 1])", torch.max(im_xy_real[:, 1]))
    seg_idx =  torch.floor(im_xy_real)[:, torch.tensor([0, 1])][:, torch.tensor([1, 0])].long()
    #print("seg_idx", seg_idx)
    #print("torch.max(seg_idx[:, 0])", torch.max(seg_idx[:, 0]))
    #print("torch.max(seg_idx[:, 1])", torch.max(seg_idx[:, 1]))
    segments_cam = segments_im[seg_idx[:, 0], seg_idx[:, 1]]
    unique_segments_cam, counts_segments_cam = torch.unique(segments_cam, return_counts=True)
    #print("unique_segments_cam", unique_segments_cam)
    #print("counts_segments_cam", counts_segments_cam)
    #print("torch.max(counts_segments_cam)", torch.max(counts_segments_cam))
    #print("plane_thresh", plane_thresh)
    if torch.max(counts_segments_cam)<plane_thresh:
        loss_planar = 0
        return loss_planar
    '''
    for i, seg_id_old in enumerate(unique_segments_cam):
        if counts_segments_cam > plane_thresh:'''
    
    # segments_cam segID are not from 0 to max_num need to change them from 0 to max_num
    segments_cam_ordered = segments_cam.clone()
    for i, seg_id_old in enumerate(unique_segments_cam):
        segments_cam_ordered[segments_cam_ordered==seg_id_old] = i
    #print("segments_cam_ordered", segments_cam_ordered)
    #print("segments_cam_ordered_range", torch.unique(segments_cam_ordered, return_counts=True))
    seg_id, frequency = torch.unique(segments_cam_ordered, return_counts=True)
    #print("seg_id_before", seg_id)
    #print("frequency", frequency)
    seg_id = seg_id[frequency>=plane_thresh]
    #print("seg_id_after", seg_id)
    segments_cam_filtered = segments_cam_ordered.clone()
    #mask_filter = segments_cam_new==seg_id.all()
    
    C = (segments_cam_ordered[:, None] == seg_id[None, :])
    #print("C", C.shape)
    #print("C", C)
    mask_filter = C[:, 0]
    i=1
    while i<len(seg_id):
        i+=1
        mask_filter = mask_filter | C[:, i-1]
    '''
    for i in range (0, len(seg_id)):
        if i == 1:
            mask_filter = C[:, i-1]|C[:, i]
        else:
            mask_filter = mask_filter | C[:, i]'''
    #mask_filter = (C[:, 0] | C[:, 1]| C[:, 2]| C[:, 3]| C[:, 4]| C[:, 5]).nonzero()
    mask_filter = mask_filter.nonzero()

    #print("mask_filter", mask_filter.shape)
    segments_cam_filtered = segments_cam_filtered[mask_filter]
    #print("segments_cam_filtered", segments_cam_filtered.shape)
    surf3d_cam_new = surf3d_cam.clone()
    surf3d_cam_new = surf3d_cam_new[mask_filter].squeeze(1)
    #print("surf3d_cam_new", surf3d_cam_new.shape)
    unique_segments_cam = torch.unique(segments_cam_filtered)
    segments_cam_new = segments_cam_filtered.clone()
    for i, seg_id_old in enumerate(unique_segments_cam):
        segments_cam_new[segments_cam_new==seg_id_old] = i
    
    #stop
    
    

    max_num = segments_cam_new.max().item() + 1 # number of plane segmentation
    #print("max_num", max_num)
    #stop
    sum_points = torch.zeros((1, max_num, 3)).to(seg_idx.device)
    #print("sum_points", sum_points)
    #area = torch.zeros((1, max_num)).to(seg_idx.device)
    _, area = torch.unique(segments_cam_new, return_counts=True)
    for channel in range(3):
        points_channel = sum_points[:, :, channel]
        points_channel = points_channel.reshape(1, -1)
        segments_cam_new = segments_cam_new.to(torch.int64)
        points_channel.scatter_add_(1, segments_cam_new.view(1, -1), surf3d_cam_new[:, channel].view(1, -1))
        #stop
    #print("sum_points", sum_points)
    
    # X^T X
    cam_points_tmp = surf3d_cam_new.transpose(1,0).unsqueeze(0)
    x_T_dot_x = (cam_points_tmp.unsqueeze(1) * cam_points_tmp.unsqueeze(2)).view(1, 9, -1)
    X_T_dot_X = torch.zeros((1, max_num, 9)).cuda()
    for channel in range(9):
        points_channel = X_T_dot_X[:, :, channel]
        points_channel = points_channel.reshape(1, -1)
        points_channel.scatter_add_(1, segments_cam_new.view(1, -1),
                                            x_T_dot_x[:, channel, ...].view(1, -1))
    xTx = X_T_dot_X.view(1, max_num, 3, 3)
    
    # take inverse
    xTx_inv = mat_3X3_inv(xTx.view(-1, 3, 3) + 0.01*torch.eye(3).view(1,3,3).expand(1*max_num, 3, 3).cuda())
    xTx_inv = xTx_inv.view(xTx.shape)
        
    #print("sum_points", sum_points)
    #print("xTx_inv", xTx_inv.shape)
    #print("sum_points", sum_points.shape)
    xTx_inv_xT = torch.matmul(xTx_inv, sum_points.unsqueeze(3))
    plane_parameters = xTx_inv_xT.squeeze(3)
    #print("plane_parameters", plane_parameters.shape)
    #stop
    
    '''
    # generate mask for superpixel with area larger than plane_num_thresh
    #print("area", area.shape)
    valid_mask = ( area.unsqueeze(0) > plane_thresh).float()
    print("valid_mask_range", torch.unique(valid_mask))
    planar_mask = torch.gather(valid_mask, 1, segments_cam_new.view(1, -1))
    #planar_mask = planar_mask.view(1, 1, self.opt.height, self.opt.width)
    #print("planar_mask", planar_mask.shape)
    #stop'''
    
    # superpixel unpooling
    unpooled_parameters = []
    for channel in range(3):
        pooled_parameters_channel = plane_parameters[:, :, channel]
        pooled_parameters_channel = pooled_parameters_channel.reshape(1, -1)
        unpooled_parameter = torch.gather(pooled_parameters_channel, 1, segments_cam_new.view(1, -1))
        #print("unpooled_parameter", unpooled_parameter.shape)
        #unpooled_parameters.append(unpooled_parameter.view(self.opt.batch_size, 1, self.opt.height, self.opt.width))
        unpooled_parameters.append(unpooled_parameter.view(1, 1, -1))
    unpooled_parameters = torch.cat(unpooled_parameters, dim=1)
    #print("unpooled_parameters", unpooled_parameters.shape)
        
    #print("surf3d_cam", surf3d_cam.shape)
    #print("surf3d_cam", surf3d_cam)
    
    K_inv_dot_xy1 = surf3d_cam_new.clone()
    K_inv_dot_xy1[:, 0] = K_inv_dot_xy1[:, 0].clone()/K_inv_dot_xy1[:, -1].clone()
    K_inv_dot_xy1[:, 1] = K_inv_dot_xy1[:, 1].clone()/K_inv_dot_xy1[:, -1].clone()
    K_inv_dot_xy1[:, 2] = K_inv_dot_xy1[:, 2].clone()/K_inv_dot_xy1[:, -1].clone()
    #print("K_inv_dot_xy1", K_inv_dot_xy1.shape)
    planar_depth = 1. / (torch.sum(K_inv_dot_xy1.transpose(1,0).unsqueeze(0) * unpooled_parameters, dim=1) + 1e-6)
    #print("planar_depth", planar_depth.shape)
    
    pred_depth = surf3d_cam_new[:, -1]
    loss_planar = torch.mean(torch.abs(pred_depth - planar_depth)) # * planar_mask)
    #print("loss_planar", loss_planar)
    return loss_planar


def rgb_consistency_loss(SDF_pairs, vox_cam_0, vox_cam_1, inputs_extrinsics, inputs_proj_matrices, inputs_imgs, inputs_segments, img_xy_original, cam0_idx, cam1_idx, scale, save_depth, is_training):
    #img_xy_original [2,n,2]
    '''
    print("SDF_pairs", SDF_pairs.shape) #[n,1]
    print("vox_cam_0", vox_cam_0.shape) #[n,3]
    print("vox_cam_1", vox_cam_1.shape) #[n,3]
    print("inputs['extrinsics']", inputs['extrinsics'].shape)
    print("inputs['intrinsics']", inputs['intrinsics'].shape)
    print("inputs['proj_matrices']", inputs['proj_matrices'].shape) # [1,9,3,4,4]
    print("img_xy_original", img_xy_original.shape)'''
    
    #h = 480/(4*(2**scale))
    #w = 640/(4*(2**scale))
    h = 480/(2**scale)
    w = 640/(2**scale)
    #print('h, w', h, w)
    #stop
    plane_thresh = 1000/(4**scale)
    '''
    key_points_cam_0 = inputs['dso_points'][cam0_idx]
    #print("key_points_cam_0", key_points_cam_0.shape)
    #print("key_points_cam_0", key_points_cam_0)
    key_points_cam_1 = inputs['dso_points'][cam1_idx]
    s_key_points = 2**scale
    if s_key_points != 1:
        key_points_cam_0 = key_points_cam_0 // s_key_points
        key_points_cam_1 = key_points_cam_1 // s_key_points
    #print("key_points_cam_0", key_points_cam_0.shape)
    #print("key_points_cam_0", key_points_cam_0)
    #stop'''
    
    ###--------------------- get pix location(index) of original and reprojected --------------------------
    ratio_cam0 = (torch.norm(vox_cam_0, dim=1)+SDF_pairs.squeeze(1))/torch.norm(vox_cam_0, dim=1) #[n]
    ratio_cam1 = (torch.norm(vox_cam_1, dim=1)+SDF_pairs.squeeze(1))/torch.norm(vox_cam_1, dim=1)

    surf3d_cam0 = (ratio_cam0.unsqueeze(1).expand(ratio_cam0.shape[0], 3)) * vox_cam_0 #[n,3]
    surf3d_cam1 = (ratio_cam1.unsqueeze(1).expand(ratio_cam1.shape[0], 3)) * vox_cam_1
    if save_depth:
        depth_cam0 = surf3d_cam0[:,-1]
        #depth_cam1 = surf3d_cam1[:,-1]
    surf3d_cam0_homo = torch.cat([surf3d_cam0.permute(1,0), torch.ones([1, surf3d_cam0.shape[0]]).cuda()], dim=0) #[4,n]
    surf3d_cam1_homo = torch.cat([surf3d_cam1.permute(1,0), torch.ones([1, surf3d_cam1.shape[0]]).cuda()], dim=0)
    surf3d_cam0_world = inputs_extrinsics[0][cam0_idx] @ surf3d_cam0_homo
    pix_cam0_proj2cam1 = inputs_proj_matrices[0][cam1_idx][scale] @ surf3d_cam0_world
    #depth_cam0_proj2cam1 = pix_cam0_proj2cam1[2, :].T
    surf3d_cam1_world = inputs_extrinsics[0][cam1_idx] @ surf3d_cam1_homo
    pix_cam1_proj2cam0 = inputs_proj_matrices[0][cam0_idx][scale] @ surf3d_cam1_world
    #depth_cam1_proj2cam0 = pix_cam1_proj2cam0[2, :].T
    #print("pix_cam0_proj2cam1", pix_cam0_proj2cam1.shape)
    #print("pix_cam1_proj2cam0", pix_cam1_proj2cam0.shape)
    #stop
    
    
    im_x, im_y, im_z = pix_cam0_proj2cam1[0], pix_cam0_proj2cam1[1], pix_cam0_proj2cam1[2]
    #mask = (im_x > 0) & (im_x < (w-10))& (im_y > 0) & (im_y < (h-10)) & (im_z > 0)
    im_x = im_x / im_z
    im_y = im_y / im_z
    im_xy_recon0_in_cam1 = torch.stack([im_x, im_y], dim=-1)
    im_xy_0 = img_xy_original[0]
    #print("im_xy_recon0_in_cam1", im_xy_recon0_in_cam1.shape)
    #print("im_xy_0", im_xy_0.shape)
    
    im_grid = torch.stack([2 * im_x / (w - 1) - 1, 2 * im_y / (h - 1) - 1], dim=-1)
    mask = im_grid.abs() < 1
    mask = (mask.sum(dim=-1) == 2) & (im_z > 0) & (depth_cam0 > 0.3)
    #print("mask", mask.shape)
    
    #mask = (im_x > 0) & (im_x < w-2)& (im_y > 0) & (im_y < h-2) & (im_z > 0)
    im_xy_recon0_in_cam1 = im_xy_recon0_in_cam1[mask] #(n', 2)
    #depth_cam0_proj2cam1 = depth_cam0_proj2cam1[mask]

    #if im_xy_recon0_in_cam1.shape[0] < 0.5 * h*w: #== 0:
    if im_xy_recon0_in_cam1.shape[0] == 0:
        return None, None, None, None
    im_xy_0 = im_xy_0[mask] #(n', 2)
    surf3d_cam0 = surf3d_cam0[mask]
    if save_depth:
        depth_cam0 = depth_cam0[mask]
    
    #print("im_xy_recon0_in_cam1", im_xy_recon0_in_cam1.shape)
    #print("im_xy_0", im_xy_0.shape)
    
    
    
    im_x, im_y, im_z = pix_cam1_proj2cam0[0], pix_cam1_proj2cam0[1], pix_cam1_proj2cam0[2]
    #mask = (im_x > 0) & (im_x < (w-10))& (im_y > 0) & (im_y < (h-10)) & (im_z > 0)
    #depth_est_cam0 = im_z
    im_x = im_x / im_z
    im_y = im_y / im_z
    im_xy_recon1_in_cam0 = torch.stack([im_x, im_y], dim=-1)
    im_xy_1 = img_xy_original[1]
    #print("im_xy_recon1_in_cam0", im_xy_recon1_in_cam0.shape)
    #print("im_xy_1", im_xy_1.shape)
    
    im_grid = torch.stack([2 * im_x / (w - 1) - 1, 2 * im_y / (h - 1) - 1], dim=-1)
    #print("im_grid", im_grid.shape)
    mask = im_grid.abs() < 1
    mask = (mask.sum(dim=-1) == 2) & (im_z > 0)
    #print("mask", mask.shape)
    
    #mask = (im_x > 0) & (im_x < w)& (im_y > 0) & (im_y < h) & (im_z > 0)
    
    im_xy_recon1_in_cam0 = im_xy_recon1_in_cam0[mask] #(n'',2)
    #depth_cam1_proj2cam0 = depth_cam1_proj2cam0[mask]
    
    #if im_xy_recon1_in_cam0.shape[0] < 0.5 * h*w: #== 0:
    if im_xy_recon1_in_cam0.shape[0] == 0:
        return None, None, None, None
    im_xy_1 = im_xy_1[mask] #(n'',2)
    surf3d_cam1 = surf3d_cam1[mask]
    '''
    if save_depth:
        depth_cam1 = depth_cam1[mask]'''
    #print("im_xy_recon1_in_cam0", im_xy_recon1_in_cam0)
    #print("im_xy_1", im_xy_1.requires_grad)
    
    
    im_xy_0_real = im_xy_0.clone()
    im_xy_1_real = im_xy_1.clone()
    #im_xy_recon0_in_cam1_real = im_xy_recon0_in_cam1.clone()
    #im_xy_recon1_in_cam0_real = im_xy_recon1_in_cam0.clone()
    ###----------------get pix intensity value of original and reprojected-----------------------
    #print("inputs['imgs']", inputs['imgs'].shape)
    #img_0 = F.interpolate(inputs['imgs'][:, cam0_idx], scale_factor=1/(4*(2**scale)), mode="bilinear") #[1,3,30,40]
    img_0 = F.interpolate(inputs_imgs[:, cam0_idx], scale_factor=1/((2**scale)), mode="bilinear") #[1,3,120,160]
    #print("img_0", img_0.shape)
    #stop
    #img_1 = F.interpolate(inputs['imgs'][:, cam1_idx], scale_factor=1/(4*(2**scale)), mode="bilinear") #[1,3,30,40]
    img_1 = F.interpolate(inputs_imgs[:, cam1_idx], scale_factor=1/((2**scale)), mode="bilinear") #[1,3,120,160]
    
    '''
    pix_intensity_0 = img_0[:,:, torch.round(im_xy_0[:,1]).long(), torch.round(im_xy_0[:,0]).long()] #[1,3,n']
    print("pix_intensity_0", pix_intensity_0.shape)
    pix_intensity_0_recon = img_1[:,:, torch.round(im_xy_recon0_in_cam1[:,1]).long(), torch.round(im_xy_recon0_in_cam1[:,0]).long()] #[1,3,n']
    #print("pix_intensity_0_recon", pix_intensity_0_recon.shape)
    pix_intensity_1 = img_1[:,:, torch.round(im_xy_1[:,1]).long(), torch.round(im_xy_1[:,0]).long()] #[1,3,n'']
    pix_intensity_1_recon = img_0[:,:, torch.round(im_xy_recon1_in_cam0[:,1]).long(), torch.round(im_xy_recon1_in_cam0[:,0]).long()] #[1,3,n'']'''
    
    im_xy_recon0_in_cam1[:, 0] /= w - 1 
    im_xy_recon0_in_cam1[:, 1] /= h - 1
    im_xy_recon0_in_cam1 = (im_xy_recon0_in_cam1 - 0.5) * 2
    im_xy_recon0_in_cam1 = im_xy_recon0_in_cam1.unsqueeze(0).unsqueeze(1) # [1,1,n',2]
    #print("im_xy_recon0_in_cam1", im_xy_recon0_in_cam1.shape)
    pix_intensity_0_recon = F.grid_sample(img_1,
                    im_xy_recon0_in_cam1,
                    padding_mode="border").squeeze(2)
    #print("pix_intensity_0_recon", pix_intensity_0_recon.shape)
    
    im_xy_0[:, 0] /= w - 1 
    im_xy_0[:, 1] /= h - 1
    im_xy_0= (im_xy_0 - 0.5) * 2
    im_xy_0 = im_xy_0.unsqueeze(0).unsqueeze(1) # [1,1,n',2]
    pix_intensity_0 = F.grid_sample(img_0,
                    im_xy_0,
                    padding_mode="border").squeeze(2)
    #print("pix_intensity_0", pix_intensity_0.shape)
    
    im_xy_recon1_in_cam0[:, 0] /= w - 1 
    im_xy_recon1_in_cam0[:, 1] /= h - 1
    im_xy_recon1_in_cam0 = (im_xy_recon1_in_cam0 - 0.5) * 2
    im_xy_recon1_in_cam0 = im_xy_recon1_in_cam0.unsqueeze(0).unsqueeze(1) # [1,1,n',2]
    #print("im_xy_recon1_in_cam0", im_xy_recon1_in_cam0.shape)
    pix_intensity_1_recon = F.grid_sample(img_0,
                    im_xy_recon1_in_cam0,
                    padding_mode="border").squeeze(2)
    #print("pix_intensity_1_recon", pix_intensity_1_recon.shape)
    
    im_xy_1[:, 0] /= w - 1 
    im_xy_1[:, 1] /= h - 1
    im_xy_1= (im_xy_1 - 0.5) * 2
    im_xy_1 = im_xy_1.unsqueeze(0).unsqueeze(1) # [1,1,n',2]
    pix_intensity_1 = F.grid_sample(img_1,
                    im_xy_1,
                    padding_mode="border").squeeze(2)
    '''
    print("pix_intensity_1", pix_intensity_1.shape)
    print("pix_intensity_1", pix_intensity_1)
    print("pix_intensity_1_recon", pix_intensity_1_recon)
    stop'''
    
    ### ---------------------------------- planer depth --------------------------------------------
    #im_xy_recon0_in_cam1 #(n',2)
    #im_xy_0 #(n',2)
    #im_xy_recon1_in_cam0 #(n'',2)
    #im_xy_1 #(n'',2)
    '''
    for key in inputs:
        print("key", key)'''
    
    #print("inputs['segments'][1]", len(inputs['segments'][1]))
    #print("inputs['segments'][2]", len(inputs['segments'][2]))
    #print("inputs['segments'][4]", len(inputs['segments'][4]))
    
    #superpixel_im0 #(B, 1, H, W) stores plane ID of im0
    #superpixel_im1 #(B, 1, H, W)
    
    if is_training:      
        # get segment ID for points
        #print("inputs_segments[2**scale][cam0_idx]", inputs_segments[2**scale][cam0_idx].shape, inputs_segments[2**scale][cam0_idx].dtype)
        segments_im0 = inputs_segments[2**scale][cam0_idx].squeeze(0)
        segments_im1 = inputs_segments[2**scale][cam1_idx].squeeze(0)
        #print("segments_im0", segments_im0.shape, torch.unique(segments_im0))
        
        planar_loss_0 = planar_depth_loss(segments_im0, surf3d_cam0, im_xy_0_real, plane_thresh)
        planar_loss_1 = planar_depth_loss(segments_im1, surf3d_cam1, im_xy_1_real, plane_thresh)
        planar_loss = planar_loss_0 + planar_loss_1
        #planar_loss = 0
    else:
        planar_loss = 0
    ### -----------------------------------rgb consistency loss--------------------------------------------
    abs_diff = torch.abs(pix_intensity_0-pix_intensity_0_recon)
    #print("abs_diff", abs_diff.shape)
    l1_loss_0 = abs_diff.mean(1)
    #print('l1_loss_0', l1_loss_0.shape)
    
    abs_diff = torch.abs(pix_intensity_1-pix_intensity_1_recon)
    #print("abs_diff", abs_diff.shape)
    l1_loss_1 = abs_diff.mean(1)
    #print('l1_loss_1', l1_loss_1.shape)
    l1_loss = torch.cat((l1_loss_0, l1_loss_1), dim=1) #[1, n'+n'']
    #print('l1_loss', l1_loss)
    '''
    print("depth_cam0", torch.unique(depth_cam0), depth_cam0.shape)
    print("depth_cam1", torch.unique(depth_cam1), depth_cam1.shape)
    print("depth_cam0_proj2cam1", torch.unique(depth_cam0_proj2cam1), depth_cam0_proj2cam1.shape)
    print("depth_cam1_proj2cam0", torch.unique(depth_cam1_proj2cam0), depth_cam1_proj2cam0.shape)
    
    print("im_xy_recon0_in_cam1_real", im_xy_recon0_in_cam1_real.shape)
    print("im_xy_0_real", im_xy_0_real.shape)
    print("im_xy_recon1_in_cam0_real", im_xy_recon1_in_cam0_real.shape)
    print("im_xy_1_real", im_xy_1_real.shape)
    
    print("im_xy_recon0_in_cam1_real", im_xy_recon0_in_cam1_real)
    print("im_xy_0_real", im_xy_0_real)
    print("im_xy_recon1_in_cam0_real", im_xy_recon1_in_cam0_real)
    print("im_xy_1_real", im_xy_1_real)
    
    stop'''
    if save_depth:
        #return l1_loss, planar_loss, depth_cam0, depth_cam1, depth_cam0_proj2cam1, depth_cam1_proj2cam0, im_xy_0_real, im_xy_1_real, im_xy_recon0_in_cam1_real, im_xy_recon1_in_cam0_real
        return l1_loss, planar_loss, depth_cam0, im_xy_0_real
    
    else:
        return l1_loss, planar_loss



