import torch


def trilinear_interpolation(xd, yd, zd, c000, c100, c001, c101, c010, c110, c011, c111):
    #print("c000", c000.shape)
    #print("xd", xd.shape)
    #stop
    c00 = c000 * (1 - xd) + c100 * xd
    c01 = c001 * (1 - xd) + c101 * xd
    c10 = c010 * (1 - xd) + c110 * xd
    c11 = c011 * (1 - xd) + c111 * xd
        
    c0 = c00 * (1 - yd) + c10 * yd
    c1 = c01 * (1 - yd) + c11 * yd
        
    c = c0 * (1 - zd) + c1 * zd
    return c
    


#input
#ray_coord(1, 1024, 128, 3)
#vox_coord_all(1, 3, 96*96*96)
#vox_coord_idx(96*96*96, 4), if with mask, (N, 4). column 0 is batch index
#vox_sdf (N, 1)
#vox_features(96*96*96, 256)

#output
#ray_sdf (9, 1024, 128, 1)
#ray_deep_features (9, 1024, 128, 256)

def vox2ray(ray_coord, vox_coord_all, vox_coord_idx, vox_sdf, vox_feat):
    print("ray_coord", ray_coord.shape)
    #print("ray_coord", ray_coord)
    #print("ray_coord_max_before", torch.max(ray_coord[:, :, :, 0]),  torch.max(ray_coord[:, :, :, 1]),  torch.max(ray_coord[:, :, :, 2]))
    #print("ray_coord_min_before", torch.min(ray_coord[:, :, :, 0]),  torch.min(ray_coord[:, :, :, 1]),  torch.min(ray_coord[:, :, :, 2]))
    #print("ray_coord", ray_coord)
    print("vox_coord_all", vox_coord_all.shape)
    #print("vox_coord_all", vox_coord_all)
    #print("vox_coord_all", vox_coord_all)
    print("vox_coord_idx", vox_coord_idx.shape)
    #print("vox_coord_idx", vox_coord_idx)
    #print("vox_sdf", vox_sdf)
    print("vox_sdf", vox_sdf.shape)
    print("vox_feat", vox_feat.shape)
    #stop
    '''
    idx = vox_coord_idx[:, 1]* 96 * 96 + vox_coord_idx[:, 2] * 96 + vox_coord_idx[:, 3]
    print("idx", idx)
    vox_coord = vox_coord_all[:, :, idx].squeeze().T #(N,3)
    print("vox_coord", vox_coord.shape)
    ray_coord = ray_coord.squeeze(0)
    ray_sdf = torch.zeros((ray_coord.shape[0], ray_coord.shape[1], 1))
    for i in range(ray_coord.shape[0]):
        for j in range(ray_coord.shape[1]):
            dist = torch.norm(ray_coord[i,j]-vox_coord, dim=-1)    
            #print("dist", dist.shape)
            knn = dist.topk(8, largest=False)
            #print('kNN dist: {}, index: {}'.format(knn.values, knn.indices))
            ray_sdf[i, j] = torch.mean(vox_sdf[knn.indices])
    print("ray_sdf", ray_sdf.shape)
    print(stop)'''
    
    ray_coord = ray_coord.squeeze(0)
    
    #print("ray_coord", ray_coord.shape)
    min_vox = torch.Tensor([vox_coord_all[0,0,0], vox_coord_all[0,1,0], vox_coord_all[0,2,0]]).cuda()
    max_vox = torch.Tensor([vox_coord_all[0,0,-1*96**2-1], vox_coord_all[0,1,-1*96-1], vox_coord_all[0,2,-1-1]]).cuda()
    #print("min_vox", min_vox)
    #print("max_vox", max_vox)
    #print("ray_coord_max_before", torch.max(ray_coord[:, :, 0]),  torch.max(ray_coord[:, :, 1]),  torch.max(ray_coord[:, :, 2]))
    ray_coord[:, :, 0][ray_coord[:, :, 0]>max_vox[0]] = max_vox[0]
    ray_coord[:, :, 1][ray_coord[:, :, 1]>max_vox[1]] = max_vox[1]
    ray_coord[:, :, 2][ray_coord[:, :, 2]>max_vox[2]] = max_vox[2]
    #print("ray_coord_max_after", torch.max(ray_coord[:, :, 0]),  torch.max(ray_coord[:, :, 1]),  torch.max(ray_coord[:, :, 2]))
    
    #idx = (ray_coord - min_vox)//0.04
    idx = torch.div(ray_coord-min_vox, 0.04, rounding_mode='floor')
    print('ray_coord', ray_coord.shape)
    print("min_vox", min_vox.shape)
    print('ray_coord', ray_coord)
    print("min_vox", min_vox)
    print(idx)
    stop
    idx_ratio = idx
    #print("idx", idx.shape)
    #print("idx", idx)
    idx_1 = idx.clone()
    idx_1[:, :, 0] = idx_1[:, :, 0] + 1
    idx_2 = idx.clone()
    idx_2[:, :, 1] = idx_2[:, :, 1] + 1
    idx_3 = idx.clone()
    idx_3[:, :, 2] = idx_3[:, :, 2] + 1
    idx_4 = idx.clone()
    idx_4[:, :, 0] = idx_4[:, :, 0] + 1
    idx_4[:, :, 1] = idx_4[:, :, 1] + 1
    idx_5 = idx.clone()
    idx_5[:, :, 0] = idx_5[:, :, 0] + 1
    idx_5[:, :, 2] = idx_5[:, :, 2] + 1
    idx_6 = idx.clone()
    idx_6[:, :, 1] = idx_6[:, :, 1] + 1
    idx_6[:, :, 2] = idx_6[:, :, 2] + 1
    idx_7 = idx.clone()
    idx_7[:, :, 0] = idx_7[:, :, 0] + 1
    idx_7[:, :, 1] = idx_7[:, :, 1] + 1
    idx_7[:, :, 2] = idx_7[:, :, 2] + 1
    idx = (idx[:, :, 0]*(96**2)+idx[:, :, 1]*96+idx[:, :, 0]).to(dtype=torch.int64)
    #print("idx", idx.shape)
    #print("idx", idx)
    #stop
    idx_1 = (idx_1[:, :, 0]*(96**2)+idx_1[:, :, 1]*96+idx_1[:, :, 0]).to(dtype=torch.int64)
    idx_2 = (idx_2[:, :, 0]*(96**2)+idx_2[:, :, 1]*96+idx_2[:, :, 0]).to(dtype=torch.int64)
    idx_3 = (idx_3[:, :, 0]*(96**2)+idx_3[:, :, 1]*96+idx_3[:, :, 0]).to(dtype=torch.int64)
    idx_4 = (idx_4[:, :, 0]*(96**2)+idx_4[:, :, 1]*96+idx_4[:, :, 0]).to(dtype=torch.int64)
    idx_5 = (idx_5[:, :, 0]*(96**2)+idx_5[:, :, 1]*96+idx_5[:, :, 0]).to(dtype=torch.int64)
    idx_6 = (idx_6[:, :, 0]*(96**2)+idx_6[:, :, 1]*96+idx_6[:, :, 0]).to(dtype=torch.int64)
    idx_7 = (idx_7[:, :, 0]*(96**2)+idx_7[:, :, 1]*96+idx_7[:, :, 0]).to(dtype=torch.int64)
    #print("idx", idx)
    #print("vox_sdf", vox_sdf)
    sdf_0 = vox_sdf[idx]
    #print("idx", idx)
    #print("idx_max", torch.max(idx))
    #print("vox_sdf", vox_sdf)
    #print("sdf_0", sdf_0.shape)
    #print("idx_1", idx_1)
    #print("idx_1", idx_1.shape)
    #print("idx_1_max", torch.max(idx_1))
    #print("vox_sdf", vox_sdf)
    #print("vox_sdf", vox_sdf.shape)
    #print("idx_1", idx_1)
    sdf_1 = vox_sdf[idx_1]
    #print("sdf_1", sdf_1)
    sdf_2 = vox_sdf[idx_2]
    #print("sdf_2", sdf_2)
    sdf_3 = vox_sdf[idx_3]
    #print("sdf_3", sdf_3)
    sdf_4 = vox_sdf[idx_4]
    #print("sdf_4", sdf_4)
    sdf_5 = vox_sdf[idx_5]
    #print("sdf_5", sdf_5)
    sdf_6 = vox_sdf[idx_6]
    #print("sdf_6", sdf_6)
    sdf_7 = vox_sdf[idx_7]
    #print("sdf_7", sdf_7)
    feat_0 = vox_feat[idx]
    feat_1 = vox_feat[idx_1]
    feat_2 = vox_feat[idx_2]
    feat_3 = vox_feat[idx_3]
    feat_4 = vox_feat[idx_4]
    feat_5 = vox_feat[idx_5]
    feat_6 = vox_feat[idx_6]
    feat_7 = vox_feat[idx_7]
    
    # Trilinear interpolation
    ratio = (ray_coord - (idx_ratio * 0.04 + min_vox))/0.04
    xd = ratio[:, :, 0].unsqueeze(-1)
    yd = ratio[:, :, 1].unsqueeze(-1)
    zd = ratio[:, :, 2].unsqueeze(-1)
    ray_sdf = trilinear_interpolation(xd, yd, zd, sdf_0, sdf_3, sdf_1, sdf_5, sdf_2, sdf_6, sdf_4, sdf_7).permute(2, 0, 1)
    ray_feat = trilinear_interpolation(xd, yd, zd, feat_0, feat_3, feat_1, feat_5, feat_2, feat_6, feat_4, feat_7)
    #print("ray_sdf", ray_sdf.shape)
    #print("ray_feat", ray_feat.shape)
    #stop
    
    '''
    #averayge sdf of the sdf at 8 vertices of the voxels enclosing the sample point
    ray_sdf = torch.mean(torch.stack((sdf_0, sdf_1, sdf_2, sdf_3, sdf_4, sdf_5, sdf_6, sdf_7)), dim=0).permute(2,0,1)
    ray_feat = torch.mean(torch.stack((feat_0, feat_1, feat_2, feat_3, feat_4, feat_5, feat_6, feat_7)), dim=0)
    #ray_sdf = ray_sdf.reshape(1, ray_sdf.shape[0]*ray_sdf.shape[1])
    print("ray_sdf", ray_sdf.shape)
    print("ray_feat", ray_feat.shape)
    stop'''
    
    '''
    vox_0 = torch.zeros((idx.shape[0], idx.shape[1], 3))
    vox_0[:,:] = vox_coord_all.squeeze(0)[:,idx].permute(1,2,0)
    vox_1 = torch.zeros((idx.shape[0], idx.shape[1], 3))
    vox_1[:,:] = vox_coord_all.squeeze(0)[:,idx_1].permute(1,2,0)
    vox_2 = torch.zeros((idx.shape[0], idx.shape[1], 3))
    vox_2[:,:] = vox_coord_all.squeeze(0)[:,idx_2].permute(1,2,0)
    vox_3 = torch.zeros((idx.shape[0], idx.shape[1], 3))
    vox_3[:,:] = vox_coord_all.squeeze(0)[:,idx_3].permute(1,2,0)
    vox_4 = torch.zeros((idx.shape[0], idx.shape[1], 3))
    vox_4[:,:] = vox_coord_all.squeeze(0)[:,idx_4].permute(1,2,0)
    vox_5 = torch.zeros((idx.shape[0], idx.shape[1], 3))
    vox_5[:,:] = vox_coord_all.squeeze(0)[:,idx_5].permute(1,2,0)
    vox_6 = torch.zeros((idx.shape[0], idx.shape[1], 3))
    vox_6[:,:] = vox_coord_all.squeeze(0)[:,idx_6].permute(1,2,0)
    vox_7 = torch.zeros((idx.shape[0], idx.shape[1], 3))
    vox_7[:,:] = vox_coord_all.squeeze(0)[:,idx_7].permute(1,2,0)
    '''


    '''
    for i in range(ray_coord.shape[0]):
        for j in range(ray_coord.shape[1]):
            x_idx = (ray_coord[i,j,0]-vox_coord_all[0,0,0])//0.04
            y_idx = (ray_coord[i,j,1]-vox_coord_all[0,1,0])//0.04
            z_idx = (ray_coord[i,j,2]-vox_coord_all[0,2,0])//0.04
            idx = (x_idx*96*96+y_idx*96+z_idx,
                   (x_idx+1)*96*96+y_idx*96+z_idx,
                   x_idx*96*96+(y_idx+1)*96+z_idx,
                   x_idx*96*96+y_idx*96+(z_idx+1),
                   (x_idx+1)*96*96+(y_idx+1)*96+z_idx,
                   (x_idx+1)*96*96+y_idx*96+(z_idx+1),
                   x_idx*96*96+(y_idx+1)*96+(z_idx+1),
                   (x_idx+1)*96*96+(y_idx+1)*96+(z_idx+1))
            sdf_vertices = vox_sdf[idx, :]
            ray_coord[i, j] = torch.mean(tsdf_vertices) #averayge sdf of the sdf at 8 vertices of the voxels enclosing the sample point
    print("end")'''
            
            
    
    return ray_sdf, ray_feat