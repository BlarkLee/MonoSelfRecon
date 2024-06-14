import torch


def trilinear_interpolation(ray_coord, c000, c100, c001, c101, c010, c110, c011, c111):
    #print("c000", c000.shape)
    #print("xd", xd.shape)
    #stop
    
    xd = ray_coord[:, 0].unsqueeze(1)
    yd = ray_coord[:, 1].unsqueeze(1)
    zd = ray_coord[:, 2].unsqueeze(1)
    
    
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

def vox2ray(ray_coord, origin, vox_coord_idx, vox_sdf, vox_feat):
    #print("ray_coord", ray_coord.shape)
    #print("vox_coord_idx", vox_coord_idx.shape)
    #print("vox_sdf", vox_sdf.shape)
    #print("vox_feat", vox_feat.shape)
    
    #print("ray_coord", ray_coord)
    #print("origin", origin)
    #stop
    indices = torch.div(ray_coord-origin, 0.04, rounding_mode='floor') #(1024,128,3)
    #print("indices", indices.shape)
    for j,idx in enumerate(indices): #idx: (128,3)
        #print("j",j)
        
        idx_1 = idx.clone()
        idx_1[:, 0] = idx_1[:, 0] + 1
        idx_2 = idx.clone()
        idx_2[:, 1] = idx_2[:, 1] + 1
        idx_3 = idx.clone()
        idx_3[:, 2] = idx_3[:, 2] + 1
        idx_4 = idx.clone()
        idx_4[:, 0] = idx_4[:, 0] + 1
        idx_4[:, 1] = idx_4[:, 1] + 1
        idx_5 = idx.clone()
        idx_5[:, 0] = idx_5[:, 0] + 1
        idx_5[:, 2] = idx_5[:, 2] + 1
        idx_6 = idx.clone()
        idx_6[:, 1] = idx_6[:, 1] + 1
        idx_6[:, 2] = idx_6[:, 2] + 1
        idx_7 = idx.clone()
        idx_7[:, 0] = idx_7[:, 0] + 1
        idx_7[:, 1] = idx_7[:, 1] + 1
        idx_7[:, 2] = idx_7[:, 2] + 1
        
        idx_unique, inverse_indices = torch.unique(idx, return_inverse=True, dim=0)
        
        #print("idx_unique", idx_unique)
        '''
        x_unique = torch.unique(idx_unique[:, 0])
        print("x_unique", x_unique)
        for i in x_unique:
            for k in vox_coord_idx:
                if k[0] == i:
                    print('haha', k)
        #stop
        print("x_unique_after", x_unique)
        print("idx_unique_after", idx_unique)'''
        #print("idx_unique", idx_unique.shape) #(8,3)
        #print("inverse_indices", inverse_indices.shape) # (128)
        mask_unique = [(vox_coord_idx==i).all(1).nonzero(as_tuple=True)[0][0] for i in idx_unique]
        
        #print("mask_unique", mask_unique)
        #print("mask_unique", len(mask_unique)) # (8)
        #mask = mask_unique[inverse_indices]
        mask = [mask_unique[i] for i in inverse_indices]
        #print("mask", mask)
        #print("mask", len(mask))
        vert_sdf0 = vox_sdf[mask]
        #print("vert_sdf", vert_sdf.shape)
        #print("vert_sdf", vert_sdf)
        vert_feat0 = vox_feat[mask]
        #print("vert_feat", vert_feat.shape)
        
        
        idx_unique, inverse_indices = torch.unique(idx_1, return_inverse=True, dim=0)
        mask_unique = [(vox_coord_idx==i).all(1).nonzero(as_tuple=True)[0][0] for i in idx_unique]
        mask = [mask_unique[i] for i in inverse_indices]
        vert_sdf1 = vox_sdf[mask]
        vert_feat1 = vox_feat[mask]
        
        idx_unique, inverse_indices = torch.unique(idx_2, return_inverse=True, dim=0)
        mask_unique = [(vox_coord_idx==i).all(1).nonzero(as_tuple=True)[0][0] for i in idx_unique]
        mask = [mask_unique[i] for i in inverse_indices]
        vert_sdf2 = vox_sdf[mask]
        vert_feat2 = vox_feat[mask]
        
        idx_unique, inverse_indices = torch.unique(idx_3, return_inverse=True, dim=0)
        mask_unique = [(vox_coord_idx==i).all(1).nonzero(as_tuple=True)[0][0] for i in idx_unique]
        mask = [mask_unique[i] for i in inverse_indices]
        vert_sdf3 = vox_sdf[mask]
        vert_feat3 = vox_feat[mask]
        
        idx_unique, inverse_indices = torch.unique(idx_4, return_inverse=True, dim=0)
        mask_unique = [(vox_coord_idx==i).all(1).nonzero(as_tuple=True)[0][0] for i in idx_unique]
        mask = [mask_unique[i] for i in inverse_indices]
        vert_sdf4 = vox_sdf[mask]
        vert_feat4 = vox_feat[mask]
        
        idx_unique, inverse_indices = torch.unique(idx_5, return_inverse=True, dim=0)
        mask_unique = [(vox_coord_idx==i).all(1).nonzero(as_tuple=True)[0][0] for i in idx_unique]
        mask = [mask_unique[i] for i in inverse_indices]
        vert_sdf5 = vox_sdf[mask]
        vert_feat5 = vox_feat[mask]
        
        idx_unique, inverse_indices = torch.unique(idx_6, return_inverse=True, dim=0)
        mask_unique = [(vox_coord_idx==i).all(1).nonzero(as_tuple=True)[0][0] for i in idx_unique]
        mask = [mask_unique[i] for i in inverse_indices]
        vert_sdf6 = vox_sdf[mask]
        vert_feat6 = vox_feat[mask]
        
        idx_unique, inverse_indices = torch.unique(idx_7, return_inverse=True, dim=0)
        mask_unique = [(vox_coord_idx==i).all(1).nonzero(as_tuple=True)[0][0] for i in idx_unique]
        mask = [mask_unique[i] for i in inverse_indices]
        vert_sdf7 = vox_sdf[mask]
        vert_feat7 = vox_feat[mask]
        
        ray_sdf = trilinear_interpolation(ray_coord[j], vert_sdf0, vert_sdf3, vert_sdf1, vert_sdf5, vert_sdf2, vert_sdf6, vert_sdf4, vert_sdf7)
        ray_sdf = ray_sdf.squeeze(1).unsqueeze(0).unsqueeze(0)
        #print('ray_sdf', ray_sdf.shape)
        ray_feat = trilinear_interpolation(ray_coord[j], vert_feat0, vert_feat3, vert_feat1, vert_feat5, vert_feat2, vert_feat6, vert_feat4, vert_feat7)
        ray_feat = ray_feat.squeeze(1).unsqueeze(0).unsqueeze(0)
        
        #print("ray_feat", ray_feat.shape)
        
        if j==0:
            ray_sdfs = ray_sdf
            ray_feats = ray_feat
        else:
            ray_sdfs = torch.cat((ray_sdfs, ray_sdf), 1)
            ray_feats = torch.cat((ray_feats, ray_feat), 1)
        
    #print("ray_sdfs", ray_sdfs.shape) 
    #stop
    return ray_sdfs, ray_feats

    