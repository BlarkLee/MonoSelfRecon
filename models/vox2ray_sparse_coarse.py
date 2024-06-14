import torch

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
        
        
        idx_unique, inverse_indices = torch.unique(idx, return_inverse=True, dim=0)
        
        #print("idx_unique", idx_unique)
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
        
        #ray_sdf = trilinear_interpolation(ray_coord[j], vert_sdf0, vert_sdf3, vert_sdf1, vert_sdf5, vert_sdf2, vert_sdf6, vert_sdf4, vert_sdf7)
        ray_sdf = vert_sdf0.squeeze(1).unsqueeze(0).unsqueeze(0)
        #print('ray_sdf', ray_sdf.shape)
        #ray_feat = trilinear_interpolation(ray_coord[j], vert_feat0, vert_feat3, vert_feat1, vert_feat5, vert_feat2, vert_feat6, vert_feat4, vert_feat7)
        ray_feat = vert_feat0.squeeze(1).unsqueeze(0).unsqueeze(0)
        
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

    