import torch
import numpy as np

def ray_sample_coarse(outputs):
    #print("outputs['coords']", outputs['coords'].shape)
    #print("outputs['coords']", outputs['coords'])
    #mask = outputs['occupancy']
    #print("mask", mask.shape)
    #mask_out, counts = torch.unique(mask, return_counts=True)
    #print("mask_out", mask_out)
    #print("counts", counts)
    #z_range = torch.sort(outputs['coords'][: ,-1])
    #print("z_range", z_range)
    #print("sdf", outputs['sdf'].shape)
    #print("sdf_range", np.unique(outputs['sdf'].detach().cpu().numpy()))
    
    
    mask = outputs['coords'][:, 1:]
    '''
    indices = torch.tensor([[2,2,2],
                           [2,2,-2],
                           [2,-2,2],
                           [2,-2,-2],
                           [-2,2,2],
                           [-2,2,-2],
                           [-2,-2,2],
                           [-2,-2,-2]]).to(mask.device)'''
    '''
    indices = torch.tensor([[1,1,1],
                           [1,1,-1],
                           [1,-1,1],
                           [1,-1,-1],
                           [-1,1,1],
                           [-1,1,-1],
                           [-1,-1,1],
                           [-1,-1,-1]])'''
    
    '''
    for i in range(-2, 3):
        for j in range (-2, 3):
            for k in range (-2, 3):
                if (i==-2 and j==-2 and k==-2):
                    indices = torch.tensor([[i,j,k]])
                else:
                    indices = torch.cat((indices, torch.tensor([[i,j,k]])))'''
    
    
    for i in range(-3, 4):
        for j in range (-3, 4):
            for k in range (-3, 4):
                if (i==-3 and j==-3 and k==-3):
                    indices = torch.tensor([[i,j,k]])
                else:
                    indices = torch.cat((indices, torch.tensor([[i,j,k]])))
    indices = indices.to(mask.device)
    #print("indices", indices.shape)
    #print("indices", indices)
    #stop
    #print("mask", mask.shape)
    #stop
    for i, idx in enumerate(indices):
        #mask_i = mask.clone()
        #mask_i[:,0] += idx[0]
        #mask_i[:,1] += idx[1]
        #mask_i[:,2] += idx[2]
        mask_i = mask + idx
        C = (mask[:, None] == mask_i[None, :])
        i_mask = (C[..., 0] & C[..., 1] & C[..., 2]).nonzero()[:, 0].unsqueeze(1)
        #print("i_mask", i_mask.shape)
        if i == 0:
            mask_current = i_mask
        else:
            #print("i_mask", i_mask)
            #print("mask_current", mask_current)
            #print("i_mask", i_mask[:100, :])
            #print("mask_current", mask_current[:100, :])
            #stop
            C = (mask_current[:, None] == i_mask[None, :])
            i_mask = (C[..., 0]).nonzero()[:, 0]
            mask_current = mask_current[i_mask, :]
            #print("mask_current", mask_current[:100, :])
    mask_all = torch.unique(mask_current)
    #print("mask_all", mask_all.shape)
    #print("mask_all", mask_all)
    
    masked_coords = outputs['coords'][mask_all, 1:]
    #print("masked_coords", masked_coords.shape)
    
    return masked_coords