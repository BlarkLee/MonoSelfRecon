from yacs.config import CfgNode as CN

_C = CN()

_C.MODE = 'train'
_C.DATASET = 'scannet'
_C.BATCH_SIZE = 1
_C.LOADCKPT = ''
_C.LOGDIR = './checkpoints/debug'
_C.ENCODER_DIR = '/home/blark/Desktop/NeuralRecon/checkpoints/release/model_000047.ckpt'
_C.RESUME = True
_C.SUMMARY_FREQ = 20
_C.SAVE_FREQ = 1
_C.SEED = 1
_C.SAVE_SCENE_MESH = False
_C.SAVE_INCREMENTAL = False
_C.VIS_INCREMENTAL = False
_C.REDUCE_GPU_MEM = False

_C.LOCAL_RANK = 0
_C.DISTRIBUTED = False

# train
_C.TRAIN = CN()
_C.TRAIN.PATH = ''
_C.TRAIN.EPOCHS = 40
_C.TRAIN.LR = 0.001
_C.TRAIN.LREPOCHS = '12,24,36:2'
_C.TRAIN.WD = 0.0
_C.TRAIN.N_VIEWS = 5
_C.TRAIN.N_WORKERS = 8
_C.TRAIN.RANDOM_ROTATION_3D = False #True
_C.TRAIN.RANDOM_TRANSLATION_3D = False #True
_C.TRAIN.PAD_XY_3D = .1
_C.TRAIN.PAD_Z_3D = .025
_C.TRAIN.near = 0.0
_C.TRAIN.far = 2.0

# test
_C.TEST = CN()
_C.TEST.PATH = ''
_C.TEST.N_VIEWS = 5
_C.TEST.N_WORKERS = 4
_C.TEST.near = 0.0
_C.TEST.far = 2.0

# model
_C.MODEL = CN()
_C.MODEL.N_VOX = [128, 224, 192]
_C.MODEL.VOXEL_SIZE = 0.04
_C.MODEL.THRESHOLDS = [0, 0, 0]
_C.MODEL.N_LAYER = 3

_C.MODEL.TRAIN_NUM_SAMPLE = [4096, 16384, 65536]
_C.MODEL.TEST_NUM_SAMPLE = [32768, 131072]

_C.MODEL.LW = [1.0, 0.8, 0.64]
_C.MODEL.rgb_weight = 1.0
_C.MODEL.eikonal_weight = 0.1

# TODO: images are currently loaded RGB, but the pretrained models expect BGR
_C.MODEL.PIXEL_MEAN = [103.53, 116.28, 123.675]
_C.MODEL.PIXEL_STD = [1., 1., 1.]
_C.MODEL.THRESHOLDS = [0, 0, 0]
_C.MODEL.POS_WEIGHT = 1.0

_C.MODEL.BACKBONE2D = CN()
_C.MODEL.BACKBONE2D.ARC = 'fpn-mnas'

_C.MODEL.SPARSEREG = CN()
_C.MODEL.SPARSEREG.DROPOUT = False

_C.MODEL.FUSION = CN()
_C.MODEL.FUSION.FUSION_ON = False
_C.MODEL.FUSION.HIDDEN_DIM = 64
_C.MODEL.FUSION.AVERAGE = False
_C.MODEL.FUSION.FULL = False
'''
_C.MODEL.radiance = CN()
_C.MODEL.radiance.net_depth = 4
_C.MODEL.radiance.skips = []
_C.MODEL.radiance.fr_pos = -1
_C.MODEL.radiance.fr_view = -1
_C.MODEL.radiance.weight_norm = True'''

  #MPINerf
_C.MODEL.disparity_end = 0.3 #0.2 #0.001
_C.MODEL.disparity_start = 2.0 #20 #1.0
_C.MODEL.is_bg_depth_inf = False
_C.MODEL.num_bins_coarse = 32
_C.MODEL.num_bins_fine = 0
_C.MODEL.valid_mask_threshold = 2
_C.MODEL.fix_disparity = False
_C.MODEL.use_alpha = False 

# SDFNerfNet
_C.MODEL.net_width = 256
_C.MODEL.feature_width = 256
_C.MODEL.bounding_radius = 2.0

_C.MODEL.sdf = CN()
_C.MODEL.sdf.geometric_init = True
_C.MODEL.sdf.radius_init = 1.0
_C.MODEL.sdf.net_depth = 8
_C.MODEL.sdf.skips = [4]
_C.MODEL.sdf.fr_pos = 6
_C.MODEL.sdf.weight_norm = True

_C.MODEL.radiance = CN()
_C.MODEL.radiance.net_depth = 4
_C.MODEL.radiance.skips = []
_C.MODEL.radiance.fr_pos = -1
_C.MODEL.radiance.fr_view = -1
_C.MODEL.radiance.weight_norm = True

_C.MODEL.semantic = CN()
_C.MODEL.semantic.net_depth = 4
_C.MODEL.semantic.skips = []
_C.MODEL.semantic.fr_pos = -1
_C.MODEL.semantic.fr_view = -1
_C.MODEL.semantic.weight_norm = True

_C.MODEL.beta_init = 0.1
_C.MODEL.speed_factor = 10.0
    
_C.sample = CN()    
_C.sample.N_samples = 128
_C.sample.N_importance = 64
_C.sample.rayschunk = 65536
_C.sample.netchunk = 1048576
_C.sample.max_upsample_steps = 6
_C.sample.max_bisection_steps = 10
_C.sample.epsilon = 0.1


def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    cfg.freeze()


def check_config(cfg):
    pass
