#BATCH_SIZE = 1024 #8
BATCH_SIZE = 1024 #8
INPUT_DIM = 4
EMBEDDING_DIM = 128
SAMPLE_SIZE = 200
K_SIZE = 20
BUDGET_RANGE = (6, 8)
SAMPLE_LENGTH = 0.2

ADAPTIVE_AREA = True
ADAPTIVE_TH = 0.4

USE_GPU = False
USE_GPU_GLOBAL = True
#CUDA_DEVICE = [0, 1, 2, 3]
CUDA_DEVICE = [0,1]
#NUM_META_AGENT = 32 #6
NUM_META_AGENT = 24
LR = 1e-4
GAMMA = 0.9   # 0.9 0.95 0.99
DECAY_STEP = 32
SUMMARY_WINDOW = 8
#FOLDER_NAME = 'catipp-server-test-008'

#FOLDER_NAME = 'ipp_DQN_IDS_001'
#FOLDER_NAME = 'test-ipp_DQN_IDS_017'
#FOLDER_NAME = 'CATIPP-053'
#FOLDER_NAME = 'test-02'
#FOLDER_NAME = 'ipp_TUF_001'
#FOLDER_NAME = 'CATIPP-reward-001'
FOLDER_NAME = 'CATIPP-add-119'
#FOLDER_NAME = 'CATIPP-add-008'
SEED = 512 # 3407
CAT_POLICY = False # False

NON_IDS = False # False  True
RAND_POLICY = True # False  True

CHANGE_DQN_NET = True
SOFTMAX_SCALE = 30
'''
 'ipp_DQN_IDS_001' : 0.5*var
 'ipp_DQN_IDS_002' : 0.8*var
 'ipp_DQN_IDS_003' : 1.5*var
 'ipp_DQN_IDS_006' : 1.0*var
 'ipp_DQN_IDS_009' : Q 正则化 去掉mul环节
 'ipp_DQN_IDS_server_001' : Q 正则化 恢复mul环节
 'ipp_DQN_IDS_server_002' : Q 正则化 没有mul环节
 'ipp_DQN_IDS_server_003' : Q 无正则化 恢复mul环节
 'ipp_DQN_IDS_server_004' : Q 正则化 没有mul环节
 'ipp_DQN_IDS_server_005' : Q 无正则化 没有mul环节
 'ipp_DQN_IDS_server_006' : Q 无正则化 恢复mul环节
 'ipp_DQN_IDS_server_007' : Q 无正则化 没有mul环节
 'ipp_DQN_IDS_server_seed_001'  : Q 无正则化 恢复mul环节 SEED = 3407
 'ipp_DQN_IDS_server_seed_002'  : Q 无正则化 恢复mul环节 SEED = 3407 去掉var里面的mean 好像意外中断了
 'ipp_DQN_IDS_server_seed_003'  : Q 无正则化 恢复mul环节 SEED = 3407 去掉var里面的mean
 'ipp_DQN_IDS_server_seed_004'  : Q 无正则化 恢复mul环节 SEED = 3407 去掉var里面的mean GAMMA = 0.9 原来是1
 'ipp_DQN_IDS_server_seed_005'  : Q 无正则化 恢复mul环节 SEED = 3407 去掉var里面的mean 更改参数记录在下边
 'ipp_DQN_IDS_server_seed_006'  : Q 无正则化 恢复mul环节 SEED = 3407 去掉var里面的mean 更改参数记录在下边 中断
 'ipp_DQN_IDS_server_seed_007'  : Q 无正则化 恢复mul环节 SEED = 3407 去掉var里面的mean 更改参数记录在下边 中断了
 'ipp_DQN_IDS_server_seed_008'  : Q 无正则化 恢复mul环节 SEED = 3407 去掉var里面的mean 更改参数记录在下边 
 'ipp_DQN_IDS_server_seed_009'  : Q 无正则化 恢复mul环节 SEED = 3407 去掉var里面的mean 更改参数记录在下边 
 'ipp_DQN_IDS_server_seed_010'  : Q 无正则化 恢复mul环节 SEED = 3407 去掉var里面的mean 更改参数记录在下边 
 'ipp_DQN_IDS_server_seed_011'  : Q 无正则化 恢复mul环节 SEED = 3407 去掉var里面的mean 更改参数记录在下边 
 'ipp_DQN_IDS_server_seed_012'  : Q 无正则化 恢复mul环节 SEED = 3407 去掉var里面的mean 更改参数记录在下边 换一下policy
 'ipp_DQN_IDS_server_seed_013'  : Q 无正则化 恢复mul环节 SEED = 3407 去掉var里面的mean 更改参数记录在下边 换回去了
 'ipp_DQN_IDS_server_seed_014'  : Q 无正则化 恢复mul环节 SEED = 3407                 更改参数记录在下边 换回去了
'''
model_path = f'../../catnipp_ws/model/{FOLDER_NAME}'
train_path = f'../../catnipp_ws/train/{FOLDER_NAME}'
gifs_path = f'../../catnipp_ws/gifs/{FOLDER_NAME}'
LOAD_MODEL = False
SAVE_IMG_GAP = 1000


EXPLORE_RATE = 0.25 # 
BUFFER_SIZE = int(1e4)

SAMPLE_SIZE = 360
START_TRAIN_BUFFER_SIZE = 1000
REPEAT_TIMES = 1
SOFT_UPDATE_TAU = 5e-3

IF_AE = False

IF_IDS = True #False  True
N_STDS = 0.1
RHO2 = 1.0**2

LAMDA = 1
'''
n_stds=0.1              # Uncertainty scale for computing regret
self.n_stds = n_stds    # Number of standard deviations for computing uncertainty
self.rho2   = 1.0**2    # Return distribution variance
'''


'''
ipp_DQN_IDS_server_seed_005:
START_TRAIN_BUFFER_SIZE = 100
BUFFER_SIZE = int(1e4)
NUM_META_AGENT = 24
LR = 0.001  #1e-4
GAMMA = 0.9   # 0.9 0.95 0.99

ipp_DQN_IDS_server_seed_006:
START_TRAIN_BUFFER_SIZE = 1000
BUFFER_SIZE = int(1e4)
NUM_META_AGENT = 24
LR = 1e-4  #1e-4
GAMMA = 0.9   # 0.9 0.95 0.99

ipp_DQN_IDS_server_seed_009:
START_TRAIN_BUFFER_SIZE = 1000
BUFFER_SIZE = int(1e4)
NUM_META_AGENT = 24
LR = 1e-4  #1e-4
GAMMA = 0.99   # 0.9 0.95 0.99
BATCH_SIZE = 512 #8

ipp_DQN_IDS_server_seed_010:
START_TRAIN_BUFFER_SIZE = 1000
BUFFER_SIZE = int(8e3)
NUM_META_AGENT = 32
LR = 1e-3  #1e-4
GAMMA = 0.95   # 0.9 0.95 0.99
BATCH_SIZE = 1024 #8

ipp_DQN_IDS_server_seed_011:
START_TRAIN_BUFFER_SIZE = 1000
BUFFER_SIZE = int(8e3)
NUM_META_AGENT = 32
LR = 1e-4  #1e-4
GAMMA = 0.95   # 0.9 0.95 0.99
BATCH_SIZE = 1024 #8

ipp_DQN_IDS_server_seed_012:
START_TRAIN_BUFFER_SIZE = 1000
BUFFER_SIZE = int(1e4)
NUM_META_AGENT = 24
LR = 1e-4  #1e-4
GAMMA = 0.9   # 0.9 0.95 0.99
BATCH_SIZE = 1024 #8
CAT_POLICY = True

ipp_DQN_IDS_server_seed_013:
START_TRAIN_BUFFER_SIZE = 1000
BUFFER_SIZE = int(1e4)
NUM_META_AGENT = 24
LR = 1e-4 #1e-4 
GAMMA = 0.9   # 0.9 0.95 0.99
CAT_POLICY = False

ipp_DQN_IDS_server_seed_014:
START_TRAIN_BUFFER_SIZE = 1000
BUFFER_SIZE = int(1e4)
NUM_META_AGENT = 24
LR = 1e-4 #1e-4 
GAMMA = 0.9   # 0.9 0.95 0.99
CAT_POLICY = False
恢复var里面的mean


'''