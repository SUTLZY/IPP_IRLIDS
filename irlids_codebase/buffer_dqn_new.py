import torch
from torch import Tensor
from parameters_dqn_new import *


class ReplayBuffer:  # for off-policy
    def __init__(self, max_size: int, ##state_dim: int, action_dim: int, 
                 sample_size: int,):
        self.p = 0  # pointer
        self.if_full = False
        self.cur_size = 0
        self.max_size = max_size
        #self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")
        #self.device = torch.device('cuda') if USE_GPU_GLOBAL else torch.device('cpu')
        self.device = torch.device('cpu')
        self.device_gpu = torch.device('cuda')
        ##self.states = torch.empty((max_size, state_dim), dtype=torch.float32, device=self.device)
        ##self.actions = torch.empty((max_size, action_dim), dtype=torch.float32, device=self.device)
        ##self.rewards = torch.empty((max_size, 1), dtype=torch.float32, device=self.device)
        ##self.undones = torch.empty((max_size, 1), dtype=torch.float32, device=self.device)
        self.node_inputs_batch = torch.empty((max_size, sample_size+2, 4), device=self.device) # (batch,sample_size+2,2)
        self.edge_inputs_batch = torch.empty((max_size, sample_size+2, 20), dtype=torch.int64, device=self.device) # (batch,sample_size+2,k_size)
        self.current_inputs_batch = torch.empty((max_size, 1, 1), dtype=torch.int64, device=self.device) # (batch,1,1)
        self.action_batch = torch.empty((max_size, 1, 1), dtype=torch.int64, device=self.device) # (batch,1,1)
        
        self.reward_batch = torch.empty((max_size, 1, 1), device=self.device) # (batch,1,1)
        self.done_batch = torch.empty((max_size, 1, 1), device=self.device) # (batch,1,1)
        ###self.value_prime_batch = torch.empty((max_size, 1, 1), device=self.device) # (batch,1,1)
        ###self.target_v_batch = torch.empty((max_size, 1, 1), device=self.device)
        
        self.budget_inputs_batch = torch.empty((max_size, sample_size+2, 1), device=self.device)
        self.LSTM_h_batch = torch.empty((max_size, 1, 128), device=self.device)
        self.LSTM_c_batch = torch.empty((max_size, 1, 128), device=self.device)
        self.mask_batch = torch.empty((max_size, sample_size+2, 20), dtype=torch.int64, device=self.device)
        self.pos_encoding_batch = torch.empty((max_size, sample_size+2, 32), device=self.device)
        
        self.experience_buffer = []
        for i in range(13):
            self.experience_buffer.append([])

    def _update(self, items: [Tensor]):
        states, actions, rewards, undones = items
        p = self.p + rewards.shape[0]  # pointer
        if p > self.max_size:
            self.if_full = True
            p0 = self.p
            p1 = self.max_size
            p2 = self.max_size - self.p
            p = p - self.max_size
            ### 当前数量:最大容量  0:超出部分        =  新加部分填满   溢出部分填到buffer开头
            self.states[p0:p1], self.states[0:p] = states[:p2], states[-p:]
            self.actions[p0:p1], self.actions[0:p] = actions[:p2], actions[-p:]
            self.rewards[p0:p1], self.rewards[0:p] = rewards[:p2], rewards[-p:]
            self.undones[p0:p1], self.undones[0:p] = undones[:p2], undones[-p:]
        else:
            self.states[self.p:p] = states
            self.actions[self.p:p] = actions
            self.rewards[self.p:p] = rewards
            self.undones[self.p:p] = undones
        self.p = p
        self.cur_size = self.max_size if self.if_full else self.p
        
    def update(self, items: [list]):
        ##states, actions, rewards, undones = items
        p = self.p + len(items[0])  # pointer
        
        node_inputs_batch = torch.stack(items[0], dim=0) # (batch,sample_size+2,2)
        edge_inputs_batch = torch.stack(items[1], dim=0) # (batch,sample_size+2,k_size)
        current_inputs_batch = torch.stack(items[2], dim=0) # (batch,1,1)
        action_batch = torch.stack(items[3], dim=0) # (batch,1,1)
        
        done_batch = torch.stack(items[4], dim=0) # (batch,1,1)
        reward_batch = torch.stack(items[5], dim=0) # (batch,1,1)
        ###value_prime_batch = torch.stack(items[6], dim=0) # (batch,1,1)
        ###target_v_batch = torch.stack(items[7])
        
        budget_inputs_batch = torch.stack(items[6], dim=0)
        LSTM_h_batch = torch.stack(items[7])
        LSTM_c_batch = torch.stack(items[8])
        mask_batch = torch.stack(items[9])
        pos_encoding_batch = torch.stack(items[10])
        
        print("----------------------------action_batch.size:{}---------------------------------------".format(action_batch.size()))
        print("----------------------------done_batch.size:{}---------------------------------------".format(done_batch.size()))
        
        if p > self.max_size:
            self.if_full = True
            p0 = self.p
            p1 = self.max_size
            p2 = self.max_size - self.p
            p = p - self.max_size
            
            self.node_inputs_batch[p0:p1], self.node_inputs_batch[0:p] = \
                 node_inputs_batch[:p2], node_inputs_batch[-p:]
            self.edge_inputs_batch[p0:p1], self.edge_inputs_batch[0:p] = \
                 edge_inputs_batch[:p2], edge_inputs_batch[-p:]
            self.current_inputs_batch[p0:p1], self.current_inputs_batch[0:p] = \
                 current_inputs_batch[:p2], current_inputs_batch[-p:]
            self.action_batch[p0:p1], self.action_batch[0:p] = \
                 action_batch[:p2], action_batch[-p:]
            ##self.value_batch[p0:p1], self.value_batch[0:p] = \
            ##     value_batch[:p2], value_batch[-p:]
            self.reward_batch[p0:p1], self.reward_batch[0:p] = \
                 reward_batch[:p2], reward_batch[-p:]
            self.done_batch[p0:p1], self.done_batch[0:p] = \
                 done_batch[:p2], done_batch[-p:]    
            ##self.value_prime_batch[p0:p1], self.value_prime_batch[0:p] = \
            ##     value_prime_batch[:p2], value_prime_batch[-p:]
            ##self.target_v_batch[p0:p1], self.target_v_batch[0:p] = \
            ##     target_v_batch[:p2], target_v_batch[-p:]
            self.budget_inputs_batch[p0:p1], self.budget_inputs_batch[0:p] = \
                 budget_inputs_batch[:p2], budget_inputs_batch[-p:]
            self.LSTM_h_batch[p0:p1], self.LSTM_h_batch[0:p] = \
                 LSTM_h_batch[:p2], LSTM_h_batch[-p:]
            self.LSTM_c_batch[p0:p1], self.LSTM_c_batch[0:p] = \
                 LSTM_c_batch[:p2], LSTM_c_batch[-p:]
            self.mask_batch[p0:p1], self.mask_batch[0:p] = \
                 mask_batch[:p2], mask_batch[-p:]
            self.pos_encoding_batch[p0:p1], self.pos_encoding_batch[0:p] = \
                 pos_encoding_batch[:p2], pos_encoding_batch[-p:]
                 
            ###for i in range(items_len):
            ###    self.experience_buffer[i][p0:p1], self.experience_buffer[i][0:p] = \
            ###         items[i][:p2], items[i][-p:]
        else:
            ##self.states[self.p:p] = states
            ##self.actions[self.p:p] = actions
            ##self.rewards[self.p:p] = rewards
            ##self.undones[self.p:p] = undones
            
            self.node_inputs_batch[self.p:p] = node_inputs_batch
            self.edge_inputs_batch[self.p:p] = edge_inputs_batch
            self.current_inputs_batch[self.p:p] = current_inputs_batch
            self.action_batch[self.p:p] = action_batch
            #self.value_batch[self.p:p] = value_batch
            self.reward_batch[self.p:p] = reward_batch
            self.done_batch[self.p:p] = done_batch
            #self.value_prime_batch[self.p:p] = value_prime_batch
            #self.target_v_batch[self.p:p] = target_v_batch
            self.budget_inputs_batch[self.p:p] = budget_inputs_batch
            self.LSTM_h_batch[self.p:p] = LSTM_h_batch
            self.LSTM_c_batch[self.p:p] = LSTM_c_batch
            self.mask_batch[self.p:p] = mask_batch
            self.pos_encoding_batch[self.p:p] = pos_encoding_batch
            
            ##for i in range(items_len):
            ##    self.experience_buffer[i][self.p:p] += items[i]
            
        self.p = p
        self.cur_size = self.max_size if self.if_full else self.p

    def sample(self, batch_size: int) -> [Tensor]:
        ids = torch.randint(self.cur_size - 1, size=(batch_size,), requires_grad=False)
        ##return self.node_inputs_batch[ids], \
        ##       self.edge_inputs_batch[ids], \
        ##       self.current_inputs_batch[ids], \
        ##       self.budget_inputs_batch[ids], \
        ##       self.LSTM_h_batch[ids], \
        ##       self.LSTM_c_batch[ids], \
        ##       self.mask_batch[ids], \
        ##       self.pos_encoding_batch[ids], \
        ##       self.node_inputs_batch[ids + 1], \
        ##       self.edge_inputs_batch[ids + 1], \
        ##       self.current_inputs_batch[ids + 1], \
        ##       self.budget_inputs_batch[ids + 1], \
        ##       self.LSTM_h_batch[ids + 1], \
        ##       self.LSTM_c_batch[ids + 1], \
        ##       self.mask_batch[ids + 1], \
        ##       self.pos_encoding_batch[ids + 1], \
        ##       self.action_batch[ids], \
        ##       self.reward_batch[ids], \
        ##       self.done_batch[ids]
        node_inputs_batch      = self.node_inputs_batch[ids]
        edge_inputs_batch      = self.edge_inputs_batch[ids]
        current_inputs_batch   = self.current_inputs_batch[ids]
        budget_inputs_batch    = self.budget_inputs_batch[ids]
        LSTM_h_batch           = self.LSTM_h_batch[ids]
        LSTM_c_batch           = self.LSTM_c_batch[ids]
        mask_batch             = self.mask_batch[ids]
        pos_encoding_batch     = self.pos_encoding_batch[ids]
        node_inputs_batch_n    = self.node_inputs_batch[ids + 1]
        edge_inputs_batch_n    = self.edge_inputs_batch[ids + 1]
        current_inputs_batch_n = self.current_inputs_batch[ids + 1]
        budget_inputs_batch_n  = self.budget_inputs_batch[ids + 1]
        LSTM_h_batch_n         = self.LSTM_h_batch[ids + 1]
        LSTM_c_batch_n         = self.LSTM_c_batch[ids + 1]
        mask_batch_n           = self.mask_batch[ids + 1]
        pos_encoding_batch_n   = self.pos_encoding_batch[ids + 1]
        action_batch           = self.action_batch[ids]
        reward_batch           = self.reward_batch[ids]
        done_batch             = self.done_batch[ids]
               
        return node_inputs_batch.to(self.device_gpu), \
               edge_inputs_batch.to(self.device_gpu), \
               current_inputs_batch.to(self.device_gpu), \
               budget_inputs_batch.to(self.device_gpu), \
               LSTM_h_batch.to(self.device_gpu), \
               LSTM_c_batch.to(self.device_gpu), \
               mask_batch.to(self.device_gpu), \
               pos_encoding_batch.to(self.device_gpu), \
               node_inputs_batch_n.to(self.device_gpu), \
               edge_inputs_batch_n.to(self.device_gpu), \
               current_inputs_batch_n.to(self.device_gpu), \
               budget_inputs_batch_n.to(self.device_gpu), \
               LSTM_h_batch_n.to(self.device_gpu), \
               LSTM_c_batch_n.to(self.device_gpu), \
               mask_batch_n.to(self.device_gpu), \
               pos_encoding_batch_n.to(self.device_gpu), \
               action_batch.to(self.device_gpu), \
               reward_batch.to(self.device_gpu), \
               done_batch.to(self.device_gpu)
   
   
               
               ##node_inputs_batch, edge_inputs_batch, current_inputs_batch, budget_inputs_batch, \
               ##    LSTM_h_batch, LSTM_c_batch, mask_batch, pos_encoding_batch, \
               ##node_inputs_batch_n, edge_inputs_batch_n, current_inputs_batch_n, budget_inputs_batch_n, \
               ##    LSTM_h_batch_n, LSTM_c_batch_n, mask_batch_n, pos_encoding_batch_n, \
               ##action_batch, reward_batch, value_batch = buffer.sample(BATCH_SIZE)
        
        ##return self.node_inputs_batch[ids], self.edge_inputs_batch[ids], self.current_inputs_batch[ids], \
        ##       self.action_batch[ids], self.value_batch[ids], self.reward_batch[ids], \
        ##       self.value_prime_batch[ids], self.target_v_batch[ids], self.budget_inputs_batch[ids], \
        ##       self.LSTM_h_batch[ids], self.LSTM_c_batch[ids], self.mask_batch[ids], self.pos_encoding_batch[ids]
