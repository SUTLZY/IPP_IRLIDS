import copy
import os

import imageio
import numpy as np
import torch
#from env import Env
from attention_net_dqn_new import AttentionNet
from parameters_dqn_new import *
import scipy.signal as signal

from scipy.optimize import minimize

import sys
sys.path.append("..")
from env import Env
#from env_copy import Env

def discount(x, gamma):
    return signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

### 使用凸优化实现的IDS环节
### X:pi
### A:regret
### B:Q的方差

def softmax(x):
    """Compute the softmax of vector x."""
    exp_x = np.exp(x - np.max(x))  # 减去最大值是为了数值稳定性
    return exp_x / exp_x.sum()

def softmax_with_scaling(x, scale):
    #e_x = np.exp(x - np.max(x))
    #scaled_softmax = (e_x / np.sum(e_x)) * scale
    e_x = np.exp(scale * (x - np.max(x)))
    scaled_softmax = e_x / np.sum(e_x)
    
    return scaled_softmax

class ConvexOptimizer:
    # ze
    def __init__(self, regret, Q_var):
        self.A = regret 
        self.b = Q_var  
        
    def objective(self, x):
        # 定义目标函数
        #return x[0]**2 + x[1]**2
        a = np.sum((x.dot(self.A))**2) / np.sum(x * self.b)
        b = np.sum(x.dot(self.A))
        #return np.sum((x.dot(self.A))**2) / np.sum(x * self.b) #+np.sum(x.dot(self.A))
        #print("001>>>>>>>>>>>>>>>>>>>>>>>>>a:", a)
        #print("002>>>>>>>>>>>>>>>>>>>>>>>>>b:", b)
        #
        return a #+ LAMDA*b
    
    def constraints(self, x):
        # 定义线性约束条件
        #return np.dot(self.A, x) -ze self.b
        return np.sum(x) - 1

    def optimize(self):
        # 初始化猜测值
        n = len(self.A)
        x0 = np.ones(n) / n
        #print("04________________n:", n)
        #print("03________________len(x0):", len(x0))
        objective = self.objective
        # 定义约束条件
        constraint = self.constraints
        cons = [{'type': 'eq', 'fun': constraint},
                {'type': 'ineq', 'fun': lambda x: x}]
        # 调用 minimize 函数进行优化
        bounds = [(0, None) for i in range(n)]
        self.result = minimize(objective, x0, constraints=cons, method='SLSQP', bounds=bounds)
        # 处理优化结果
        ###if result.success:
        ###    print("优化成功！")
        ###    print("最优解：", result.x)
        ###    print("目标函数值：", result.fun)
        ###else:
        ###    print("优化失败！")


class Worker:
    #def __init__(self, metaAgentID, localNetwork, global_step, budget_range, sample_size=SAMPLE_SIZE, sample_length=None, device='cuda', greedy=False, save_image=False):
    def __init__(self, metaAgentID, localNetwork, global_step, budget_range, sample_size=SAMPLE_SIZE, sample_length=None, device='cuda', greedy=False, save_image=False, random = True):

        self.device = device
        self.greedy = greedy
        self.metaAgentID = metaAgentID
        self.global_step = global_step
        self.save_image = save_image
        self.sample_length = sample_length
        self.sample_size = sample_size

        self.env = Env(sample_size=self.sample_size, k_size=K_SIZE, budget_range=budget_range, save_image=self.save_image)
        # self.local_net = AttentionNet(2, 128, device=self.device)
        # self.local_net.to(device)
        self.local_net = localNetwork
        self.experience = None
        self.random = random
        #self.optimizer = ConvexOptimizer()  # 创建凸优化器实例

    def run_episode(self, currEpisode):
        idsmax_NUM = []
        episode_buffer = []
        perf_metrics = dict()
        for i in range(11):
            episode_buffer.append([])

        done = False
        node_coords, graph, node_info, node_std, budget = self.env.reset()
        
        n_nodes = node_coords.shape[0]
        node_info_inputs = node_info.reshape((n_nodes, 1))
        node_std_inputs = node_std.reshape((n_nodes,1))
        budget_inputs = self.calc_estimate_budget(budget, current_idx=1)
        node_inputs = np.concatenate((node_coords, node_info_inputs, node_std_inputs), axis=1)
        node_inputs = torch.FloatTensor(node_inputs).unsqueeze(0).to(self.device) # (1, sample_size+2, 4)
        budget_inputs = torch.FloatTensor(budget_inputs).unsqueeze(0).to(self.device) # (1, sample_size+2, 1)
        
        graph = list(graph.values())
        edge_inputs = []
        for node in graph:
            node_edges = list(map(int, node))
            edge_inputs.append(node_edges)

        pos_encoding = self.calculate_position_embedding(edge_inputs)
        pos_encoding = torch.from_numpy(pos_encoding).float().unsqueeze(0).to(self.device) # (1, sample_size+2, 32)

        edge_inputs = torch.tensor(edge_inputs).unsqueeze(0).to(self.device) # (1, sample_size+2, k_size)

        current_index = torch.tensor([self.env.current_node_index]).unsqueeze(0).unsqueeze(0).to(self.device) # (1,1,1)
        route = [current_index.item()]

        LSTM_h = torch.zeros((1,1,EMBEDDING_DIM)).to(self.device)
        LSTM_c = torch.zeros((1,1,EMBEDDING_DIM)).to(self.device)

        mask = torch.zeros((1, self.sample_size+2, K_SIZE), dtype=torch.int64).to(self.device)
        for i in range(256):
            #episode_buffer[9] += LSTM_h
            #episode_buffer[10] += LSTM_c
            #episode_buffer[11] += mask
            #episode_buffer[12] += pos_encoding
            episode_buffer[7] += LSTM_h
            episode_buffer[8] += LSTM_c
            episode_buffer[9] += mask
            episode_buffer[10] += pos_encoding

            '''
            ####with torch.no_grad():
            ####    ##logp_list, value, LSTM_h, LSTM_c = self.local_net(node_inputs, edge_inputs, budget_inputs, current_index, LSTM_h, LSTM_c, pos_encoding, mask)
            ####    logp_list, value, LSTM_h, LSTM_c = self.local_net(node_inputs, edge_inputs, budget_inputs, current_index, LSTM_h, LSTM_c, pos_encoding, mask)
            #print("01=======================logp_list.size:{}=======================".format(logp_list.size()))
            #                                logp_list.size:torch.Size([1, 20])
            # next_node (1), logp_list (1, 10), value (1,1,1)
            ##########if self.greedy:
            ##########    action_index = torch.argmax(logp_list, dim=1).long()
            ##########else:
            ##########    #action_index = torch.multinomial(logp_list.exp(), 1).long().squeeze(1)
            #action_index = torch.randint(20, size=(1,))
            '''
            if self.random:
                with torch.no_grad():
                    if IF_AE:
                        logp_list, Q_, LSTM_h, LSTM_c, _, _ = self.local_net(node_inputs, edge_inputs, budget_inputs, current_index, LSTM_h, LSTM_c, pos_encoding, mask)
                    elif CHANGE_DQN_NET:
                        Q_, LSTM_h, LSTM_c = self.local_net(node_inputs, edge_inputs, budget_inputs, current_index, LSTM_h, LSTM_c, pos_encoding, mask)
                    else:
                        if RAND_POLICY:
                            _, Q_, LSTM_h, LSTM_c, validp_list = self.local_net(node_inputs, edge_inputs, budget_inputs, current_index, LSTM_h, LSTM_c, pos_encoding, mask)
                        else:
                            logp_list, Q_, LSTM_h, LSTM_c, _ = self.local_net(node_inputs, edge_inputs, budget_inputs, current_index, LSTM_h, LSTM_c, pos_encoding, mask)
                #action_index = torch.randint(20, size=(1,)).long()
                if CHANGE_DQN_NET:
                    #print("001*****---++++++++++++++****************Q_.size():",Q_.size())
                    action_index = torch.argmax(Q_, dim=2).squeeze(0).long()
                    #print("001*****---++++++++++++++****************action_index.size():",action_index.size())
                elif RAND_POLICY:
                    validQ_list = Q_.squeeze(0).squeeze(0)[validp_list]
                    validQ_list = validQ_list
                    action_index = self.act_train_non_ids(validQ_list, validp_list[0])
                else:
                    action_index = torch.argmax(logp_list, dim=1).long()
                
            else:
                if IF_IDS:
                    with torch.no_grad():
                        if IF_AE:
                            _, Q_, LSTM_h, LSTM_c, _, _ = self.local_net(node_inputs, edge_inputs, budget_inputs, current_index, LSTM_h, LSTM_c, pos_encoding, mask)
                        else:
                            if CAT_POLICY:
                                logp_list, Q_, LSTM_h, LSTM_c, validp_list = self.local_net(node_inputs, edge_inputs, budget_inputs, current_index, LSTM_h, LSTM_c, pos_encoding, mask)
                            elif CHANGE_DQN_NET:
                                Q_, LSTM_h, LSTM_c = self.local_net(node_inputs, edge_inputs, budget_inputs, current_index, LSTM_h, LSTM_c, pos_encoding, mask)
                            else:
                                _, Q_, LSTM_h, LSTM_c, validp_list = self.local_net(node_inputs, edge_inputs, budget_inputs, current_index, LSTM_h, LSTM_c, pos_encoding, mask)
                        #print("01>>>>>>>>>>>Q_ =",Q_)
                        '''
                        validQ_list = Q_.squeeze(0).squeeze(0)[validp_list]
                        validQ_list = validQ_list.unsqueeze(0).unsqueeze(0)
                        action_index = self.act_train(validQ_list, validp_list)
                        '''
                        if CAT_POLICY:
                            action_index = torch.argmax(logp_list, dim=1).long()
                        elif NON_IDS: # 不带IDS的DQN
                            if CHANGE_DQN_NET:
                                #print("001*****---++++++++++++++****************Q_.size():",Q_.size())
                                action_index = self.act_train_DQN(Q_)
                                #print("002*****---++++++++++++++****************action_index.size():",action_index.size())  
                            else:
                                validQ_list = Q_.squeeze(0).squeeze(0)[validp_list]
                                validQ_list = validQ_list
                                action_index = self.act_train_non_ids(validQ_list, validp_list[0])
                        else: # 带IDS的DQN
                            if CHANGE_DQN_NET:
                                action_index, idsmax_num = self.act_train_DQN_IDS(Q_)
                                idsmax_NUM.append(idsmax_num)
                            else:
                                validQ_list = Q_.squeeze(0).squeeze(0)[validp_list]
                                validQ_list = validQ_list.unsqueeze(0).unsqueeze(0)
                                action_index = self.act_train(validQ_list, validp_list)
                        #print("001*****---++++++++++++++****************action_index.size():", action_index.size())
                        #print("002*****---++++++++++++++****************action_index:", action_index)
                        
                else:
                    with torch.no_grad():
                        action_index, LSTM_h, LSTM_c = self.local_net.get_action(node_inputs, edge_inputs, budget_inputs, current_index, LSTM_h, LSTM_c, pos_encoding, mask)
            '''
            #print("01===================action_index.type:", type(action_index))
            #print("02--------------------action_index:", action_index)
            #######with torch.no_grad():
            #######    logp_list, Q_, LSTM_h, LSTM_c = self.local_net(node_inputs, edge_inputs, budget_inputs, current_index, LSTM_h, LSTM_c, pos_encoding, mask)
            #######Q_ = Q_.squeeze(1)
            #######action_index = torch.argmax(Q_, dim=1).long()
            
            #####with torch.no_grad():
            #####    action_index, LSTM_h, LSTM_c = self.local_net.get_action(node_inputs, edge_inputs, budget_inputs, current_index, LSTM_h, LSTM_c, pos_encoding, mask)

            #print("01>>>>>>>>>>>>>>>>>>>>action_index.unsqueeze(0).unsqueeze(0) =",action_index.unsqueeze(0).unsqueeze(0))
            #print("--------------------action_index:{}--------------------".format(action_index))
            #print("01=======================action_index.size:{}=======================".format(action_index.size()))
            #print("02=======================action_index.size:{}=======================".format(action_index.unsqueeze(0).unsqueeze(0).size()))
            #print("01=======================value.size:{}=======================".format(value.size()))
            #print("02=======================logp_list.size:{}=======================".format(logp_list.size()))
            #print("02=======================action_index:{}=======================".format(action_index))
            '''
            episode_buffer[0] += node_inputs
            episode_buffer[1] += edge_inputs
            episode_buffer[2] += current_index
            episode_buffer[3] += action_index.unsqueeze(0).unsqueeze(0)
            #print("03=======================episode_buffer[3]:", episode_buffer[3])
            #episode_buffer[4] += value
            #episode_buffer[8] += budget_inputs 
            episode_buffer[6] += budget_inputs 
            #print("02=======================action_index.size:{}=======================".format(action_index.size()))
            #print("01>>>>>>>>>>>>>>>>>>>>action_index.unsqueeze(0).unsqueeze(0) =",action_index.unsqueeze(0).unsqueeze(0))
            #######print("02>>>>>>>>>>>>>>>>>>>>done_batch =",torch.FloatTensor([[[0]]]))
            next_node_index = edge_inputs[:, current_index.item(), action_index.item()]
            route.append(next_node_index.item())
            reward, done, node_info, node_std, remain_budget = self.env.step(next_node_index.item(), self.sample_length)
            '''
            #if (not done and i==127):
                #reward += -np.linalg.norm(self.env.node_coords[self.env.current_node_index,:]-self.env.node_coords[0,:])

            ##print("===============type(reward) :", type(reward))
            #yxz = torch.FloatTensor([[[reward]]])
            ###print("===============type(reward) :", type(yxz))
            #print("===============reward.size :", yxz.size())
            '''
            episode_buffer[5] += torch.FloatTensor([[[reward]]]).to(self.device)
       

            current_index = next_node_index.unsqueeze(0).unsqueeze(0)
            node_info_inputs = node_info.reshape(n_nodes, 1)
            node_std_inputs = node_std.reshape(n_nodes, 1)
            budget_inputs = self.calc_estimate_budget(remain_budget, current_idx=current_index.item())
            node_inputs = np.concatenate((node_coords, node_info_inputs, node_std_inputs), axis=1)
            node_inputs = torch.FloatTensor(node_inputs).unsqueeze(0).to(self.device)
            budget_inputs = torch.FloatTensor(budget_inputs).unsqueeze(0).to(self.device)
            #print(node_inputs)
            
            # mask last five node
            mask = torch.zeros((1, self.sample_size+2, K_SIZE), dtype=torch.int64).to(self.device)
            '''
            #connected_nodes = edge_inputs[0, current_index.item()]
            #current_edge = torch.gather(edge_inputs, 1, current_index.repeat(1, 1, K_SIZE))
            #current_edge = current_edge.permute(0, 2, 1)
            #connected_nodes_budget = torch.gather(budget_inputs, 1, current_edge) # (1, k_size, 1)
            #n_available_node = sum(int(x>0) for x in connected_nodes_budget.squeeze(0))
            #if n_available_node > 5:
            #    for j, node in enumerate(connected_nodes.squeeze(0)):
            #        if node.item() in route[-2:]:
            #            mask[0, route[-1], j] = 1
            '''

            # save a frame
            if self.save_image:
                if not os.path.exists(gifs_path):
                    os.makedirs(gifs_path)
                self.env.plot(route, self.global_step, i, gifs_path)

            if not done:
                episode_buffer[4] += torch.FloatTensor([[[1]]]).to(self.device)

            if done:
                #episode_buffer[6] = episode_buffer[4][1:]
                #episode_buffer[6].append(torch.FloatTensor([[0]]).to(self.device))
                episode_buffer[4] += torch.FloatTensor([[[0]]]).to(self.device)
                if self.env.current_node_index == 0:
                    perf_metrics['remain_budget'] = remain_budget / budget
                    #perf_metrics['collect_info'] = 1 - remain_info.sum()
                    perf_metrics['RMSE'] = self.env.gp_ipp.evaluate_RMSE(self.env.ground_truth)
                    perf_metrics['F1Score'] = self.env.gp_ipp.evaluate_F1score(self.env.ground_truth)
                    perf_metrics['delta_cov_trace'] = self.env.cov_trace0 - self.env.cov_trace
                    perf_metrics['MI'] = self.env.gp_ipp.evaluate_mutual_info(self.env.high_info_area)
                    perf_metrics['cov_trace'] = self.env.cov_trace
                    perf_metrics['success_rate'] = True
                    cur_reward = sum(episode_buffer[5])/len(episode_buffer[5]) # <class 'torch.Tensor'>
                    cur_reward = cur_reward.squeeze(0).squeeze(0)
                    cur_reward = np.array(cur_reward).astype(np.float64)
                    perf_metrics['cur_reward'] = cur_reward
                    if not NON_IDS:
                        perf_metrics['idsmax_num'] = np.mean(idsmax_NUM) #sum(idsmax_NUM)/len(idsmax_NUM)
                    print('{} Goodbye world! We did it!'.format(i))
                else:
                    perf_metrics['remain_budget'] = np.nan
                    perf_metrics['RMSE'] = self.env.gp_ipp.evaluate_RMSE(self.env.ground_truth)
                    perf_metrics['F1Score'] = self.env.gp_ipp.evaluate_F1score(self.env.ground_truth)
                    perf_metrics['delta_cov_trace'] = self.env.cov_trace0 - self.env.cov_trace
                    perf_metrics['MI'] = self.env.gp_ipp.evaluate_MI(self.env.high_info_area)
                    perf_metrics['cov_trace'] = self.env.cov_trace
                    perf_metrics['success_rate'] = False
                    cur_reward = sum(episode_buffer[5])/len(episode_buffer[5]) # <class 'torch.Tensor'>
                    cur_reward = cur_reward.squeeze(0).squeeze(0)
                    cur_reward = np.array(cur_reward).astype(np.float64)
                    perf_metrics['cur_reward'] = cur_reward
                    if not NON_IDS:
                        perf_metrics['idsmax_num'] = np.mean(idsmax_NUM) #sum(idsmax_NUM)/len(idsmax_NUM)
                    print('{} Overbudget!'.format(i))
                break
        if not done:
            #episode_buffer[6] = episode_buffer[4][1:]
            with torch.no_grad():
                if IF_AE:
                    _, _, LSTM_h, LSTM_c, _, _ = self.local_net(node_inputs, edge_inputs, budget_inputs, current_index, LSTM_h, LSTM_c, pos_encoding, mask)
                else:
                    if CHANGE_DQN_NET:
                        _, LSTM_h, LSTM_c = self.local_net(node_inputs, edge_inputs, budget_inputs, current_index, LSTM_h, LSTM_c, pos_encoding, mask)
                    else:
                        _, _, LSTM_h, LSTM_c, _ = self.local_net(node_inputs, edge_inputs, budget_inputs, current_index, LSTM_h, LSTM_c, pos_encoding, mask)
                 # _, value, LSTM_h, LSTM_c = self.local_net(node_inputs, edge_inputs, budget_inputs, current_index, LSTM_h, LSTM_c, pos_encoding, mask)
            #episode_buffer[6].append(value.squeeze(0))
            ##########episode_buffer[4] += torch.FloatTensor([[[1]]]).to(self.device)
            perf_metrics['remain_budget'] = remain_budget / budget
            perf_metrics['RMSE'] = self.env.gp_ipp.evaluate_RMSE(self.env.ground_truth)
            perf_metrics['F1Score'] = self.env.gp_ipp.evaluate_F1score(self.env.ground_truth)
            perf_metrics['delta_cov_trace'] =  self.env.cov_trace0 - self.env.cov_trace
            perf_metrics['MI'] = self.env.gp_ipp.evaluate_mutual_info(self.env.high_info_area)
            perf_metrics['cov_trace'] = self.env.cov_trace
            perf_metrics['success_rate'] = False
            cur_reward = sum(episode_buffer[5])/len(episode_buffer[5]) # <class 'torch.Tensor'>
            cur_reward = cur_reward.squeeze(0).squeeze(0)
            cur_reward = np.array(cur_reward).astype(np.float64)
            perf_metrics['cur_reward'] = cur_reward
            if not NON_IDS:
                perf_metrics['idsmax_num'] = np.mean(idsmax_NUM) #sum(idsmax_NUM)/len(idsmax_NUM)


        print('route is ', route)
        '''
        ###reward = copy.deepcopy(episode_buffer[5])
        ###reward.append(episode_buffer[6][-1])
        ###for i in range(len(reward)):
        ###    reward[i] = reward[i].cpu().numpy()
        ###reward_plus = np.array(reward,dtype=object).reshape(-1)
        ###discounted_rewards = discount(reward_plus, GAMMA)[:-1]
        ###discounted_rewards = discounted_rewards.tolist()
        ###target_v = torch.FloatTensor(discounted_rewards).unsqueeze(1).unsqueeze(1).to(self.device)
        #####print("02============target_v.size = {}===========".format(target_v.size()))

        ###for i in range(target_v.size()[0]):
        ###    episode_buffer[7].append(target_v[i,:,:])
        '''

        # save gif
        if self.save_image:
            if self.greedy:
                from test_driver import result_path as path
            else:
                path = gifs_path
            self.make_gif(path, currEpisode)

        self.experience = episode_buffer
        return perf_metrics

    def work(self, currEpisode):
        '''
        Interacts with the environment. The agent gets either gradients or experience buffer
        '''
        self.currEpisode = currEpisode
        self.perf_metrics = self.run_episode(currEpisode)

    def calc_estimate_budget(self, budget, current_idx):
        all_budget = []
        current_coord = self.env.node_coords[current_idx]
        end_coord = self.env.node_coords[0]
        for i, point_coord in enumerate(self.env.node_coords):
            dist_current2point = self.env.prm.calcDistance(current_coord, point_coord)
            dist_point2end = self.env.prm.calcDistance(point_coord, end_coord)
            estimate_budget = (budget - dist_current2point - dist_point2end) / 10
            # estimate_budget = (budget - dist_current2point - dist_point2end) / budget
            all_budget.append(estimate_budget)
        return np.asarray(all_budget).reshape(i+1, 1)

    
    def calculate_position_embedding(self, edge_inputs):
        A_matrix = np.zeros((self.sample_size+2, self.sample_size+2))
        D_matrix = np.zeros((self.sample_size+2, self.sample_size+2))
        for i in range(self.sample_size+2):
            for j in range(self.sample_size+2):
                if j in edge_inputs[i] and i != j:
                    A_matrix[i][j] = 1.0
        for i in range(self.sample_size+2):
            D_matrix[i][i] = 1/np.sqrt(len(edge_inputs[i])-1)
        L = np.eye(self.sample_size+2) - np.matmul(D_matrix, A_matrix, D_matrix)
        eigen_values, eigen_vector = np.linalg.eig(L)
        idx = eigen_values.argsort()
        eigen_values, eigen_vector = eigen_values[idx], np.real(eigen_vector[:, idx])
        eigen_vector = eigen_vector[:,1:32+1]
        return eigen_vector
    
    def make_gif(self, path, n):
        with imageio.get_writer('{}/{}_cov_trace_{:.4g}.gif'.format(path, n, self.env.cov_trace), mode='I', duration=0.5) as writer:
            for frame in self.env.frame_files:
                image = imageio.imread(frame)
                writer.append_data(image)
        print('gif complete\n')

        # Remove files
        for filename in self.env.frame_files[:-1]:
            os.remove(filename)

    def _act_train(self, agent_net, name):
        mean      = torch.mean(agent_net, dim=1)
        zero_mean = agent_net - torch.unsqueeze(mean, dim=-2)
        var       = torch.mean(torch.square(zero_mean), dim=1)
        std       = torch.sqrt(var)
        regret    = torch.max(mean + self.n_stds * std, dim=-1, keepdim=True)[0]
        regret    = regret - (mean - self.n_stds * std)
        regret_sq = torch.square(regret)
        info_gain = torch.log(1 + var / self.rho2) + 1e-5
        ids_score = regret_sq / info_gain
        action    = torch.argmin(ids_score, dim=-1, keepdim=False, dtype=torch.int32, out=None)
        return action
        '''
        n_stds=0.1              # Uncertainty scale for computing regret
        self.n_stds = n_stds    # Number of standard deviations for computing uncertainty
        self.rho2   = 1.0**2    # Return distribution variance
        '''
    
    def act_train(self, agent_net, validp_list):
        #print("001>>>>>==========>>agent_net.size() =", agent_net.size())  torch.Size([1, 1, 19])
        agent_net = agent_net.squeeze(0).squeeze(0)
        #print("002>>>>>==========>>agent_net.size() =", agent_net.size())   torch.Size([19])
        
        mean      = torch.mean(agent_net, dim=-1)
        zero_mean = agent_net - mean
        #var       = torch.square(zero_mean)
        var       = torch.mean(torch.square(zero_mean))
        Q_max, _  = torch.max(agent_net, dim=-1)
        regret    = Q_max - agent_net
        optimizer = ConvexOptimizer(regret.detach().numpy(), var.detach().numpy())
        optimizer.optimize()
        result    = optimizer.result
        policy_pi = result.x
        policy_pi = torch.from_numpy(policy_pi)
        _ , max_index = torch.max(policy_pi, dim=-1)
        try:
            action    = validp_list[0][max_index]
        except IndexError as e:
            print("Error:", str(e))
            print("Variable values:")
            print("validp_list =", validp_list)
            print("max_index =", max_index)
            print("agent_net =", agent_net)
        return action.unsqueeze(0)
    
    def act_train_non_ids(self, validQ_list, validp_list):
        index_of_max = torch.argmax(validQ_list)
        action = validp_list[index_of_max].unsqueeze(0)
        
        return action
    
    def act_train_DQN_IDS(self, Q_Value):
        Q_Value = Q_Value.squeeze(0).squeeze(0)
        print("002-1>>>>>==========>>Q_Value.size() =", Q_Value.size())
        validp_list = torch.nonzero(Q_Value != -1e4)
        agent_net = Q_Value[validp_list]
        print("003-1>>>>>==========>>validp_list.size() =", validp_list.size())
        
        max_value_index = torch.argmax(agent_net)
        original_index_of_max = validp_list[max_value_index]
        
        validp_list = validp_list.squeeze(1)
        agent_net   = agent_net.squeeze(1)
        mean = torch.mean(agent_net, dim=-1)
        zero_mean = agent_net - mean
        #var       = torch.mean(torch.square(zero_mean))
        var       = torch.square(zero_mean).detach().numpy()
        #var       = softmax(var)
        var       = softmax_with_scaling(var,SOFTMAX_SCALE)
        #var       = var/np.sum(var)
        #Q_max, _  = torch.max(agent_net, dim=-1)
        Q_max     = 1.0
        regret    = Q_max - agent_net.detach().numpy()
        #optimizer = ConvexOptimizer(regret.detach().numpy(), var.detach().numpy())
        optimizer = ConvexOptimizer(regret, var)
        optimizer.optimize()
        result    = optimizer.result
        policy_pi = result.x
        policy_pi = torch.from_numpy(policy_pi)
        _ , max_index = torch.max(policy_pi, dim=-1)
        try:
            action    = validp_list[max_index].unsqueeze(0)
        except IndexError as e:
            print("Error:", str(e))
            print("Variable values:")
            print("validp_list =", validp_list)
            print("max_index =", max_index)
            print("agent_net =", agent_net)
        idsmax_num = 1 if action == original_index_of_max else 0
        
        return action, idsmax_num
    
    def act_train_DQN(self, Q_Value):
        #print("001>>>>>==========>>Q_Value.size() =", Q_Value.size())
        Q_Value = Q_Value.squeeze(0).squeeze(0)
        #print("002>>>>>==========>>Q_Value.size() =", Q_Value.size())
        non_minus_100_indices = torch.nonzero(Q_Value != -1e4)
        non_minus_100_values = Q_Value[non_minus_100_indices]
        print("002>>>>>==========>>non_minus_100_values.size() =", non_minus_100_values.size())
        
        max_value_index = torch.argmax(non_minus_100_values)
        #original_index_of_max = non_minus_100_indices[max_value_index]
        action = non_minus_100_indices[max_value_index]
        
        return action

    '''
    import numpy as np
    from scipy.optimize import minimize
    
    
    ### X:pi
    ### A:regret
    ### B:Q的方差
    
    # 定义目标函数和约束条件
    def objective(X, A, B): # 目标函数
        return np.sum((X.dot(A))**2) / np.sum(X * B)
    
    def constraint(X): # 约束
        return np.sum(X) - 1
    
    # 初始化变量
    ###n = len(X)
    n = len(B)
    X0 = np.ones(n) / n
    
    # 求解凸优化问题
    cons = [{'type': 'eq', 'fun': constraint},
            {'type': 'ineq', 'fun': lambda X: X}]
    res = minimize(objective, X0, args=(A, B), constraints=cons, method='SLSQP')
    
    # 最优解
    X_opt = res.x
    
    多出来：列出优化问题，定义，界是在森么推的
    
    '''
    
    
    ### 使用凸优化实现的IDS环节
    ### X:pi
    ### A:regret
    ### B:Q的方差
    #def objective(self, X, A, B):
    #    return np.sum((X.dot(A))**2) / np.sum(X * B)
    #def constraints(self, X):
    #    return np.sum(X) - 1


if __name__=='__main__':
    device = torch.device('cuda')
    localNetwork = AttentionNet(INPUT_DIM, EMBEDDING_DIM).cuda()
    worker = Worker(1, localNetwork, 0, budget_range=(4, 6), save_image=False, sample_length=0.05)
    worker.run_episode(0)




