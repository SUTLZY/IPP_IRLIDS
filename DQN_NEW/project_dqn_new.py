import copy
#  /home/dlut/lyn/code/cat-dqn/catnipp/DQN_NEW/project_dqn_new.py
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import ray
import os
import numpy as np
import random
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp.autocast_mode import autocast

from attention_net_dqn_new import AttentionNet
from runner_dqn_new import RLRunner
from parameters_dqn_new import *
from buffer_dqn_new import ReplayBuffer

#from torchsummary import summary

seed = SEED  # "Set your seed value"
random.seed(seed)      # Python built-in random module
np.random.seed(seed)   # Numpy module
# PyTorch module
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)  # "Also set this if you are using a GPU"
torch.backends.cudnn.deterministic = True  # "Also set this to ensure consistency"

ray.init()
#ray.init(address='auto')
print("Welcome to PRM-AN!")

writer = SummaryWriter(train_path)
if not os.path.exists(model_path):
    os.makedirs(model_path)
if not os.path.exists(gifs_path):
    os.makedirs(gifs_path)

global_step = None


def writeToTensorBoard(writer, tensorboardData, curr_episode, plotMeans=True):
    # each row in tensorboardData represents an episode
    # each column is a specific metric

    if plotMeans == True:
        #print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>tensorboardData:", tensorboardData)
        tensorboardData = np.array(tensorboardData)
        tensorboardData = list(np.nanmean(tensorboardData, axis=0))
        metric_name = ['remain_budget', 'success_rate', 'RMSE', 'delta_cov_trace', 'MI', 'F1Score', 'cov_trace']
        ###reward, value, policyLoss, valueLoss, entropy, gradNorm, returns, \
        ###    remain_budget, success_rate, RMSE, dct, MI, F1, cov_tr = tensorboardData
        #### [reward_batch_mean.cpu(), obj_critics, q_values, *perf_data]
        '''
        reward, obj_critics, q_values,\
            remain_budget, success_rate, RMSE, dct, MI, F1, cov_tr, cur_reward = tensorboardData
        '''
        if CHANGE_DQN_NET and not NON_IDS:
            reward, obj_critics, q_values,\
                remain_budget, success_rate, RMSE, dct, MI, F1, cov_tr, cur_reward, idsmax_per = tensorboardData
        else:
            reward, obj_critics, q_values,\
                remain_budget, success_rate, RMSE, dct, MI, F1, cov_tr, cur_reward = tensorboardData
    else:
        reward, value, policyLoss, valueLoss, entropy, gradNorm, returns, \
            remain_budget, success_rate, RMSE, dct, MI, F1, cov_tr = tensorboardData
            
    writer.add_scalar(tag='Losses/obj_critics', scalar_value=obj_critics, global_step=curr_episode)
    writer.add_scalar(tag='Perf/q_values', scalar_value=q_values, global_step=curr_episode)
    #writer.add_scalar(tag='Losses/Value', scalar_value=value, global_step=curr_episode)
    #writer.add_scalar(tag='Losses/Policy Loss', scalar_value=policyLoss, global_step=curr_episode)
    #writer.add_scalar(tag='Losses/Value Loss', scalar_value=valueLoss, global_step=curr_episode)
    #writer.add_scalar(tag='Losses/Entropy', scalar_value=entropy, global_step=curr_episode)
    #writer.add_scalar(tag='Losses/Grad Norm', scalar_value=gradNorm, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Reward', scalar_value=reward, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Cur_Reward', scalar_value=cur_reward, global_step=curr_episode)
    if CHANGE_DQN_NET and not NON_IDS:
        writer.add_scalar(tag='Perf/IDSMAX_per', scalar_value=idsmax_per, global_step=curr_episode)
    #writer.add_scalar(tag='Perf/Returns', scalar_value=returns, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Remain Budget', scalar_value=remain_budget, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Success Rate', scalar_value=success_rate, global_step=curr_episode)
    writer.add_scalar(tag='Perf/RMSE', scalar_value=RMSE, global_step=curr_episode)
    writer.add_scalar(tag='Perf/F1 Score', scalar_value=F1, global_step=curr_episode)
    writer.add_scalar(tag='GP/MI', scalar_value=MI, global_step=curr_episode)
    writer.add_scalar(tag='GP/Delta Cov Trace', scalar_value=dct, global_step=curr_episode)
    writer.add_scalar(tag='GP/Cov Trace', scalar_value=cov_tr, global_step=curr_episode)

#@staticmethod
def soft_update(target_net: torch.nn.Module, current_net: torch.nn.Module, tau: float):
    # assert target_net is not current_net
    for tar, cur in zip(target_net.parameters(), current_net.parameters()):
        tar.data.copy_(cur.data * tau + tar.data * (1.0 - tau))

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = str(CUDA_DEVICE)[1:-1]
    device = torch.device('cuda') if USE_GPU_GLOBAL else torch.device('cpu')
    local_device = torch.device('cuda') if USE_GPU else torch.device('cpu')
    
    #global_network = AttentionNet(INPUT_DIM, EMBEDDING_DIM).to(device)
    eval_net = AttentionNet(INPUT_DIM, EMBEDDING_DIM).to(device)
    target_net = AttentionNet(INPUT_DIM, EMBEDDING_DIM).to(device)
    buffer = ReplayBuffer(max_size=BUFFER_SIZE, sample_size=SAMPLE_SIZE)
    
    # global_network.share_memory()
    global_optimizer = optim.Adam(eval_net.parameters(), lr=LR)
    lr_decay = optim.lr_scheduler.StepLR(global_optimizer, step_size=DECAY_STEP, gamma=0.96)
    ##criterion = torch.nn.SmoothL1Loss()
    ##ae_loss = nn.BCEWithLogitsLoss()
    # Automatically logs gradients of pytorch model
    #wandb.watch(global_network, log_freq = SUMMARY_WINDOW)

    best_perf = 900
    curr_episode = 0
    if LOAD_MODEL:
        print('Loading Model...')
        checkpoint = torch.load(model_path + '/checkpoint.pth')
        eval_net.load_state_dict(checkpoint['model'])
        global_optimizer.load_state_dict(checkpoint['optimizer'])
        lr_decay.load_state_dict(checkpoint['lr_decay'])
        curr_episode = checkpoint['episode']
        print("curr_episode set to ", curr_episode)

        best_model_checkpoint = torch.load(model_path + '/best_model_checkpoint.pth')
        best_perf = best_model_checkpoint['best_perf']
        print('best performance so far:', best_perf)
        print(global_optimizer.state_dict()['param_groups'][0]['lr'])

    # launch meta agents
    meta_agents = [RLRunner.remote(i) for i in range(NUM_META_AGENT)]

    # get initial weigths
    if device != local_device:
        weights = eval_net.to(local_device).state_dict()
        eval_net.to(device)
    else:
        weights = eval_net.state_dict()
        
    #summary(global_network, (4,), (128,))

    # launch the first job on each runner
    dp_model = nn.DataParallel(eval_net)

    # worker explore env to collect buffer
    try:
        while True:
            print("----------------------------explore env----------------------------")
            if_random = True
            if buffer.cur_size <= START_TRAIN_BUFFER_SIZE:
                jobList = []
                for i, meta_agent in enumerate(meta_agents):
                    jobList.append(meta_agent.job.remote(weights, curr_episode, BUDGET_RANGE, SAMPLE_SIZE, SAMPLE_LENGTH, if_random))
                    curr_episode += 1
                if not NON_IDS:
                    metric_name = ['remain_budget', 'success_rate', 'RMSE', 'delta_cov_trace', 'MI', 'F1Score', 'cov_trace', 'cur_reward', 'idsmax_num']
                else:
                    metric_name = ['remain_budget', 'success_rate', 'RMSE', 'delta_cov_trace', 'MI', 'F1Score', 'cov_trace', 'cur_reward']
                tensorboardData = []
                trainingData = []
                experience_buffer = []
                for i in range(11):
                    experience_buffer.append([])
                    
                ''' 开始收集数据 '''
                # wait for any job to be completed
                done_id, jobList = ray.wait(jobList, num_returns=NUM_META_AGENT)
                # get the results
                #jobResults, metrics, info = ray.get(done_id)[0]
                done_jobs = ray.get(done_id)
                random.shuffle(done_jobs)
                #done_jobs = list(reversed(done_jobs))
                perf_metrics = {}
                for n in metric_name:
                    perf_metrics[n] = []
                for job in done_jobs:
                    jobResults, metrics, info = job
                    for i in range(11):
                        experience_buffer[i] += jobResults[i]
                    for n in metric_name:
                        perf_metrics[n].append(metrics[n])
                        
                buffer.update(experience_buffer)
            print(">>>>>>>>>>>>buffer.cur_size ={}>>>>>>>>>>>>".format(buffer.cur_size))
            if buffer.cur_size > START_TRAIN_BUFFER_SIZE:
                break
    except KeyboardInterrupt:
        print("CTRL_C pressed. Killing remote workers")
        for a in meta_agents:
            ray.kill(a)
    
    # start training
    try:
        while True:
            if_random = False
            jobList = []
            #sample_size = np.random.randint(200,400)
            for i, meta_agent in enumerate(meta_agents):
                jobList.append(meta_agent.job.remote(weights, curr_episode, BUDGET_RANGE, SAMPLE_SIZE, SAMPLE_LENGTH, if_random))
                curr_episode += 1
            if not NON_IDS:
                metric_name = ['remain_budget', 'success_rate', 'RMSE', 'delta_cov_trace', 'MI', 'F1Score', 'cov_trace', 'cur_reward', 'idsmax_num']
            else:
                metric_name = ['remain_budget', 'success_rate', 'RMSE', 'delta_cov_trace', 'MI', 'F1Score', 'cov_trace', 'cur_reward']
            #metric_name = ['remain_budget', 'success_rate', 'RMSE', 'delta_cov_trace', 'MI', 'F1Score', 'cov_trace', 'cur_reward', 'idsmax_num']
            tensorboardData = []
            #trainingData = []
            experience_buffer = []
            for i in range(11):
                experience_buffer.append([])
                
            # wait for any job to be completed
            done_id, jobList = ray.wait(jobList, num_returns=NUM_META_AGENT)
            # get the results
            #jobResults, metrics, info = ray.get(done_id)[0]
            done_jobs = ray.get(done_id)
            random.shuffle(done_jobs)
            #done_jobs = list(reversed(done_jobs))
            perf_metrics = {}
            for n in metric_name:
                perf_metrics[n] = []
            for job in done_jobs:
                jobResults, metrics, info = job
                for i in range(11):
                    experience_buffer[i] += jobResults[i]
                for n in metric_name:
                    perf_metrics[n].append(metrics[n])
                    
            buffer.update(experience_buffer)
            
            if np.mean(perf_metrics['cov_trace']) < best_perf and curr_episode % 32 == 0:
                best_perf = np.mean(perf_metrics['cov_trace'])
                print('Saving best model', end='\n')
                checkpoint = {"model": eval_net.state_dict(),
                              "optimizer": global_optimizer.state_dict(),
                              "episode": curr_episode,
                              "lr_decay": lr_decay.state_dict(),
                              "best_perf": best_perf}
                path_checkpoint = "./" + model_path + "/best_model_checkpoint.pth"
                torch.save(checkpoint, path_checkpoint)
                print('Saved model', end='\n')
                
            print("----------------------------start training----------------------------")
            obj_critics = 0.0
            q_values = 0.0
            reward_batch_mean = 0.0
            scaler = GradScaler()
            #update_times = int(buffer.cur_size * REPEAT_TIMES / BATCH_SIZE)
            update_times = 16
            for i in range(update_times):
                with torch.no_grad():
                    node_inputs_batch, edge_inputs_batch, current_inputs_batch, budget_inputs_batch, \
                        LSTM_h_batch, LSTM_c_batch, mask_batch, pos_encoding_batch, \
                    node_inputs_batch_n, edge_inputs_batch_n, current_inputs_batch_n, budget_inputs_batch_n, \
                        LSTM_h_batch_n, LSTM_c_batch_n, mask_batch_n, pos_encoding_batch_n, \
                    action_batch, reward_batch, done_batch = buffer.sample(BATCH_SIZE)
                    
                    ##_, next_q, _, _ = target_net(node_inputs_batch_n, edge_inputs_batch_n, budget_inputs_batch_n, \
                    ##    current_inputs_batch_n, LSTM_h_batch_n, LSTM_c_batch_n, pos_encoding_batch_n, mask_batch_n).max(dim=2, keepdim=True)[0]
                    if not IF_AE:
                        if IF_IDS:
                            if CHANGE_DQN_NET:
                                next_q1, _, _ = target_net(node_inputs_batch_n, edge_inputs_batch_n, budget_inputs_batch_n, \
                                current_inputs_batch_n, LSTM_h_batch_n, LSTM_c_batch_n, pos_encoding_batch_n, mask_batch_n)
                            else:
                                _, next_q1, _, _, _ = target_net(node_inputs_batch_n, edge_inputs_batch_n, budget_inputs_batch_n, \
                                current_inputs_batch_n, LSTM_h_batch_n, LSTM_c_batch_n, pos_encoding_batch_n, mask_batch_n)
                            
                        else:
                            _, next_q1, _, _ = target_net(node_inputs_batch_n, edge_inputs_batch_n, budget_inputs_batch_n, \
                                current_inputs_batch_n, LSTM_h_batch_n, LSTM_c_batch_n, pos_encoding_batch_n, mask_batch_n)
                    else:
                        _, next_q1, _, _, _, _ = target_net(node_inputs_batch_n, edge_inputs_batch_n, budget_inputs_batch_n, \
                            current_inputs_batch_n, LSTM_h_batch_n, LSTM_c_batch_n, pos_encoding_batch_n, mask_batch_n)
                    next_q = next_q1.max(dim=2, keepdim=True)[0]

                    q_label =  reward_batch + done_batch * GAMMA * next_q
                    
                with autocast():
                    #_,q_value,_,_ = dp_model(node_inputs_batch, edge_inputs_batch, budget_inputs_batch, \
                    #    current_inputs_batch, LSTM_h_batch, LSTM_c_batch, pos_encoding_batch, mask_batch).gather(2, action_batch.long())
                    if not IF_AE:
                        if IF_IDS:
                            if CHANGE_DQN_NET:
                                q_value1,_,_ = dp_model(node_inputs_batch, edge_inputs_batch, budget_inputs_batch, \
                                current_inputs_batch, LSTM_h_batch, LSTM_c_batch, pos_encoding_batch, mask_batch)
                            else:
                                _,q_value1,_,_, _ = dp_model(node_inputs_batch, edge_inputs_batch, budget_inputs_batch, \
                                current_inputs_batch, LSTM_h_batch, LSTM_c_batch, pos_encoding_batch, mask_batch)
                        else:
                            _,q_value1,_,_ = dp_model(node_inputs_batch, edge_inputs_batch, budget_inputs_batch, \
                                current_inputs_batch, LSTM_h_batch, LSTM_c_batch, pos_encoding_batch, mask_batch)
                    else:
                        _,q_value1,_,_, embedding_feature_pre, embedding_feature_deocde = dp_model(node_inputs_batch, edge_inputs_batch, budget_inputs_batch, \
                            current_inputs_batch, LSTM_h_batch, LSTM_c_batch, pos_encoding_batch, mask_batch)
                    
                    criterion = torch.nn.SmoothL1Loss()
                    ae_loss = nn.BCEWithLogitsLoss()
                    
                    #print("01--------------------------q_value1:", q_value1)
                    #print("02-------------------q_value1.size():", q_value1.size())
                    #print("03----------------------action_batch:", action_batch)
                    #print("04---------------action_batch.size():", action_batch.size())
                    q_value = q_value1.gather(2, action_batch.long())
                    obj_critic = criterion(q_value, q_label).mean()
                    if IF_AE:
                        ae_loss = nn.BCEWithLogitsLoss()
                        AE_loss = ae_loss(embedding_feature_pre, embedding_feature_deocde).mean()
                        loss = obj_critic + AE_loss
                    else:
                        loss = obj_critic
                    
                    q_value = q_value.mean()
                    obj_critics += obj_critic.item()
                    q_values += q_value.item()
                    reward_batch_mean += reward_batch.mean()
                
                global_optimizer.zero_grad()
                # loss.backward()
                scaler.scale(loss).backward()
                #scaler.scale(obj_critic).backward()
                scaler.unscale_(global_optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(eval_net.parameters(), max_norm=10, norm_type=2)
                # global_optimizer.step()
                scaler.step(global_optimizer)
                scaler.update()
                soft_update(target_net, eval_net, SOFT_UPDATE_TAU)
            lr_decay.step()
            obj_critics = obj_critics / update_times
            q_values = q_values / update_times
            reward_batch_mean = reward_batch_mean / update_times
            
            perf_data = []
            for n in metric_name:
                perf_data.append(np.nanmean(perf_metrics[n]))
            data = [reward_batch_mean.cpu(), obj_critics, q_values, *perf_data]
            trainingData.append(data)
            
            if len(trainingData) >= SUMMARY_WINDOW:
                writeToTensorBoard(writer, trainingData, curr_episode)
                trainingData = []
                
            # get the updated global weights
            ###if update_done == True:
            if device != local_device:
                weights = eval_net.to(local_device).state_dict()
                eval_net.to(device)
            else:
                weights = eval_net.state_dict()
            
            if curr_episode % 32 == 0:
                print('Saving model', end='\n')
                checkpoint = {"model": eval_net.state_dict(),
                              "optimizer": global_optimizer.state_dict(),
                              "episode": curr_episode,
                              "lr_decay": lr_decay.state_dict()}
                path_checkpoint = "./" + model_path + "/checkpoint.pth"
                torch.save(checkpoint, path_checkpoint)
                print('Saved model', end='\n')
            
    except KeyboardInterrupt:
        print("CTRL_C pressed. Killing remote workers")
        for a in meta_agents:
            ray.kill(a)
  
if __name__ == "__main__":
    main()
