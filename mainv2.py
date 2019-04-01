from __future__ import print_function
from collections import deque

from dataset import datasets
from node import  CloudNet
import numpy as np
from sklearn.metrics import accuracy_score as acc
import os

from pg_actor_critic import PolicyGradientActorCritic

import tensorflow as tf
import keras
import keras.layers
import keras.models
import keras.optimizers
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from node import CloudNet, Enviroment_e

import matplotlib.pyplot as plt
import math

from imageshow import show_img
import matplotlib.cm as cm

env_name='multisensor'

iftrain_CloudNet=0
iftrain_RLNet = 1
load_model = 0
iftest_compl = 1
init_exp=0.8
final_exp=0.2
anneal_steps=1#500    


sess = tf.InteractiveSession()

X_train, X_test, Y_train, Y_test = datasets.get_mvmc(te_percent=0.20)

def input_dict_to_list(x,y):
    y_ = to_categorical(y)
    x_ = [x['0'].transpose(0, 2, 3, 1), x['1'].transpose(0, 2, 3, 1), x['2'].transpose(0, 2, 3, 1),
          x['3'].transpose(0, 2, 3, 1), x['4'].transpose(0, 2, 3, 1), x['5'].transpose(0, 2, 3, 1)]
    return x_,y_

X_train_o, X_test_o, Y_train_o, Y_test_o = X_train, X_test, Y_train, Y_test

X_train, Y_train = input_dict_to_list(X_train, Y_train)
X_test, Y_test   = input_dict_to_list(X_test, Y_test)

if iftrain_CloudNet==1:
    # tf.reset_default_graph()    
    cl_train = CloudNet(train=1)
    cl_train.train_model(X_train_o, Y_train_o, bt_s=50, eps=100)

if iftrain_RLNet==1:
    env=Enviroment_e(X_train,Y_train)

    optimizer = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=0.9)
    writer = tf.summary.FileWriter("/tmp/{}-experiment-1".format(env_name))

    state_dim   = env.observation_space_dim #observation_space.shape[0]
    num_actions = env.action_space_number #action_space.n

    pg_reinforce = PolicyGradientActorCritic(sess,
                                            optimizer,
                                            # actor_network,
                                            # critic_network,
                                            state_dim,
                                            num_actions,
                                            summary_writer=writer,load_model=load_model,
                                            init_exp=init_exp, final_exp=final_exp,anneal_steps=anneal_steps, if_train= 1)

    MAX_EPISODES = 2500
    MAX_STEPS    = 100  

    no_reward_since = 0
    Best_avg_reward = 0
    last_saved_episod=0

    episode_history = deque(maxlen=200)
    episode_accuracy = deque(maxlen=200)
    episode_action = deque(maxlen=200)
    reward_history_mean = []
    accuracy_history_mean = []
    action_history_mean = []
    cnt = 0
    for i_episode in range(MAX_EPISODES):

        # initialize
        observed_state = env.reset()
        total_rewards = 0
        number_of_correct=0
        
        for t in range(MAX_STEPS):
            action,state_value,action_prob_v = pg_reinforce.sampleAction(observed_state[np.newaxis,:])
            next_state, reward, done, is_correct = env.step(action)
            # print('=======')

            print((action[:],state_value[0],reward[0],is_correct),end=",")     
            cnt += 1
            #if cnt%500 == 0 and is_correct:
            # if i_episode > 2900:
            #     show_img(env.input_data_x, env.current_x_cl[1][0], action)
            # print((action[:],action_prob_v,state_value[0],reward[0],is_correct),end="\n")
            # print(state_value)
            # print(next_state.shape)

            total_rewards += reward
            number_of_correct +=is_correct
            # reward = 5.0 if done else -0.1
            reward = reward if done else -0.1
            pg_reinforce.storeRollout(observed_state, action, reward)
            # print(state_value,end=',')

            observed_state = next_state
            if done: 
                break

        # if we don't see rewards in consecutive episodes
        # it's likely that the model gets stuck in bad local optima
        # we simply reset the model and try again
        # print("----------e2----------")
        # print(total_rewards)

        if total_rewards <= -500:
            no_reward_since += 1
            if no_reward_since > 5:
                # create and initialize variables
                print('Resetting model ... start anew!')
                pg_reinforce.resetModel()
                no_reward_since = 0
                continue
        else:
            no_reward_since = 0

        cross_entropy_loss_v, mean_square_loss_v=pg_reinforce.updateModel()

        episode_history.append(total_rewards)
        episode_accuracy.append(number_of_correct)
        episode_action.append(action)
        mean_rewards = np.mean(episode_history)
        reward_history_mean.append(mean_rewards)
        mean_accuracy = np.mean(episode_accuracy)
        accuracy_history_mean.append(mean_accuracy)
        action_history_mean.append(np.mean(episode_action))
        if (i_episode%50)==1:
            # print(cross_entropy_loss_v.shape)
            # print(cross_entropy_loss_v)
            print("\n Episode {}".format(i_episode))
            print("Finished after {} timesteps".format(t+1))
            print("cross_entropy_loss: {}".format(cross_entropy_loss_v))
            print("mean_square_loss_v: {:.2f}".format(mean_square_loss_v))
            print("Reward for this episode: {}".format(total_rewards))
            print("Average reward for last 200 episodes: {:.2f}".format(mean_rewards))
            print("Average accuracy for last 200 episodes: {:.2f}".format(mean_accuracy))
            print("exploration rate",pg_reinforce.exploration)
            if mean_rewards >= 200.0 and len(episode_history) >= 100:
                print("Environment {} solved after {} episodes".format(env_name, i_episode+1))
                break
            

        # if ((i_episode-last_saved_episod)>300) and (Best_avg_reward<mean_rewards):    
        if ((i_episode-last_saved_episod)>300):    
            save_path = pg_reinforce.saver.save(sess, "./policy_weights.ckpt")
            print("\n Model saved in path: %s" % save_path)
            Best_avg_reward=mean_rewards
            last_saved_episod=i_episode

    # for p in plot_list[1:5]:
    #     f = plt.figure()
    #     for i in range(p.shape[1]): 
    #         ax = f.add_subplot(p.shape[1], 1, i+1) 
    #         ax.plot(p[:,i], label=str(i)) 
    #         ax.legend()
    # for p in plot_list[-5:-1]:
    #     f = plt.figure()
    #     for i in range(p.shape[1]): 
    #         ax = f.add_subplot(p.shape[1], 1, i+1) 
    #         ax.plot(p[:,i], label=str(i)) 
    #         ax.legend()
    # plt.show()
   
    

# exit()

# node_output = []
# for l in range(6):
#     node_output.append(node[l].calculate(X_test[str(l)]))
if iftest_compl==1:


    env_test1=Enviroment_e(X_train,Y_train)

    print("here1")
    x = []
    for l in range(6):
        x.append(X_test_o[str(l)].reshape((-1, 32, 32, 3)))
    print(env_test1.cl_rl.eval_comp_model(x, Y_test_o))



    #----------------------------------------------------------------
    print("here2")
    # env_test=Enviroment_e(X_train,Y_train)
    env_test=Enviroment_e(X_test,Y_test)

    # print(env_test.cl_rl.calculate_claud(env_test.node_output,action=0,apply_action=0))
    print("here2.5")
    # print(env_test.cl_rl.evaluate_claud(env_test.node_output,Y_test_o,action=0,apply_action=0))



    print("here3")

    optimizer = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=0.9)
    writer = tf.summary.FileWriter("/tmp/{}-experiment-1".format(env_name))

    state_dim   = env_test.observation_space_dim #observation_space.shape[0]
    num_actions = env_test.action_space_number #action_space.n


    if iftrain_RLNet==1:
        pg_reinforce_t=pg_reinforce
    else:
        pg_reinforce_t = PolicyGradientActorCritic(sess,
                                                optimizer,
                                                # actor_network,
                                                # critic_network,
                                                state_dim,
                                                num_actions,
                                                summary_writer=writer,load_model=1,init_exp=0, if_train=0)



    pg_reinforce_t.if_train=0
    # policy_for_test = policy_net.model.predict(node_output_t)

    # pnet_in, pnet_out = policy_net.get_in_out_tensor()
    
    observed_state=env_test.reset()
    action,state_value,action_prob_v = pg_reinforce_t.sampleAction(observed_state[np.newaxis,:])
    next_state, reward, done, _ = env_test.step(action)

    # print(pnet_out[0].shape)
    print((action))

    print("------RL Selected-----")
    total_rewards=0
    num_correct=0
    Test_num=1500
    how_many_used=np.zeros(action.shape)
    for t in range(Test_num):
        action,state_value,action_prob_v = pg_reinforce_t.sampleAction(observed_state[np.newaxis,:])
        next_state, reward, done, is_correct = env_test.step(action)
        
        # print(reward)
        # print(action)
        # print(action_prob_v)
        observed_state=next_state

        total_rewards += reward
        how_many_used += (action)
        if is_correct==1:
            num_correct+=1

    print("total reward",total_rewards/Test_num)
    print("print avg accuracy",num_correct/Test_num)
    print("per sensor activity percente ",how_many_used/Test_num)
    print("average sensor activity percente",np.sum(how_many_used)/Test_num/6)
    print("exploration rate",pg_reinforce_t.exploration)

    print("------requested-----")
    total_rewards=0
    num_correct=0
    fixed_action = np.array([1,0,1,0,0,0])
    how_many_used=np.zeros(action.shape)
    for t in range(Test_num):
        action = fixed_action
        next_state, reward, done, is_correct = env_test.step(action)
        observed_state=next_state

        total_rewards += reward
        # how_many_used += np.count_nonzero(action)
        how_many_used += (action)
        if is_correct==1:
            num_correct+=1

    print("total reward",total_rewards/Test_num)
    print("print avg accuracy",num_correct/Test_num)
    print("active node per sensor",how_many_used/Test_num)
    print("average sensor activity percente",np.sum(how_many_used)/Test_num/6)

    print("-"*100)
    print("------selective-----")
    fixed_actions = np.array([[1,0,0,0,0,0],
                              [0,1,0,0,0,0],
                              [0,0,1,0,0,0],
                              [0,0,0,1,0,0],
                              [0,0,0,0,1,0],
                              [0,0,0,0,0,1]])
    for fa in fixed_actions:
        total_rewards=0
        num_correct=0
        how_many_used=np.zeros(action.shape)
        for t in range(Test_num):
            action = fa
            next_state, reward, done, is_correct = env_test.step(action)
            observed_state=next_state

            total_rewards += reward
            # how_many_used += np.count_nonzero(action)
            how_many_used += (action)
            if is_correct==1:
                num_correct+=1
        print(fa)
        print("total reward",total_rewards/Test_num)
        print("print avg accuracy",num_correct/Test_num)
        print("active node per sensor",how_many_used/Test_num)
        print("average sensor activity percente",np.sum(how_many_used)/Test_num/6)

    print("-"*100)


    print("------all in-----")
    total_rewards=0
    num_correct=0
    how_many_used=np.zeros(action.shape)
    for t in range(Test_num):
        action = np.ones(action.shape)
        next_state, reward, done, is_correct = env_test.step(action)
        observed_state=next_state

        total_rewards += reward
        # how_many_used += np.count_nonzero(action)
        how_many_used += (action)
        if is_correct==1:
            num_correct+=1

    total_acc = num_correct/Test_num
    total_rew = total_rewards/Test_num
    print("total reward", total_rewards/Test_num)
    print("print avg accuracy", num_correct/Test_num)
    print("active node per sensor", how_many_used/Test_num)
    print("average sensor activity percente", np.sum(how_many_used)/Test_num/6)

    
    ave_fig = plt.figure()
    ax = ave_fig.add_subplot(2, 1, 1)
    # ax.set_title("a")
    ax.plot(reward_history_mean[200:], 'r-', label='reward')
    ax.legend()
    ax.plot(accuracy_history_mean[200:], 'k-', label='accuracy')
    ax.legend()
    ax.plot([total_acc for e in accuracy_history_mean[200:]], 'C5-.', label='without drop accuracy')
    ax.legend()
    ax.plot([total_rew for e in accuracy_history_mean[200:]], 'C9-.', label='without drop reward')
    ax.legend()
    #ax.set_ylim(0, 1.05)
    ax.set_xlabel('Iterations')
    ax.grid()

    # ave_fig2 = plt.figure()
    # ax2 = ave_fig2.add_subplot(1, 1, 1)
    ax = ave_fig.add_subplot(2, 1, 2)
    # ax.set_title("b")
    ax.plot(action_history_mean[200:], 'b-', label='communication overhead')
    ax.legend()
    ax.plot(accuracy_history_mean[200:], 'k-', label='accuracy')
    ax.legend()
    ax.plot([total_acc for e in accuracy_history_mean[200:]], 'C5-.', label='without drop accuracy')
    ax.legend()
    ax.grid()
    ax.set_xlabel('Iterations')
    #####draw overhead per iterations
    overhead_accuracy_fig = plt.figure()
    ax = overhead_accuracy_fig.add_subplot(1, 1, 1)
    colors = cm.rainbow(np.linspace(0, 1, len(accuracy_history_mean[200:])))
    colors = np.flip(colors, axis = 0)
    for x, y, c in zip(accuracy_history_mean[200:], action_history_mean[200:], colors):
        ax.plot(x, y, '.',color=c)
    cmap = plt.get_cmap("Spectral")
    norm = plt.Normalize(0, len(colors))
    sm =  cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = overhead_accuracy_fig.colorbar(sm, ax=ax)
    cbar.ax.set_title("iterations")   
    ax.set_xlabel('accuracy')
    ax.set_ylabel('overhead')
    ax.grid()
 
plt.show()