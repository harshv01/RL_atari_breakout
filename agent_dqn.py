#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque
import os
import sys
import pickle
# from numpy import random
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import time

from agent import Agent
from dqn_model import DQN
"""
you can import any package and define any extra function as you need
"""

torch.manual_seed(595)
np.random.seed(595)
random.seed(595)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEvice", device)
class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize everything you need here.
        For example: 
            paramters for neural network  
            initialize Q net and target Q net
            parameters for repaly buffer
            parameters for q-learning; decaying epsilon-greedy
            ...
        """

        super(Agent_DQN,self).__init__(env)
        ###########################
        # YOUR IMPLEMENTATION HERE #
        
        self.args = args
        self.Q_net = DQN()          # Q network initialize
        self.target_Q_net = DQN()   # Target Q network initialize

        self.Q_net = self.Q_net.to(device)
        self.target_Q_net = self.target_Q_net.to(device)

        self.env = env
        self.epsilon = 1    # Exploration probability
        self.epsilon_decay = 0.99
        self.min_epsilon = 0.1
        self.learning_rate = 0.0000625

        # self.total_steps = 100
        self.target_Q_update = 1000
        self.episodes_list = []
        self.rewards_list = []

        self.avg_episodes_list = []
        self.avg_rewards_list = []


        self.batch_size = 32
        self.buffer_size = 50000
        self.replay_buff = deque(maxlen=self.buffer_size)  # Doubly linked queue
        self.gamma = 0.99
        self.episodes = 12000000
        self.optimiser = optim.Adam(self.Q_net.parameters(), self.learning_rate)


        if args.test_dqn:
            #you can load your model here
            # print('loading trained model')
            ###########################
            # YOUR IMPLEMENTATION HERE #
            state_dict = torch.load('target_q_net_weights.pth')
            self.Q_net.load_state_dict(state_dict)
            self.Q_net.eval()

    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary.
        If no parameters need to be initialized, you can leave it as blank.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        
        # Reverse the epsilon decay:
        self.epsilon=0
        self.epsilon_decay=1.0

        ###########################
        pass
    
    
    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent
        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)
        Return:
            action: int
                the predicted action from trained model
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        
        if random.random() > self.epsilon:  # Epsilon greedy
            observation = torch.tensor(observation, dtype=torch.float32)    #convert to tensor
            # observation = torch.transpose(observation, (2, 0, 1))   # converts 84 x 84 x 4 to 4 x 84 x 84
            # observation = observation.permute(0, 3, 1, 2)  # converts 84 x 84 x 4 to 4 x 84 x 84
            observation = observation.unsqueeze(0)
            observation = observation.to(device)
            with torch.no_grad():
                output_qval = self.Q_net(observation)
                output_action_idx = output_qval.argmax(dim=1)
                output_action = output_action_idx.item()
        
            return output_action

        else:
            return random.randint(0, 3)


        ###########################
        # return action
    
    def push(self, state, action, reward, nextstate, terminated):
        """ You can add additional arguments as you need. 
        Push new data to buffer and remove the old one if the buffer is full.
        
        Hints:
        -----
            you can consider deque(maxlen = 10000) list
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
    

        self.replay_buff.append((
            torch.tensor(state, dtype=torch.float32), 
            torch.tensor(action, dtype=torch.int64), 
            torch.tensor(reward, dtype=torch.float32), 
            torch.tensor(nextstate, dtype=torch.float32), 
            torch.tensor(terminated, dtype=torch.float32)
            ))
        return

        ###########################
        
        
    def replay_buffer(self):
        """ You can add additional arguments as you need.
        Select batch from buffer.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        
        if len(self.replay_buff) < self.batch_size:
            return
        else:
            return random.sample(self.replay_buff, self.batch_size)
        
        ###########################
        # return 
        

    # def update_plot(episode, total_reward):
    #     """
    #     Function to update the plot during training.
    #     """
    #     x_vals.append(episode)
    #     y_vals.append(total_reward)
        
    #     ax.clear()  # Clear the current plot
    #     ax.plot(x_vals, y_vals, label='Total Reward')  # Plot rewards over episodes
    #     ax.set_xlabel('Episodes')
    #     ax.set_ylabel('Total Reward')
    #     ax.set_title('Reward vs Episodes')
    #     ax.legend()
        
    #     plt.draw()  # Redraw the plot to show the updates
    #     plt.pause(0.001)  # Pause to allow for updates and interactivity

    def plot_rewards(self, episodes, rewards, type):
        plt.figure(figsize=(10,6))
        plt.plot(episodes, rewards, label='Total Reward per Episode', color='b')
        plt.xlabel('Episodes')
        plt.ylabel('Total Reward')
        plt.title('Reward vs Episodes')
        plt.grid(True)
        plt.legend()
        # plt.show()

        if type==1:
            plt.savefig('latest_plot.png')
        else:
            plt.savefig('avg_latest_plot.png')
        plt.close()

    def train(self):
        """
        Implement your training algorithm here
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        begin_time = time.time()
        steps = 0
        for episode in range(self.episodes):
            
            if (episode+1) % 100 ==0:
                # print('should print this')
                print('mean reward of last 100 episodes: ', np.mean(self.rewards_list[-100:]), flush=True)

            if episode % 1000 ==0:
                print("saving")
                torch.save(self.Q_net.state_dict(), 'q_net_weights.pth')
                torch.save(self.target_Q_net.state_dict(), 'target_q_net_weights.pth')
                self.plot_rewards(self.episodes_list, self.rewards_list, 1)
                self.plot_rewards(self.avg_episodes_list, self.avg_rewards_list, 2)
                
                with open('plot_data.pkl', 'wb') as file:
                    pickle.dump((self.episodes_list, self.rewards_list, self.avg_episodes_list, self.avg_rewards_list), file)

                # self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

                print("Saved all files")
            
            state = self.env.reset()
            total_loss = 0
            total_reward = 0
            isDone = False


            while not isDone:

                # Update the target network periodically
                if steps % self.target_Q_update == 0:
                    print('here-----')
                    self.target_Q_net.load_state_dict(self.Q_net.state_dict())
                steps+=1
                    
                eps_greedy_action = self.make_action(state)
                next_state, reward, isDone, _, info = self.env.step(eps_greedy_action)

                self.push(state, eps_greedy_action, reward, next_state, isDone)
                
                if len(self.replay_buff) < self.batch_size:
                    state = next_state
                    total_reward+=reward
                    continue

                batch = self.replay_buffer()
                if batch is None:
                    print("Empty batch")
                    continue
                
                states, actions, rewards, next_states, dones = map(torch.stack, zip(*batch))

                # states = states.permute(0, 3, 1, 2)

                # print("states size", states.shape)
                # print("actions size", actions.shape)


                states = states.to(device)
                actions = actions.to(device)
                rewards = rewards.to(device)
                next_states = next_states.to(device)
                dones = dones.to(device)
                
                state = next_state
                total_reward+=reward

            ##-----------------------------------TO BE REVIEWED
                # print("------one")
                current_q_values = self.Q_net(states).gather(1, actions.unsqueeze(1))

                with torch.no_grad():
                    # print("------two")
                    next_q_values = self.target_Q_net(next_states).max(dim=1, keepdim=True)[0]  #selects the maximum 
                    target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

                loss = F.mse_loss(current_q_values, target_q_values)
                total_loss+=loss

                # Backpropagation
                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()

            # Decay epsilon
            if episode % 100==0:
                self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
                # self.epsilon *= self.epsilon_decay

            print(f"Episode {episode + 1}/{self.episodes} - Total Reward: {total_reward} - Loss: {total_loss} - Time Elapsed: {time.time()-begin_time}", flush=True)
            self.episodes_list.append(episode)
            self.rewards_list.append(total_reward)

            self.avg_episodes_list.append(episode)
            self.avg_rewards_list.append(np.mean(self.rewards_list[-30:]))

            if(np.mean(self.rewards_list[-100:])>40):
                print("Mean is greater than 40. Ending episode loop")
                break

        torch.save(self.Q_net.state_dict(), 'q_net_weights.pth')
        torch.save(self.target_Q_net.state_dict(), 'target_q_net_weights.pth')

        with open('plot_data.pkl', 'wb') as file:
            pickle.dump((self.episodes_list, self.rewards_list), file)

        self.plot_rewards(self.episodes, self.rewards)
        ###########################

        ##-----------------------------------
        
        ###########################

