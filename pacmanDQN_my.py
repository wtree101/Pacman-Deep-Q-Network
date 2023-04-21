# import pacman game 
#from pacman import Directions
from game import Directions
from pacmanUtils import *
from game import Agent
import game
from myTeam import OffensiveReflexAgent
# import torch library
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from DQN import *
from captureAgents import CaptureAgent
from dummyteam import ReflexCaptureAgent
#import other libraries
import os
import util
import random
import numpy as np
import time
import sys

from time import gmtime, strftime
from collections import deque

# model parameters
model_trained = False
tuned_from_network = False
GAMMA = 0.95  # discount factor
LR = 0.01     # learning rate

batch_size = 32            # memory replay batch size
#batch_size = 8          # memory replay batch size
memory_size = 50000		   # memory replay size
start_training = 300 	   # start training at this episode
#start_training = 32 	   # start training at this episode
TARGET_REPLACE_ITER = 100  # update network step

epsilon_final = 0.1   # epsilon final
epsilon_step = 7500

def createTeam(firstIndex, secondIndex, isRed,
               first = 'PacmanDQN', second = 'ReflexCaptureAgent',numTraining=0):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

class PacmanDQN(PacmanUtils,OffensiveReflexAgent):
    def __init__(self,index,numTraining=0): #再写了一个重构函数
        #PacmanUtils.__init__(self,index,numTraining) #也就是CaptureAgent的重构函数
        OffensiveReflexAgent.__init__(self,index,numTraining)
        print("Started Pacman DQN algorithm")
        if(model_trained == True):
            print("Model has been trained")
        else:
            print("Training model")

        # pytorch parameters
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.device = torch.device("mps") #apple
        #self.device = "cpu"
        if(model_trained == True):
            self.policy_net = torch.load('pacman_policy_net.pt',map_location=self.device).to(self.device)
            self.target_net = torch.load('pacman_policy_net.pt',map_location=self.device).to(self.device)
        else:
            if tuned_from_network == True:
                self.policy_net = torch.load('pacman_policy_net_20000_418.pt').to(self.device)
                self.target_net = torch.load('pacman_target_net_20000_418.pt').to(self.device)
            else:
                self.policy_net = DQN().to(self.device)
                self.target_net = DQN().to(self.device)
        
        # self.policy_net.double()
        # self.target_net.double()    
        self.policy_net.float()
        self.target_net.float()      
        
        # init optim
        self.optim = torch.optim.RMSprop(self.policy_net.parameters(), lr=0.00025, alpha=0.95, eps=0.01)
        
        # init counters
        self.counter = 0
        self.win_counter = 0
        self.memory_counter = 0
        self.local_cnt = 0

        if(model_trained == False):
            self.epsilon = 0.5     # epsilon init value
            # if tuned_from_network == True:
            #     self.epsilon = 0.1
    
        else:
            self.epsilon = 0.0     # epsilon init value

        # init parameters
        # self.width = args['width']
        # self.height = args['height'] ??
        self.num_training = numTraining
        
        # statistics
        self.episode_number = 0
        self.last_score = 0
        self.last_reward = 0.
        
		# memory replay and score databases
        self.replay_mem = deque()
        
		# Q(s, a)
        self.Q_global = []  
		
		# open file to store information
        self.f= open("data_dqn_2.txt","a")
        self.startEpisode()
    
    
    
    def registerInitialState(self, gameState): #defined in Pacman utils
        PacmanUtils.registerInitialState(self,gameState) #也就是CaptureAgent的重构函数
        OffensiveReflexAgent.registerInitialState(self,gameState)
         #初始化时调用开始函数
        #pass #unfinished 相比于init 更多根据state信息的初始化。 
        #self.device = "cpu"
        # if(model_trained == True):
        #     self.policy_net = torch.load('pacman_policy_net.pt').to(self.device)
        #     self.target_net = torch.load('pacman_policy_net.pt').to(self.device)
        # else:
        #     if tuned_from_network == True:
        #         self.policy_net = torch.load('pacman_policy_net_7000_417.pt').to(self.device)
        #         self.target_net = torch.load('pacman_target_net_7000_417.pt').to(self.device)
        #     else:
        #         self.policy_net = DQN().to(self.device)
        #         self.target_net = DQN().to(self.device)
        
        # # self.policy_net.double()
        # # self.target_net.double()    
        # self.policy_net.float()
        # self.target_net.float()    
       

    
    def observationFunction(self, gameState):#重写这个观察函数，改成RL版本，会被调用
        """ Changing this won't affect pacclient.py, but will affect capture. """
        if self.last_action is not None:
            myState = gameState.getAgentState(self.index)
            #features = self.getFeatures(gameState,'Stop')
            if not self.lastState is None:
                
                features = self.getFeatures(self.lastState,self.get_direction(self.last_action))
                features = self.getFeatures_offensive(self.lastState,self.get_direction(self.last_action),features)
                #print(features['stop'])
                last_myState = self.lastState.getAgentState(self.index)
                score_change = gameState.getScore() - self.lastState.getScore() #排除对面的影响
                reward = (myState.numCarrying -last_myState.numCarrying) * 2 + max(score_change,0)*8 + features['stop'] * (-6) + features['distanceToFood'] * (-0.02)#+ features['check_died'] * (-10)
                #print('reward',reward)
                #     self.last_reward = 50.    # ate a ghost 
                # elif reward > 0:
                #     self.last_reward = 10.    # ate food 
                # elif reward < -10:
                #     self.last_reward = -500.  # was eaten
                #     self.won = False
                # elif reward < 0:
                #     self.last_reward = -1.    # didn't eat
                self.last_reward = reward
                if(self.terminal and self.won):
                    self.last_reward = 100.
                    self.win_counter += 1
                self.episode_reward += self.last_reward


                self.last_state = np.copy(self.current_state)
                
                self.current_state = self.getStateMatrices(gameState)

                # store transition 
                transition = (self.last_state, self.last_reward, self.last_action, self.current_state, self.terminal)
                self.replay_mem.append(transition)
                if len(self.replay_mem) > memory_size:
                    self.replay_mem.popleft()
                
                self.train()

                #self.observeTransition(self.lastState, self.lastAction, gameState, reward)
            # next
        self.local_cnt += 1
        self.frame += 1

        # update epsilon
        if(model_trained == False):
            self.epsilon = max(epsilon_final, 1.00 - float(self.episode_number) / float(epsilon_step))
            # if tuned_from_network==True:
            #     self.epsilon = 0.1
        self.lastState = gameState #还是需要存储，不然reward没有计算
        #self.lastAction = action
        return gameState
    
    
    def chooseAction(self, state): #ps 在这里叫chooseAction，虽然在Qlearning 那边叫getAction，迁移过来
        """
        Compute the action to take in the current state.  With
        probability self.epsilon, we should take a random action and
        take the best policy action otherwise.  Note that if there are
        no legal actions, which is the case at the terminal state, you
        should choose None as the action.

        HINT: You might want to use util.flipCoin(prob)
        HINT: To pick randomly from a list, use random.choice(list)
        """
        
        # Pick Action
        legalActions_dict = {}
        legalActions = state.getLegalActions(self.index)  #两边的getlegalAction有些不同
        
        action = None
        "*** YOUR CODE HERE ***"
        #print(legalActions)
        if len(legalActions)==0:
          print('warn!')
          return None
        is_explore = util.flipCoin(self.epsilon)
        #is_explore =0
        if not is_explore:
            
            # get current state
            temp_current_state = torch.from_numpy(np.stack(self.current_state))
            temp_current_state = temp_current_state.unsqueeze(0)
            temp_current_state = temp_current_state.to(self.device,dtype=torch.float32)
            #action = random.choice(legalActions)

            # get Qsa
            #print(temp_current_state.shape)
            #print(temp_current_state[0][6])
            self.Q_found = self.policy_net(temp_current_state)        
            self.Q_found =  self.Q_found.detach().cpu()
            
            self.Q_found = self.Q_found.numpy()[0]
            #print(self.Q_found)
            for a in legalActions:
                legalActions_dict[a] = self.Q_found[int(self.get_value(a))]
            max_val = max(legalActions_dict.values())  # 找到最大值
            #print(max_val)
            best_action = [k for k, v in legalActions_dict.items() if v == max_val]
            #print(best_action)
            #self.Q_found = self.Q_found[legalActions]  #actions is index? 0...3?
            # store max Qsa
            self.Q_global.append(max_val) #之前忘记加入了。 这是一个
			# get best_action - value between 0 and 3
            #best_action = np.argwhere(self.Q_found == np.amax(self.Q_found)) 
            
            if len(best_action) > 1:  # two actions give the same max
                random_value = np.random.randint(0, len(best_action)) # random value between 0 and actions-1
                move = best_action[random_value]
            else:
                move = best_action[0]

        else:
            move = random.choice(legalActions)
            #move = self.get_direction(action)
          #action = self.computeActionFromQValues(state)
       # util.raiseNotDefined()

       # return action
        #self.doAction(state,action)
        ##这部分和环境交互
        self.last_action = self.get_value(move)
        #self.lastState = state
        #self.lastAction = action
        #self.observationHistory.append((state,action)) #这里改写了，会加入一个动作

        return move

    
    def final(self, state):
        # Next
        self.episode_reward += self.last_reward

        # do observation
        self.terminal = True
        if state.getScore()>0:
            self.won = True
        else:
            self.won = False
        _ = self.observationFunction(state)
        
		# print episode information
        print("Episode no = " + str(self.episode_number) + "; won: " + str(self.won) 
		+ "; Q(s,a) = " + str(max(self.Q_global, default=float('nan'))) + "; reward = " +  str(self.episode_reward) + "; and epsilon = " + str(self.epsilon))
		
		# copy episode information to file
        self.counter += 1
        if(self.counter % 10 == 0):
            self.f.write("Episode no = " + str(self.episode_number) + "; won: " + str(self.won) 
		+ "; Q(s,a) = " + str(max(self.Q_global, default=float('nan'))) + "; reward = " +  str(self.episode_reward) + "; and epsilon = " 
		+ str(self.epsilon) + ", win percentage = " + str(self.win_counter / 10.0) + ", " + str(strftime("%Y-%m-%d %H:%M:%S", gmtime())) + "\n")
            self.win_counter = 0

        if(self.counter % 100 == 0):
            # save model
            torch.save(self.policy_net, 'pacman_policy_net.pt')
            torch.save(self.target_net, 'pacman_target_net.pt')
        
        if(self.episode_number % TARGET_REPLACE_ITER == 0):
            print("UPDATING target network")
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def train(self):
        if (self.local_cnt > start_training):
            batch = random.sample(self.replay_mem, batch_size)
            batch_s, batch_r, batch_a, batch_n, batch_t = zip(*batch)
            
            # convert from numpy to pytorch 
            batch_s = torch.from_numpy(np.stack(batch_s))
            batch_s = batch_s.to(self.device,dtype=torch.float32)
            batch_r = torch.DoubleTensor(batch_r).unsqueeze(1).to(self.device,dtype=torch.float32)
            batch_a = torch.LongTensor(batch_a).unsqueeze(1).to(self.device,dtype=torch.long) # it is int
            batch_n = torch.from_numpy(np.stack(batch_n)).to(self.device,dtype=torch.float32)
            #batch_t = torch.ByteTensor(batch_t).unsqueeze(1).to(self.device,dtype=torch.float32) is terminal not used?
            
            # get Q(s, a)
            state_action_values = self.policy_net(batch_s).gather(1, batch_a)

            # get V(s')
            next_state_values = self.target_net(batch_n)

            # Compute the expected Q values                        
            next_state_values = next_state_values.detach().max(1)[0]
            next_state_values = next_state_values.unsqueeze(1)
            
            expected_state_action_values = (next_state_values * GAMMA) + batch_r
            
			# calculate loss
            loss_function = torch.nn.SmoothL1Loss()
            self.loss = loss_function(state_action_values, expected_state_action_values)
            
			# optimize model - update weights
            self.optim.zero_grad()
            self.loss.backward()
            self.optim.step()   
    def startEpisode(self):
        """
          Called by environment when new episode is starting
        """
        self.last_action = None
        self.last_state = None
        self.lastState = None
        self.lastAction = None
        self.episodeRewards = 0.0

    def stopEpisode(self): #这个也可以和环境交互着写
        """
          Called by environment when episode is done
        """
        if self.episodesSoFar < self.numTraining:
            self.accumTrainRewards += self.episodeRewards
        else:
            self.accumTestRewards += self.episodeRewards
        self.episodesSoFar += 1
        if self.episodesSoFar >= self.numTraining:
            # Take off the training wheels
            self.epsilon = 0.0    # no exploration
            self.alpha = 0.0      # no learning
             
