# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
from game import Actions
from util import nearestPoint
import math

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'RLagent', second = 'DefensiveReflexAgent',numTraining=0):
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
  print('create',numTraining)
  return [eval(first)(firstIndex,numTraining), eval(second)(secondIndex,numTraining)]
 

##########
# Agents #
##########
class ReflexCaptureAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """

  
  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def bfs_for_divided_points(self,gameState,t='generate'):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    #self.divides = [] #装着分界点
  
    q = util.Queue()
    track = {} #create a dict, to judge if a node is visited and record its father
    q.push(gameState)
    track[self.start] = self.start
    # key 就是单重节点，但是father存储信息是 father节点+到达此节点的动作
    while (not q.isEmpty()):
      state_now  = q.pop()     
      myState = state_now.getAgentState(self.index)   
      if (t=='generate'):
        if myState.isPacman: #track_chase_back
          self.divides.append(track[state_now.getAgentPosition(self.index)])
          continue
      
      elif(t=='is_trapped'):
        if (not myState.isPacman): #成功回到了安全区域
          return False


      actions = state_now.getLegalActions(self.index)
      for action in actions:
        son = state_now.generateSuccessor(self.index,action)
        if (son.getAgentPosition(self.index) in track.keys()):
          continue
        track[son.getAgentPosition(self.index)] = state_now.getAgentPosition(self.index)
        q.push(son)

    #print(self.divides)
    return True #看看是否被卡住了
    util.raiseNotDefined()
  def P_deadcorners(self,pos,walls):
    
    
    visited = set()
    start = pos
    visited.add(pos)
    
    tunnel_len = 0
    
    while (1):
      (x_int,y_int) = pos
      possible = []
      for dir, vec in Actions._directionsAsList:
        dx, dy = vec
        next_y = y_int + dy
        next_x = x_int + dx
        if not walls[next_x][next_y] and (not (next_x,next_y)in visited): 
          possible.append((next_x,next_y))
      #visited.add(pos) #加入当前位置
      
      if (len(possible)<=1):
        tunnel_len += 1
        pos = possible[0]
        visited.add(pos)
      else:
        break
    
    if tunnel_len>0:
      self.deadCorners[start] = tunnel_len #之前错了是因为start - Pos 而 Pos 后面改掉了
      #self.debugDraw(start,[1,0,0])
    
    
    
  def registerInitialState(self, gameState):
    #print(self.index,self.numTraining)
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)
    self.divides = []
    self.bfs_for_divided_points(gameState)
    self.deadCorners = {}
    walls = gameState.getWalls()
    divide_line = (walls.width-1)/2
    for x in range(walls.width):
      for y in range(walls.height):
        if (walls[x][y]==False):
          if self.red and x>divide_line:
            self.P_deadcorners((x,y),walls)
            #if (x,y) in self.deadCorners.keys():
              
          if (not self.red) and x<divide_line:
            self.P_deadcorners((x,y),walls)


  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    foodLeft = len(self.getFood(gameState).asList())

    if foodLeft <= 2:
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start,pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction

    return random.choice(bestActions)

  

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action,features)
    return features * weights

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action,features):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}



class DefensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    features['invadercarrying'] = 0
    if len(invaders) > 0: #如果等于0，这个特征就不存在了。
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)
    
    for invader in invaders:
        features['invadercarrying'] += invader.numCarrying
    
    
    dists = [self.getMazeDistance(myPos, divide_point) for divide_point in self.divides]
    features['dividesDistance'] = min(dists)
    
    Capsules = self.getCapsulesYouAreDefending(gameState) #不用aslist，因为就是List
    features['numCapsules'] = len(Capsules)
    if len(Capsules) > 0: #如果等于0，这个特征也不存在了
      dists = [self.getMazeDistance(myPos, a) for a in Capsules]
      features['CapsulesDistance'] = min(dists)
      
    if action == Directions.STOP: features['stop'] = 1 #有趣的是只有停下来才加入这个特征，否则是不会怪罪到它的
   # rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
   # if action == rev: features['reverse'] = 1

    dists = [self.getMazeDistance(myPos, enemy.getPosition()) for enemy in enemies]
    features['enemyDistance'] = min(dists)




    return features

  def getWeights(self, gameState, action,features):
    #要小心不要为了追人跑出防守距离。
    weights_dict = {'numInvaders': -1000, 'onDefense': 500, 'invaderDistance': -20, 'stop': -50,
    'enemyDistance':-5}
    if (features['numInvaders']>0) and features['numCapsules']>0:
      weights_dict['CapsulesDistance'] = -2
    return weights_dict

class OffensiveReflexAgent(DefensiveReflexAgent):
  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
  def getFeatures_offensive(self, gameState, action,features): #在defensive基础上改。feacture一定要对于下一个状态而言
    
    successor = self.getSuccessor(gameState, action)
    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()
#################分数
    foodList = self.getFood(successor).asList()    
    features['successorScore'] = -len(foodList)#self.getScore(successor)
    features['realScore'] = self.getScore(successor)
#################危机
###########1 可走的路少 -- 但是感觉很难实现的
    actions = successor.getLegalActions(self.index)
    features['available_actions'] = len(actions)

    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    defenders = [a for a in enemies if (not a.isPacman) and a.getPosition() != None and a.scaredTimer<=5] #被恐吓的幽灵不算defenders
    features['defenders'] = len(defenders)

    features['check_died'] = 0
    if len(defenders) > 0: #如果等于0，这个特征就不存在了。
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in defenders if a.scaredTimer<=3 ] #被恐吓的幽灵不需要害怕
      if (len(dists)>0):
        features['defendersDistance'] = min(dists)
    
      #思考：错误的check_died 不是Pacman的话，根本不用害怕幽灵。
      for defender in defenders:
        #defender_pos = defender.getPosition()
        defender_scared = defender.scaredTimer
        if defender_scared==0 and features['defendersDistance']<=1: #有=0的肯定特征存在了
          features['check_died'] = 1
    
    #如果敌人是恐吓状态，应该可以继续前进到达安全区域
    features['check_trapped'] = 0
    if self.bfs_for_divided_points(gameState,t='is_trapped'):
      features['check_trapped'] = 1
    
    #features['successorScore'] = -len(foodList)  
#############################吃豆豆
    # Compute distance to the nearest food
    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance
    
    #到死角的距离
    features['possibly_trapped'] = 0
    if features['defenders']>0:
      for deadcorner in self.deadCorners.keys():
        dis_to_dead = self.getMazeDistance(myPos, deadcorner)
        tunnel_len = self.deadCorners[deadcorner]
        dis_out = tunnel_len - dis_to_dead
        if (dis_out*2>=features['defendersDistance']) and (dis_to_dead<tunnel_len):
          features['possibly_trapped'] = 1
          #print(myPos,dis_out*2,features['defendersDistance'],dis_to_dead,tunnel_len,deadcorner)
          
         # self.debugDraw(myPos,[1,0,0])
          break


#############################吃胶囊
    Capsules = self.getCapsules(successor)
    features['numCapsules_opponent'] = len(Capsules)
    if len(Capsules) > 0: #如果等于0，这个特征也不存在了
      dists = [self.getMazeDistance(myPos, a) for a in Capsules]
      features['CapsulesDistance_opponent'] = min(dists)
############################构造性特征
    features['Capsule_attract'] = features['CapsulesDistance_opponent']
    if features['defenders']!=0:
      features['Capsule_attract'] /= features['defendersDistance']

    

    

    
    

    return features

  def getWeights_offensive(self, gameState, action,features):
    #s = self.getSuccessor(gameState, action)
    myState = gameState.getAgentState(self.index)
    score_now = self.getScore(gameState)
    weights_dict = {'numCarring': 200, 'distanceToFood': -5,'check_died':-1000,'check_trapped':-300,'defendersDistance':1
    ,'dividesDistance':-1,'CapsulesDistance_opponent':-1,'numCapsules_opponent':-200,
    'Capsule_attract':-12,'stop': -300,'possibly_trapped':-300} #-300是为了如果有机会吃掉它，不因为吃了之后胶囊的惩罚而止步不前
    #危险状态
    if features['defenders']>1 and (score_now + myState.numCarrying - features['invadercarrying']>0) and myState.numCarrying>0: #考虑返回,但要保证分界点处可以进行——要吃掉最后一个豆豆
      weights_dict['dividesDistance'] = -10
    if (score_now + myState.numCarrying - features['invadercarrying']>0) and myState.numCarrying>15: #考虑返回,但要保证分界点处可以进行——要吃掉最后一个豆豆
      weights_dict['dividesDistance'] = -20 #一定要返回了，比食物吸引力大
    # #被追逐状态
    # if features['defenders']>0 and features['defendersDistance']<6: #正在被追
    #   weights_dict['CapsulesDistance_opponent'] = -6
    #   weights_dict['distanceToFood'] = -1
    #   weights_dict['deadcornerDistance'] = 2
      


    return weights_dict
  
  def evaluate(self, gameState, action): #重写这个evaluate函数
    """
    Computes a linear combination of features and feature weights
    """
    #successor = self.getSuccessor(gameState, action)
    features = self.getFeatures(gameState, action)

    #if (score<=0): #进攻模式,否则就猥琐地防守
    features = self.getFeatures_offensive(gameState, action,features)
    weights = self.getWeights_offensive(gameState, action,features)
    return features * weights



class RLagent(DefensiveReflexAgent):
    def __init__(self,index,numTraining=0): #再写了一个重构函数
        CaptureAgent.__init__(self,index,numTraining)
        ##########RL准备部分
        self.alpha = 0.3
        self.epsilon = 0.3
        self.discount = 0.9
        #self.numTraining = 9
       # print('RL',self.numTraining)
        self.accumTrainRewards = 0
        self.accumTestRewards = 0
        self.episodesSoFar = 0

        self.weights = util.Counter({'eat_food':200*0.8,'distanceToFood': -5,'check_died':-1000
    ,'divides_attract':-1,'CapsulesDistance_opponent':-2,'Capsule_attract':-12,'stop': -300,'Capsule_eaten':200,
    'CornerAvoid':-100,'bias':200,'defendersAvoid':-1})
        self.weights.divideAll(200)



    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)
        self.divides = []
        self.bfs_for_divided_points(gameState)
        self.deadCorners = {}
        walls = gameState.getWalls()
        divide_line = (walls.width-1)/2
        for x in range(walls.width):
            for y in range(walls.height):
                if (walls[x][y]==False):
                    if self.red and x>divide_line:
                        self.P_deadcorners((x,y),walls)
                        #if (x,y) in self.deadCorners.keys():
                        
                    if (not self.red) and x<divide_line:
                        self.P_deadcorners((x,y),walls)
                        
        if (self.numTraining==0):
            self.alpha = 0
            self.epsilon = 0
        self.startEpisode() #初始化时调用开始函数
            
        
        

    def Setalpha(self,alpha):
        self.alpha = alpha
    def Setepsilon(self,epsilon):
        self.epsilon = epsilon
    ###################环境调用部分

    def observeTransition(self, state,action,nextState,deltaReward):
        """
            Called by environment to inform agent that a transition has
            been observed. This will result in a call to self.update
            on the same arguments

            NOTE: Do *not* override or call this function
        """
        self.episodeRewards += deltaReward
        self.update(state,action,nextState,deltaReward)

    def startEpisode(self):
        """
          Called by environment when new episode is starting
        """
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
    
    def final(self, state): #RL版本，会被调用，我想在里面加一个全体更新。
        """
          Called by Pacman game at the terminal state
        """
        
        deltaReward = state.getScore() - self.lastState.getScore()
        self.observeTransition(self.lastState, self.lastAction, state, deltaReward)
        
        self.stopEpisode()
        #if (self.episodesSoFar>self.numTraining):
        #print (self.weights)
        if (self.episodesSoFar<=self.numTraining):
          if (self.episodesSoFar%10==0):
            with open('Output2.txt','a') as f:
             
              f.write(str(self.episodesSoFar)+':'+str(self.accumTrainRewards/10)) #这里的游戏盘数就是除以50
              f.write('\n')
              f.write(str(self.weights))
              f.write('\n')
            self.accumTrainRewards = 0
        
        if (self.episodesSoFar==self.numTraining):
          with open('Output2.txt','a') as f:
            f.write(str(self.weights))
            f.write('\n')
            f.write('Testing\n')
        elif (self.episodesSoFar>self.numTraining):
         
          with open('Output2.txt','a') as f:
            f.write(str(self.episodesSoFar)+':'+str(self.accumTestRewards/(self.episodesSoFar-self.numTraining)))
            f.write('\n')
            

        
        #if (self.episodesSoFar>self.numTraining): test外面会输出

        

        


        self.observationHistory = []
    
    
    
    def observationFunction(self, gameState):#重写这个观察函数，改成RL版本，会被调用
        """ Changing this won't affect pacclient.py, but will affect capture. """
        myState = gameState.getAgentState(self.index)
       
        if not self.lastState is None:
            last_myState = self.lastState.getAgentState(self.index)
            score_change = gameState.getScore() - self.lastState.getScore() #排除对面的影响
            reward = (myState.numCarrying -last_myState.numCarrying) * 0.8 + score_change#这个设定只适用于正分队
            self.observeTransition(self.lastState, self.lastAction, gameState, reward)

        return gameState
    
    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        difference = (reward + self.discount*self.getValue(nextState)) - self.getQValue(state,action)
        #for key in self.weights():
        self.weights = self.getWeights() + self.alpha*difference* (self.getFeatures_offensive(state,action))
       # util.raiseNotDefined()
    
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
        legalActions = state.getLegalActions(self.index)  #两边的getlegalAction有些不同
        action = None
        "*** YOUR CODE HERE ***"
        if len(legalActions)==0:
          return None
        is_explore = util.flipCoin(self.epsilon)
        if is_explore:
          action = random.choice(legalActions)
        else:
          action = self.computeActionFromQValues(state)
       # util.raiseNotDefined()

       # return action
        #self.doAction(state,action)
        ##这部分和环境交互
        self.lastState = state
        self.lastAction = action
        self.observationHistory.append((state,action)) #这里改写了，会加入一个动作

        return action

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        actions = state.getLegalActions(self.index)
        if len(actions)==0:
          return None
        else:
          action_list = []
          best_q = -float('inf')
          for action in actions:
            q = self.getQValue(state,action)
            #print(q)
            if (q>best_q):
              best_q = q
              action_list = [action]
            elif (q==best_q):
              action_list.append(action)
            else:
              pass
          
         
        return random.choice(action_list)
    
   
   #############################估计计算部分 
    def getWeights(self):
        return self.weights
    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        q = self.getWeights()*self.getFeatures_offensive(state,action)
        if self.episodesSoFar >= self.numTraining:
          pass
          #print(self.numTraining)
          #print(self.episodesSoFar)
          # print('Q',q)
          # print(self.weights)
          # print(self.getFeatures_offensive(state,action))
        return q
    def getValue(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        actions = state.getLegalActions(self.index) #事实上， nextState里面，应该行动的都不是我们，这样计算会出问题
        if len(actions)==0:
          return 0
        else:
          ans = -float('inf')
          for action in actions:
            ans = max(ans,self.getQValue(state,action))
          
          return ans

    def getFeatures_offensive(self, gameState, action): #在defensive基础上改。feacture一定要对于下一个状态而言
        
        features = self.getFeatures(gameState,action)
        successor = self.getSuccessor(gameState, action)
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()
    #################分数
        foodList_old = self.getFood(gameState).asList()
        foodList = self.getFood(successor).asList()   
        walls = gameState.getWalls() 
        #features['successorScore'] = -len(foodList)#self.getScore(successor)
       # features['realScore'] = self.getScore(successor) #权1.0 realscore是环境的真实反馈，我觉得这里不需要了。
        #features['numCarrying'] = myState.numCarrying #权1
        features["bias"] = 1.0
        
        
    #################危机
    ###########1 可走的路少 -- 但是感觉很难实现的
       # actions = successor.getLegalActions(self.index)
       # features['available_actions'] = len(actions)

        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        defenders = [a for a in enemies if (not a.isPacman) and a.getPosition() != None and a.scaredTimer<=5] #被恐吓的幽灵不算defenders
        features['defenders'] = len(defenders)

        features['check_died'] = 0
        
        if (len(defenders) > 0): #如果等于0，这个特征就不存在了。
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in defenders if a.scaredTimer<=5 ] #被恐吓的幽灵不需要害怕,还是统一成5吧，不要区分了，会造成一些除0
            if (len(dists)>0):
                features['defendersDistance'] = min(dists)
                features['defendersAvoid'] = math.exp(-(features['defendersDistance']-1))/(walls.width * walls.height/100) #做了一个指数级别下降的特征,然后稍微scale变小一点不要太大。
            if (myState.isPacman):
                for defender in defenders:
                    #defender_pos = defender.getPosition()
                    defender_scared = defender.scaredTimer
                    if defender_scared==0 and features['defendersDistance']<=1: #有=0的肯定特征存在了 在这里还不要把Distance标准化，不然一直死
                        features['check_died'] = 1 #权-2先吧
        
            features['defendersDistance'] /= (walls.width * walls.height) #后续都小化成小于1的值。
        
        if (features['check_died']==0): #没有死亡危险的情况下，吃豆豆
          if myPos in foodList_old: #注意是旧的foodlist
            features['eat_food'] = 1
          else:
            features['eat_food'] = 0
        
        #如果敌人是恐吓状态，应该可以继续前进到达安全区域
        # features['check_trapped'] = 0
        # if self.bfs_for_divided_points(gameState,t='is_trapped'):
        #     features['check_trapped'] = 1
        
        #features['successorScore'] = -len(foodList)  
    #############################吃豆豆
        # Compute distance to the nearest food
        if features['eat_food']==0:
          if len(foodList) > 0: # This should always be True,  but better safe than sorry
              minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
          features['distanceToFood'] = minDistance/(walls.width * walls.height)
        
        #到死角的距离
        
        if features['defenders']>0:
            features['distanceOutofDeadcorner'] = float('inf')
            for deadcorner in self.deadCorners.keys():
                dis_to_dead = self.getMazeDistance(myPos, deadcorner)
                tunnel_len = self.deadCorners[deadcorner]
                dis_out = tunnel_len - dis_to_dead
                if dis_to_dead<tunnel_len:
                  features['inTunnel'] = 1
                  features['distanceOutofDeadcorner'] = min(features['distanceOutofDeadcorner'],dis_out)
            if features['inTunnel']:
              features['distanceOutofDeadcorner'] /= (walls.width * walls.height)
              features['CornerAvoid'] = features['distanceOutofDeadcorner']/features['defendersDistance'] #负权
              features['CornerAvoid'] /= (walls.width * walls.height)

                #print(myPos,dis_out*2,features['defendersDistance'],dis_to_dead,tunnel_len,deadcorner)
                
                # self.debugDraw(myPos,[1,0,0])
        
               


    #############################吃胶囊
        Capsules = self.getCapsules(successor)
        features['numCapsules_opponent'] = len(Capsules)
        if len(Capsules) > 0: #如果等于0，这个特征也不存在了
          dists = [self.getMazeDistance(myPos, a) for a in Capsules]
          features['CapsulesDistance_opponent'] = min(dists)/(walls.width * walls.height)
    ############################构造性特征
        features['dividesDistance'] /= (walls.width * walls.height) #都要规范化

        if features['defenders']!=0:  #有防守者才要考虑的东西 <=5 此时防守者距离也是存在的
          if myPos in Capsules:
            features['Capsule_eaten'] = 1
          elif features['numCapsules_opponent']>0:
            features['Capsule_attract'] = features['CapsulesDistance_opponent']
            features['Capsule_attract'] /= features['defendersDistance']
            features['Capsule_attract'] /= (walls.width * walls.height)
            
          features['divides_attract'] = features['dividesDistance']*features['numCarrying']/features['defendersDistance']
          features['divides_attract'] /= (walls.width * walls.height)
        elif myPos in Capsules:
          features['Capsule_eaten'] = -1 #如果防守者不多那就不要吃掉胶囊的意思
        return features
        
              

          

        

        

        
        


    
    

    

    


