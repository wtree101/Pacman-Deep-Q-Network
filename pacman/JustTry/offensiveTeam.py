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

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveReflexAgent', second = 'OffensiveReflexAgent_support'):
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
    neibors = Actions.getLegalNeighbors(pos,walls)
    possible_actions = neibors
    visited = set()
    
    tunnel_len = 0
    while len(possible_actions)==2:
      visited.add(pos) #加入当前位置
      pos = possible_actions[0]
      neibors = Actions.getLegalNeighbors(pos,walls)
      possible_actions = []
      for neibor in neibors:
        if (not neibor in visited):
          possible_actions.append(neibor)
      tunnel_len += 1
    
    if tunnel_len>0:
      self.deadCorners[pos] = tunnel_len
    
    
    
  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)
    self.divides = []
    self.bfs_for_divided_points(gameState)
    self.deadCorners = {}
    walls = gameState.getWalls()
    for food in self.getFood(gameState).asList():
      self.P_deadcorners(food,walls)
    
    #print(self.deadCorners)

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
      
    if action == Directions.STOP: features['stop'] = 1
   # rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
   # if action == rev: features['reverse'] = 1

    dists = [self.getMazeDistance(myPos, enemy.getPosition()) for enemy in enemies]
    features['enemyDistance'] = min(dists)




    return features

  def getWeights(self, gameState, action,features):
    #要小心不要为了追人跑出防守距离。
    weights_dict = {'numInvaders': -1000, 'onDefense': 500, 'invaderDistance': -20, 'stop': -1,
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
#################危机
###########1 可走的路少 -- 但是感觉很难实现的
    actions = successor.getLegalActions(self.index)
    features['available_actions'] = len(actions)

    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    defenders = [a for a in enemies if (not a.isPacman) and a.getPosition() != None and a.scaredTimer<=3] #被恐吓的幽灵不算defenders
    features['defenders'] = len(defenders)

    features['check_died'] = 0
    if len(defenders) > 0: #如果等于0，这个特征就不存在了。
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in defenders if a.scaredTimer<=3 ] #被恐吓的幽灵不需要害怕
      if (len(dists)>0):
        features['defendersDistance'] = min(dists)
    
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
    
    # #到死角的距离
    # features['possibly_trapped'] = 0
    # if features['defenders']>0:
    #   for deadcorner in self.deadCorners.keys():
    #     dis_to_dead = self.getMazeDistance(myPos, deadcorner)
    #     tunnel_len = self.deadCorners[deadcorner]
    #     dis_out = tunnel_len - dis_to_dead
    #     if (dis_out*2<=features['defendersDistance']):
    #       features['possibly_trapped'] = 1
    #       break


#############################吃胶囊
    Capsules = self.getCapsules(successor)
    features['numCapsules_opponent'] = len(Capsules)
    if len(Capsules) > 0: #如果等于0，这个特征也不存在了
      dists = [self.getMazeDistance(myPos, a) for a in Capsules]
      features['CapsulesDistance_opponent'] = min(dists)
############################构造性特征
    features['Capsule_attract'] = features['CapsulesDistance_opponent']*features['defenders']
    if features['defenders']!=0:
      features['Capsule_attract'] /= features['defendersDistance']

    

    

    
    

    return features

  def getWeights_offensive(self, gameState, action,features):
    #s = self.getSuccessor(gameState, action)
    myState = gameState.getAgentState(self.index)
    score_now = self.getScore(gameState)
    weights_dict = {'successorScore': 200, 'distanceToFood': -5,'check_died':-500,'check_trapped':-300,'defendersDistance':1
    ,'dividesDistance':-1,'CapsulesDistance_opponent':-1,'numCapsules_opponent':-200,
    'Capsule_attract':-6,'stop': -50,'numInvaders':-1000} #-300是为了如果有机会吃掉它，不因为吃了之后胶囊的惩罚而止步不前
    #危险状态
    if (score_now + myState.numCarrying - features['invadercarrying']>0) and myState.numCarrying>0: #考虑返回,但要保证分界点处可以进行——要吃掉最后一个豆豆
      weights_dict['dividesDistance'] = -10
      if features['defenders']>1:
        weights_dict['dividesDistance'] = -210 #绝对地回家
    if (score_now + myState.numCarrying - features['invadercarrying']>0) and myState.numCarrying>20: #考虑返回了,但要保证分界点处可以进行——要吃掉最后一个豆豆
      weights_dict['dividesDistance'] = -10 #比食物吸引力大,但如果一步就能吃到一个食物，还是会去吃
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


class OffensiveReflexAgent_support(OffensiveReflexAgent):
  def getFeatures_offensive(self, gameState, action,features): #在defensive基础上改。feacture一定要对于下一个状态而言
    
    successor = self.getSuccessor(gameState, action)
    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()
#################分数
    foodList = self.getFood(successor).asList()    
    features['successorScore'] = -len(foodList)#self.getScore(successor)
#################危机
###########1 可走的路少 -- 但是感觉很难实现的
    actions = successor.getLegalActions(self.index)
    features['available_actions'] = len(actions)
    friend = (self.index + 2)%4
    if friend==0:
        friend==4
    friendState = successor.getAgentState(self.index)
    friendPos = friendState.getPosition()

    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    defenders = [a for a in enemies if (not a.isPacman) and a.getPosition() != None and a.scaredTimer<=3] #被恐吓的幽灵不算defenders
    features['defenders'] = len(defenders)

    features['check_died'] = 0
    if len(defenders) > 0: #如果等于0，这个特征就不存在了。
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in defenders if a.scaredTimer<=3 ] #被恐吓的幽灵不需要害怕
      if (len(dists)>0):
        features['defendersDistance'] = min(dists)
      
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
    far_food_dis = 0

    minDistance = float('inf')
      
    if len(foodList) > 0: # This should always be True,  but better safe than sorry
        for food in foodList:
            dis_friend = self.getMazeDistance(friendPos, food)
            if dis_friend>far_food_dis:
                far_food = food
                far_food_dis = dis_friend
            dis_me = self.getMazeDistance(myPos, food)
            if (dis_me<dis_friend):
                minDistance = min(minDistance,self.getMazeDistance(myPos, food))
        
    features['dishelpfriend'] = self.getMazeDistance(myPos, far_food)
    if  minDistance != float('inf'):
        features['distanceToFood'] = minDistance
    


    
    
    #到死角的距离
    # features['possibly_trapped'] = 0
    # if features['defenders']>0:
    #   for deadcorner in self.deadCorners.keys():
    #     dis_to_dead = self.getMazeDistance(myPos, deadcorner)
    #     tunnel_len = self.deadCorners[deadcorner]
    #     dis_out = tunnel_len - dis_to_dead
    #     if (dis_out*2<=features['defendersDistance']):
    #       features['possibly_trapped'] = 1
    #       break


#############################吃胶囊
    Capsules = self.getCapsules(successor)
    features['numCapsules_opponent'] = len(Capsules)
    if len(Capsules) > 0: #如果等于0，这个特征也不存在了
      dists = [self.getMazeDistance(myPos, a) for a in Capsules]
      features['CapsulesDistance_opponent'] = min(dists)
############################构造性特征
    features['Capsule_attract'] = features['CapsulesDistance_opponent']*features['defenders']
    if features['defenders']!=0:
      features['Capsule_attract'] /= features['defendersDistance']

    

    

    
    

    return features

  def getWeights_offensive(self, gameState, action,features):
    #s = self.getSuccessor(gameState, action)
    myState = gameState.getAgentState(self.index)
    score_now = self.getScore(gameState)
    weights_dict = {'successorScore': 200, 'distanceToFood': -5,'check_died':-500,'check_trapped':-300,'defendersDistance':1
    ,'dividesDistance':-1,'CapsulesDistance_opponent':-1,'numCapsules_opponent':-200,
    'Capsule_attract':-6,'stop': -50,'numInvaders':-1000,'dishelpfriend':-5} #-300是为了如果有机会吃掉它，不因为吃了之后胶囊的惩罚而止步不前
    #危险状态
    if (score_now + myState.numCarrying - features['invadercarrying']>0) and myState.numCarrying>0: #考虑返回,但要保证分界点处可以进行——要吃掉最后一个豆豆
      weights_dict['dividesDistance'] = -15
      if features['defenders']>1:
        weights_dict['dividesDistance'] = -210 #绝对地回家
    if (score_now + myState.numCarrying - features['invadercarrying']>0) and myState.numCarrying>20: #考虑返回了,但要保证分界点处可以进行——要吃掉最后一个豆豆
      weights_dict['dividesDistance'] = -20 #比食物吸引力大,但如果一步就能吃到一个食物，还是会去吃
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



class DummyAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    CaptureAgent.registerInitialState(self, gameState)

    '''
    Your initialization code goes here, if you need any.
    '''


  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """
    actions = gameState.getLegalActions(self.index)

    '''
    You should change this in your own agent.
    '''

    return random.choice(actions)

