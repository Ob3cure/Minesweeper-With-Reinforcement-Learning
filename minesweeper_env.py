import numpy as np

class Minesweeper_Env:
    '''RL Environment

    :mine_map: has the positions of the mines
    :mask: has the revealed entries
    :info_map: has the informations of the map
    :observation: current observation,
      entries with mines => -1
      entries unknown => -2
    '''
    def __init__(self,row,column,mine_ratio,random_seed = 42):
        if mine_ratio > 1 or mine_ratio < 0:
            raise ValueError("mine ratio should be within 0 and 1")
        self.row = row
        self.column = column
        self.n = row * column
        self.mine_ratio = mine_ratio
        self.mine_count = (int)(self.n * mine_ratio)
        self._set_random(random_seed)

        self.mine_map = np.zeros((row,column),dtype=int)
        self.mask = np.zeros((row,column),dtype=int)
        self.info_map  = np.zeros((row,column),dtype=int)

        self._action_space = [(i,j) for j in range(self.column) for i in range(self.row)]
        self.action_space = {a:1 for a in self._action_space}
        self.observation = self.mask - 2
        self.reward = 0
        self.terminated = False

    def _set_random(self,random_seed):
        self.random_seed = random_seed
        self.np_random = np.random.default_rng(self.random_seed)

    def _place_mines(self):
        self.mine_map = np.zeros((self.row,self.column),dtype=int)
        _idx = self.np_random.choice(len(self._action_space),self.mine_count,replace=False)
        self._mine_index = np.array([self._action_space[_idx[i]] for i in range(len(_idx))])

        self.mine_map[tuple(self._mine_index.T)] = 1

    def _count_neighbour_mines(self,i,j):
        if self.mine_map[i,j] == 1:
            return -1

        return self.mine_map[max(i-1,0):i+2,max(j-1,0):j+2].sum()

    def _generate_info(self):
        self.info_map = np.array([[self._count_neighbour_mines(i,j) for j in range(self.column)] for i in range(self.row)])

    def reset(self):
        '''Reset to the begining without generating a new environment.
        '''
        self.np_random = np.random.default_rng(self.random_seed)
        self.mask = np.zeros((self.row,self.column),dtype=int)
        self.observation = self.mask - 2
        self.action_space = {a:1 for a in self._action_space}
        self.reward = 0
        self.terminated = False

    def make(self,start:tuple=None,random_seed=None):
        '''Generate a new environment。Pass an index tuple to ensure that your first step is safe.

        :param start: The starting position and no mine is guarentee.
        :type start: tuple
        :param random_seed: Default None. The random seed will be used if it is not None.
        :type random_seed: int
        '''
        self.reset()
        if random_seed is not None:
            self._set_random(random_seed)

        self._place_mines()
        if start is not None:
            self.mine_map[start] = 0
        self._generate_info()
        self.reset()

        return self.observation

    def get_status(self):
        '''
        :return: observation, mine_count, reward, terminated, mine_info
        :rtype: np.ndarray, int, float, bool, np.ndarray
        '''

        return self.observation,self.mine_count, self.reward, self.terminated, self.info_map

    def toggle(self,action):
        if action[0] < 0 or action[0] >= self.row or action[1] < 0 or action[1] >= self.column:
            return

        if self.mask[action]:
            reward = -5
            return reward

        self.observation[action] = self.info_map[action]
        self.mask[action] = 1
        self.terminated = True if np.array_equal((1 - self.mask),self.mine_map) or self.terminated else False
        self.action_space.pop(action,None)

        reward = 0
        if self.info_map[action] == -1:
            self.terminated = True
            reward = -1

        if self.info_map[action] == 0:
            i = action[0]
            j = action[1]
            reward = 1
            self.toggle((i - 1,j))
            self.toggle((i + 1,j))
            self.toggle((i,j - 1))
            self.toggle((i,j + 1))
            self.toggle((i - 1,j - 1))
            self.toggle((i - 1,j + 1))
            self.toggle((i + 1,j - 1))
            self.toggle((i + 1,j + 1))

        return reward

    def step(self,action:tuple):
        '''Toggle an entry. \n
        Do not trigger mine => reward = 1 \n
        Trigger mine => reward = -1 \n
        Invalid operation => reward = -5
        Entry has already been toggled => reward = -5

        :param action: The entry to be toggled
        :type action: tuple
        :return: observation, reward, terminated, mine_info
        :rtype: np.ndarray, float, bool, np.ndarray
        '''

        #If the action is invalid we should penalize the model heavily
        if action[0] >= self.row or action[0] < 0 or action[1] >= self.column or action[1] < 0:
            self.reward = -5
            return self.observation, self.reward, self.terminated, self.info_map

        self.reward = self.toggle(action)
        return self.get_status()

    def sample_action(self):
        action_space = list(self.action_space.keys())
        return action_space[self.np_random.choice(len(action_space),1,replace=False)]

    def clone(self):
        env = Minesweeper_Env(self.row,self.column,self.mine_ratio,self.random_seed)
        env.observation = self.observation.copy()
        env.terminated = self.terminated
        env.reward = self.reward
        env.action_space = self.action_space.copy()
        env.mine_map = self.mine_map.copy()
        env.mask = self.mask.copy()
        env.info_map = self.info_map.copy()

        return env

    def render(self,reveal_all=False):
        sym = {
            -2: '■',
            -1: '✱ ',
            0: '  ',
            1: '1 ',
            2: '2 ',
            3: '3 ',
            4: '4 ',
            5: '5 ',
            6: '6 ',
            7: '7 ',
            8: '8 ',
        }
        for i in range(self.row):
            if(not reveal_all):
                print(''.join(sym[int(self.info_map[i, j])] if self.mask[i, j]
                        else '■' for j in range(self.column)))
            else:
                print(''.join(sym[int(self.info_map[i, j])] for j in range(self.column)))
        print()
