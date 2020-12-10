import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

class Simulation:
    
    def __init__(self, city_size=5, min_ambulance=3, accident_prob=1, p_int=0.8, max_time=4, seed=42):
        
        np.random.seed(seed)
        
        # simulation parameters
        self.city_size = city_size
        self.min_ambulance = min_ambulance
        self.accident_prob = accident_prob
        self.p_int = p_int
        self.p_unint = (1-p_int)/2
        self.max_time = max_time
        self.accident_location = None
        self.ambulance_reached = False
        self.ambulances = {}
        self.response_time = 99999
        
        # create a new city
        # self.city_vertices = np.zeros([self.city_size, self.city_size])
        self.city_horizontal_edges = np.random.randint(low=1, high=self.max_time+1, 
                                                       size=[self.city_size, self.city_size-1])
        self.city_vertical_edges = np.random.randint(low=1, high=self.max_time+1, 
                                                     size=[self.city_size-1, self.city_size])
        
        # initialize time steps
        self.turn = 0

        # initialize MDP variables
        self.policies = {}
        self.current_policy = {}
        self.actions = ["north", "east", "south", "west"]
        self.all_states = [] #generate list of all possible states
        for m in range(self.city_size):
            for n in range(self.city_size):
                self.all_states.append((m, n))

    def step(self):
            
        # increment time step
        self.turn += 1
        
        # pre-accident state
        if self.accident_location is None:
            
            # accident happens in current time step
            if np.random.uniform(low=0.0, high=1.0) <= self.accident_prob:
                
                # generate accident location
                self.accident_location = np.random.randint(low=0, high=self.city_size, size=2)

                # generate optimal policies
                self.generate_policy(tuple(self.accident_location))

                # saved policy corresponding to accident
                self.current_policy = self.policies[(self.accident_location[0], self.accident_location[1])]

                # pick ambulance based on accident location
                self.responding_ambulance_id, self.response_time = self.pick_ambulance((self.accident_location[0], self.accident_location[1]))

        # post-accident state
        else:

            responding_ambulance = self.ambulances[self.responding_ambulance_id]
            responding_ambulance_loc = responding_ambulance.location.copy()
            responding_ambulance_action = self.current_policy[(responding_ambulance_loc[0], responding_ambulance_loc[1])]
            
            self.move_ambulance(self.responding_ambulance_id, responding_ambulance_action)

            if (responding_ambulance.location == self.accident_location).all():
                self.ambulance_reached = True

        return

    def random_ambulance(self, verbose=False):
        """
        initializes a dict with ambulances placed randomly (in lieu of solving a CSP)
        """
        # initialize a dict to save ambulance IDs and locations
        ambulances = self.ambulances

        # start with randomly placed ambulances
        for i in range(self.min_ambulance):
            
            ambulance_id = i
            ambulance_location = np.random.randint(low=0, high=self.city_size, size=2)
            ambulances[i] = Ambulance(ambulance_id, ambulance_location)

        if verbose:
            self.show_city()

        return
    
    def csp_ambulance(self, guarantee_time=20, csp_iterations=10, num_blocks=2, verbose=False):
        """
        initializes a dict with ambulances and solves the CSP
        """
        # initialize a dict to save ambulance IDs and locations
        ambulances = self.ambulances

        # start with randomly placed ambulances
        for i in range(self.min_ambulance):
            
            ambulance_id = i
            ambulance_location = np.random.randint(low=0, high=self.city_size, size=2)
            ambulances[i] = Ambulance(ambulance_id, ambulance_location)

        if verbose:
            self.show_city()

        counter = 0
        while True:

            times_to_loc = self.find_times_to_loc(ambulances)
                    
            what_times = np.min(times_to_loc, axis=-1)
            which_ambs = np.argmin(times_to_loc, axis=-1)
            max_time = np.max(what_times)

            if max_time <= guarantee_time:
                break
            else:
                self.change_amb_locs(ambulances, max_time, what_times, which_ambs, num_blocks)

            # add an ambulance if constraint still not satisfied
            if (counter != 0) and (counter % csp_iterations == 0):
                
                # place new ambulance randomly
                i += 1
                ambulance_id = i
                ambulance_location = np.random.randint(low=0, high=self.city_size, size=2)
                ambulances[i] = Ambulance(ambulance_id, ambulance_location)

            if verbose:
                self.show_city()

            counter += 1

        return 
         
    def find_times_to_loc(self, ambulances):
        """
        ambulances: dict with all current ambulance objects
        max_time: max response time to reach any vertex in the city
        what_times: m-by-n matrix with response times to reach each vertex
        which_ambs: m-by-n matrix with ID of the ambulance that takes the least time
        """
        num_ambulances = len(ambulances.keys())
        times_to_loc = np.empty([self.city_size, self.city_size, num_ambulances])

        for m in range(self.city_size):
            for n in range(self.city_size):
                for i in range(num_ambulances):
                    
                    amb_location = ambulances[i].location
                    times_to_loc[m,n,i] = self.triangular_distance(amb_location, np.array([m,n]))

        return times_to_loc

    def triangular_distance(self, from_loc, to_loc):
        """
        returns time taken to traverse the distance as an int
        """
        # calculate North-South travel time
        if from_loc[0] == to_loc[0]:
            vert_travel_time = 0
        elif from_loc[0] < to_loc[0]:
            vert_travel_time = np.sum(self.city_vertical_edges[from_loc[0]:to_loc[0], from_loc[1]])
        else:
            vert_travel_time = np.sum(self.city_vertical_edges[to_loc[0]:from_loc[0], from_loc[1]])

        # calculate East-West travel time
        if from_loc[1] == to_loc[1]:
            horiz_travel_time = 0
        elif from_loc[1] < to_loc[1]:
            horiz_travel_time = np.sum(self.city_horizontal_edges[to_loc[0], from_loc[1]:to_loc[1]])
        else:
            horiz_travel_time = np.sum(self.city_horizontal_edges[to_loc[0], to_loc[1]:from_loc[1]])

        # calculate total triangular travel time
        triangular_travel_time = int(vert_travel_time + horiz_travel_time)
        
        return triangular_travel_time

    def change_amb_locs(self, ambulances, max_time, what_times, which_ambs, num_blocks):
        
        farthest_vertex = np.where(what_times == what_times.max())
        amb_id = which_ambs[farthest_vertex[0][0], farthest_vertex[1][0]] # take first if multiple
        amb_loc = ambulances[amb_id].location

        for n in range(num_blocks):

            vertical_distance = farthest_vertex[0][0] - amb_loc[0]
            horizontal_distance = farthest_vertex[1][0] - amb_loc[1]

            mn_distances = [vertical_distance, horizontal_distance]
            mn_selector = np.argmax([abs(vertical_distance), abs(horizontal_distance)])
            amb_loc[mn_selector] += np.sign(mn_distances[mn_selector])

        return

    def pick_ambulance(self, target_loc):
        """
        returns ID of the ambulance that should respond to the accident
        target_loc: tuple of accident location
        """
        # new temp dictionary for saving times
        distances = {}

        # loop over each ambulance
        for temp_id in self.ambulances.keys():
            
            temp_location = self.ambulances[temp_id].location.copy()
            temp_distance = 0

            # move until reaching accident location
            while (temp_location[0], temp_location[1]) != target_loc:
                
                direction = self.current_policy[(temp_location[0], temp_location[1])]
                temp_location, time_spent = self.next_location(temp_location, direction)
                temp_distance += time_spent

            # save time taken by the ambulance
            distances[temp_id] = temp_distance

            # determine responding ambulance
            responding_id = min(distances, key=distances.get)
            response_time = distances[responding_id]

        return responding_id, response_time
    
    def next_location(self, current_location, direction):
        """
        returns next location as a Numpy array given current location as a Numpy array
        also returns time spent moving along the corresponding edge
        direction: ["north", "east", "south", "west"]
        """

        # default values if nothing happens
        next_location = current_location.copy()
        time_spent = 1

        # move north
        if direction == "north":
            
            if current_location[0] != 0:
                
                time_spent = self.city_vertical_edges[current_location[0]-1, current_location[1]]
                next_location[0] -= 1
        
        # move south
        elif direction == "south":
            
            if current_location[0] != self.city_size-1:
                
                time_spent = self.city_vertical_edges[current_location[0], current_location[1]]
                next_location[0] += 1
        
        # move east
        elif direction == "east":
            
            if current_location[1] != self.city_size-1:

                time_spent = self.city_horizontal_edges[current_location[0], current_location[1]]
                next_location[1] += 1

        # move west     
        elif direction == "west":
            
            if current_location[1] != 0:

                time_spent = self.city_horizontal_edges[current_location[0], current_location[1]-1]
                next_location[1] -= 1

        return next_location, time_spent

    def move_ambulance(self, ambulance_id, requested_direction):
        """
        moves given ambulance in the direction stochastically determined from the requested direction
        """
        
        p_int = self.p_int          # intended direction
        p_unint = self.p_unint      # unintended, perpendicular direction

        # given requested direction, pick actual direction
        if requested_direction == "north":
            
            direction = np.random.choice(["north", "east", "west"], p=[p_int, p_unint, p_unint])

        elif requested_direction == "south":
            
            direction = np.random.choice(["south", "east", "west"], p=[p_int, p_unint, p_unint])

        elif requested_direction == "east":
            
            direction = np.random.choice(["east", "north", "south"], p=[p_int, p_unint, p_unint])

        elif requested_direction == "west":
            
            direction = np.random.choice(["west", "north", "south"], p=[p_int, p_unint, p_unint])

        ambulance = self.ambulances[ambulance_id]
        current_location = ambulance.location
        
        next_location, time_spent = self.next_location(current_location, direction)
        
        ambulance.location = next_location 

        return

    def get_neighbors(self, s):
        """
        Returns neighboring states to the north, south, east, and west of state s
        """
        # initialize neighbors
        north_neighbor = (s[0]-1, s[1])
        south_neighbor = (s[0]+1, s[1])
        east_neighbor = (s[0], s[1]+1)
        west_neighbor = (s[0], s[1]-1)

        # account for edge cases
        if s[0] == 0:
            north_neighbor = (s[0],s[1])
        if s[0] == self.city_size - 1:
            south_neighbor = (s[0],s[1])
        if s[1] == 0:
            west_neighbor = (s[0],s[1])
        if s[1] == self.city_size - 1:
            east_neighbor = (s[0],s[1])
        
        # get unique list of neighbors (duplicates occur if in corner)
        neighbors = list(set([north_neighbor, south_neighbor, east_neighbor, west_neighbor]))
        
        return neighbors        

    def T(self, s, a, s_, target_loc):
        """
        returns probability of moving from state s to state s_ given action a
        s: tuple(m,n)
        s_: tuple(m_,n_)
        a: ["north", "east", "south", "west"]
        target_loc: tuple of accident location
        """
        
        p_int = self.p_int           # intended direction
        p_unint = self.p_unint       # unintended, perpendicular direction
        prob_matrix = np.zeros([self.city_size, self.city_size])
        
        # clip function handles agent hitting the wall and staying put
        clip = lambda val: np.clip(val, 0, self.city_size-1)
        
        if s == target_loc:
            prob_matrix[s[0], s[1]] += 1

        elif a == "north":
            
            prob_matrix[clip(s[0]-1), s[1]] += p_int
            prob_matrix[s[0], clip(s[1]-1)] += p_unint
            prob_matrix[s[0], clip(s[1]+1)] += p_unint
            
        elif a == "south":
            
            prob_matrix[clip(s[0]+1), s[1]] += p_int
            prob_matrix[s[0], clip(s[1]-1)] += p_unint
            prob_matrix[s[0], clip(s[1]+1)] += p_unint
        
        elif a == "west":

            prob_matrix[s[0], clip(s[1]-1)] += p_int
            prob_matrix[clip(s[0]-1), s[1]] += p_unint
            prob_matrix[clip(s[0]+1), s[1]] += p_unint
            
        elif a == "east":

            prob_matrix[s[0], clip(s[1]+1)] += p_int
            prob_matrix[clip(s[0]-1), s[1]] += p_unint
            prob_matrix[clip(s[0]+1), s[1]] += p_unint
        
        # fails when probabilities don't sum to 1
        assert np.sum(prob_matrix) == 1
        
        return prob_matrix[s_[0], s_[1]]
    
    def R(self, s, s_, target_loc):
        """
        returns reward of moving from state s to state s_
        s: tuple(m,n)
        s_: tuple(m_,n_)
        target_loc = tuple with (m,n) coordinates of the accident location
        """   
        # start with default 0 reward
        reward = 0
      
        # north movement
        if s[0]-s_[0] == 1:
            reward -= self.city_vertical_edges[s[0]-1, s[1]]        
        # south movement
        elif s[0]-s_[0] == -1:
            reward -= self.city_vertical_edges[s[0], s[1]]       
        # west movement
        elif s[1]-s_[1] == 1:
            reward -= self.city_horizontal_edges[s[0], s[1]-1]
        # east movement
        elif s[1]-s_[1] == -1:
            reward -= self.city_horizontal_edges[s[0], s[1]]
        elif s == s_:
            reward -= 1

        if s_ == target_loc:
            reward += 50

        return reward
    
    def policy_eval(self, policy, U_init, target_loc, gamma=0.9, epsilon=0.0001):
        """
        Evaluates a given policy and returns the updated utility for each state under the policy
        
        Inputs:
        policy = dictionary of actions for each possible state
        U_init = initial utility of each state in a NumPy array
        target_loc = tuple with (m,n) coordinates of the accident location
        gamma = discount factor
        epsilon = threshold for convergence
        """
        U = U_init #intialize utility

        # for each state, calculate its value under the current policy until convergence
        while True:
            delta = 0

            # loop through states
            for s in self.all_states:
                u = 0
                a = policy[s]
                neighbors = self.get_neighbors(s)

                # sum expected utility of moving from current state to all possible future states
                for s_ in neighbors:
                    u += self.T(s, a, s_, target_loc)*(self.R(s, s_, target_loc) + gamma*U[s_[0],s_[1]])

                # update delta if new difference is larger than existing value
                delta = max(delta, abs(u-U[s[0],s[1]])) 
                U[s[0],s[1]] = u #update utility

            # break if values have converged
            if delta*gamma < epsilon*(1-gamma):
                break

        return U

    def policyIteration(self, U_init, target_loc, gamma=0.9):
        """ 
        Implements policy iteration and returns a policy dictionary 

        Inputs:
        U_init = initial value of each state in a NumPy array 
        target_loc = tuple with (m,n) coordinates of the accident location
        gamma = discount factor

        Returns:
        A dictionary where the keys are states and the values are actions (the policy) for 
        reaching a specific accident location
        """
        # iniialize policy w/ random choice of action for each state
        policy = {}
        for s in self.all_states:
            policy[s] = np.random.choice(self.actions)

        # for each state, generate new "best" policy from converged values ("best" = max value)
        while True:
            U = self.policy_eval(policy, U_init, target_loc)
            policy_stable = True

            # loop through states
            for s in self.all_states:  
                policy_a = policy[s]

                # for each state, calculate value of taking each action
                temp_a = [0]*len(self.actions)
                neighbors = self.get_neighbors(s)
                for a in self.actions:
                    for s_ in neighbors:
                        temp_a[self.actions.index(a)] += self.T(s, a, s_, target_loc)*(self.R(s, s_, target_loc) + gamma*U[s_[0],s_[1]])
                
                # "best" action = action that maximizes the value of state
                alt_a = self.actions[temp_a.index(max(temp_a))]
                
                # compare "best" action to current policy
                if policy_a != alt_a:
                    policy_stable = False
                
                # update policy
                policy[s] = alt_a

            # check if policy has converged    
            if policy_stable:
                break
            
        return policy

    def generate_policy(self, target_loc):
        """
        Updates self.policies dictionary with a policy given the location of an accident.
        
        Each policy is a dictionary where the keys represent a potential ambulance location and 
        the values represent the action the ambulance should take to move toward the accident. 
        The self.policies dictionary contains keys that represent accident locations
        (states/vertices of the city) and the values are policy dictionaries.
        """   
        # initial utility for each state = 0 except for accident location, which = 1000      
        U_init = np.zeros([self.city_size, self.city_size])
        U_init[target_loc[0], target_loc[1]] += 1000.
    
        # update policy
        policy = self.policyIteration(U_init, target_loc)

        # update self.policies
        self.policies[target_loc] = policy
        
        return

    def show_policy(self, target_loc):
        try:
            policy = self.policies[target_loc]
            output = []
            for i,v in enumerate(policy.values()):
                if i == target_loc[0]*self.city_size + target_loc[1]:
                    output.append('A')
                else:
                    output.append(v[0])
            
            output = np.array(output).reshape(self.city_size, self.city_size)

            return print(output)
        except KeyError:
            print('Error: No policy exists for accident location {}.'.format(target_loc))
            return

    def show_city(self, size=(7,5)):
        # intialize plot
        fig,ax = plt.subplots(1,1, figsize=size)
        ax.set_ylim([self.city_size-1 + 0.1, -0.1])
        z = 0

        # prepare colors
        cmap = mpl.cm.get_cmap('RdYlGn_r', lut=self.max_time)
        colors = []
        for i in range(self.max_time):
            colors.append(cmap(i))

        # plot east-west streets
        for s in self.all_states:
            if s[1] == self.city_size-1:
                pass
            else:
                c = colors[self.city_horizontal_edges[s[0],s[1]]-1]
                ax.hlines(s[0], s[1], s[1]+1, color=c, linewidth=3)
                z += 1

        # plot north-south streets
        for s in self.all_states:
            if s[0] == self.city_size-1:
                pass
            else:
                c = colors[self.city_vertical_edges[s[0],s[1]]-1]
                ax.vlines(s[1], s[0], s[0]+1, color=c, linewidth=3)
                z += 1

        # plot intersections
        m = [s[0] for s in self.all_states]
        n = [s[1] for s in self.all_states]
        z += 1
        ax.scatter(m, n, zorder=z, color='white')

        # plot accidents and ambulances
        if self.accident_location is not None:
            z += 1
            ax.scatter(self.accident_location[1], self.accident_location[0], color='red', zorder=z, s=90, edgecolors='w')
        for i in self.ambulances.keys():
            z += 1
            ax.scatter(self.ambulances[i].location[1], self.ambulances[i].location[0], color='black', zorder=z, s=90, edgecolors='w')
            ax.text(self.ambulances[i].location[1] + 0.2, self.ambulances[i].location[0] + 0.3, str(i))

        # format axes
        ax.xaxis.tick_top()
        plt.xticks(np.arange(min(m), max(m)+1, 1))
        plt.yticks(np.arange(min(n), max(n)+1, 1))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        # format colorbar
        norm = mpl.colors.Normalize(vmin=0, vmax=self.max_time)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = plt.colorbar(sm, boundaries=np.arange(0,self.max_time+1,1))
        cb.set_label('travel time (min)')

        # add title
        plt.title('City at Timestep {}'.format(self.turn), fontsize=14, fontweight='bold')
        
        return


class Ambulance():
    
    def __init__(self, ambulance_id, ambulance_location):
        
        self.id = ambulance_id
        self.location = ambulance_location