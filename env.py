import gym
import math
from gym import spaces
import numpy as np
import geopandas as gpd
from shapely.geometry import MultiPoint,Point
from matplotlib.ticker import ScalarFormatter
import matplotlib.pyplot as plt
import warnings

class sea(gym.Env): 
    """
    Description:
        
        
    Observation:
        Type: Box(2)
        Num     Observation               Min                     Max
        0       Vessel's Longitude        -18e6                   18e6
        1       Vessel's Latitude         -9e6                    9e6
        
        In meters (Robinson projection)
    Actions:
        Type: Discrete(8)
        Num   Action
        0     Move to the North
        1     Move to North East
        2     Move to the East
        3     Move to South East
        4     Move to the South
        5     Move to South West
        6     Move to the West
        7     Move to North West
               
    Reward:
        Reward is -1/srt(2e5^2 + 2e5^2) for every step taken in cardinal direction.
        Reward is -1 for every diagonal move.
        Reward (-c,1): (total_distance - current_position)/total_distance
        Reward is [0,-0.5] for every change of direction. 
        Reward is 100 for the termination step.
        
    Starting State:
        Assigned from the user a starting and ending point. 
    Episode Termination:
        Vessel reaches the ending point.
   
    """
     
    metadata = {'render.modes': ['human']}    
    def __init__(self,start_longitude,start_latitude,end_longitude,end_latitude):
        self.start_longitude = start_longitude
        self.start_latitude = start_latitude
        self.end_latitude = end_latitude
        self.end_longitude = end_longitude
#       distance between starting and ending port
        self.total_distance = math.sqrt((end_longitude-start_longitude)**2 + (end_latitude-start_latitude)**2)
        self.state = None
        self.previous_action = None
        self.action_space = spaces.Discrete(8)
        self.map_ = create_map()
        obs_max = [18e6,9e6]
        obs_min = [-18e6,-9e6]
        self.observation_space = spaces.Box(np.array(obs_min),np.array(obs_max),dtype=np.float64)
        self.history = {"action": [],"reward": []}
        try:
            with open('edges.npy','rb') as fin:
                edges = np.load(fin,allow_pickle=True).tolist()
#             self.left_edges = edges['left_edges']
            self.right_edges = edges['right_edges']
        except:
            warnings.warn("Map Edges are not used")
#             self.left_edges = None
            self.right_edges = None
        
#   Compute the distance between starting and ending point
    def compute_distance(self,position):
        return math.sqrt((self.end_longitude-position[0])**2 + (self.end_latitude-position[1])**2)
    
    def reset(self):
        self.state = self.start_longitude,self.start_latitude
        self.previous_action = None
        return  np.array(self.state) 
    
    #   check whether a ship position is in the sea or land 
    def isvalid(self, position):
        return len(self.map_[self.map_.geometry.geom_almost_equals(Point(position[0],position[1]))]) != 0
    
    def step(self, action):
        if self.state == None:
            raise ValueError('Cannot call env.step() without calling reset()')
    #   Teleport to the other side of the map in the edges          
        if np.all(abs(np.array(self.state)) == self.right_edges,axis=1).sum() == 1:
            self.state = -self.state[0],self.state[1]     
        if action == 0:  # move to the North
            move = self.state[0],self.state[1]+2e5
        elif action == 1:  # move to the North East
            move = self.state[0]+2e5,self.state[1]+2e5
        elif action == 2: # move to the East
            move = self.state[0]+2e5,self.state[1]
        elif action == 3: # move to the South East
            move = self.state[0]+2e5,self.state[1]-2e5
        elif action == 4: # move to the South
            move = self.state[0],self.state[1]-2e5
        elif action == 5: # move to the South West
            move = self.state[0]-2e5,self.state[1]-2e5
        elif action == 6:  # move to the West
            move = self.state[0]-2e5,self.state[1]
        elif action == 7: # move to the North West
            move = self.state[0]-2e5,self.state[1]+2e5
        else:
            raise ValueError('Actions are between 0 and 7')
        if self.isvalid(move):
            self.state = move
            done = bool(self.state == (self.end_longitude,self.end_latitude))
            a = 0.6
            if not done:
                if action%2 == 0:
                    reward = (- (1-a)*(2e5/math.sqrt(2e5**2+2e5**2)) + a*((self.total_distance - self.compute_distance(move))/self.total_distance))
                else:
                    reward = (-(1-a)*1 + a*((self.total_distance - self.compute_distance(move))/self.total_distance))
                if self.did_turn(action):
                    reward += -self.angle_turn(action)
            else:
                reward = 100
        else:
          done = True
          reward = -10
        self.history['action'].append(action)
        self.history['reward'].append(reward)
            
        self.previous_action = action 
        return np.array(self.state), reward, done, {}
#   Check whether the vessel did turn 
    def did_turn(self,action):
        if self.previous_action == None:
            return False
        elif self.previous_action == action :
            return False
        else:
            return True        
#   Mesure the turning angle
    def angle_turn(self,action):
        r = abs((action/8 - self.previous_action/8))
        return r if r < 0.5 else 1-r
    
    
    def render(self,mode='human'):
        fig, ax = plt.subplots(figsize=(20, 20))
        self.map_.plot(ax=ax)
        point = Point(self.state[0],self.state[1])
        start = Point(self.start_longitude,self.start_latitude)
        end = Point(self.end_longitude,self.end_latitude)

        ax.scatter(point.x,point.y,color='red',marker='*',s=150,label='Ship\'s location')
        ax.scatter(start.x,start.y,color='black',s=150,label='Starting Port')
        ax.scatter(end.x,end.y,color='orange',s=150,label='Ending Port')
        

        ax.set_title("Ocean grid: Robinson Coordinate Reference System,",fontsize = 20)
        ax.set_xlabel("X Coordinates (meters)",fontsize = 20)
        ax.set_ylabel("Y Coordinates (meters)",fontsize = 20)
        ax.legend()

        for axis in [ax.xaxis, ax.yaxis]:
            formatter = ScalarFormatter()
            formatter.set_scientific(False)
            axis.set_major_formatter(formatter)
        plt.show()

# Create the grid
def create_map(n=5e-6):
    ocean = gpd.read_file('C://Users//potis//Desktop//b//ne_10m_ocean_scale_rank.shp')
    ocean = ocean.to_crs('ESRI:54030')
    result=[]
    for i in range(len(ocean)):
        xmin, ymin, xmax, ymax = ocean.bounds.loc[i].T.values
        x = np.arange(np.floor(xmin * n) / n, np.ceil(xmax * n) / n, 1 / n)
        y = np.arange(np.floor(ymin * n) / n, np.ceil(ymax * n) / n, 1 / n)    

        grid = np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])
        points = MultiPoint(grid)

        if ocean.geometry[i].is_valid:
            result.append(points.intersection(ocean.geometry.loc[i]))
        else:
            result.append(points.intersection(ocean.geometry.loc[i].buffer(0)))
        results = [j for i in result for j in i]
    return gpd.GeoDataFrame(results,columns=['geometry'],crs=ocean.crs)
