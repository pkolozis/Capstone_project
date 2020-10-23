import networkx as nx
from scipy import spatial
from env import *
import matplotlib.pyplot as plt


def optimal_path(env,visualize=True,equal_weights=True):
    '''
    Description:
        Function that creates a graph and computes the shortest path using Dijkstra algorithm
    
    Inputs:    
        env: Environmetnt 
        visualize: boolean. Whether to print the path or not
        equal_weights: boolean. Whether to create a graph with all edges weights equal to 1 or
                        diagonal edges equal to math.sqrt(2)
    Returns:
        The lenght of the shortest path
    '''
    # get list of points
    points = env.map_['geometry'].apply(lambda g:[g.x,g.y]).tolist()
    #spatially organising the points on a tree for quick nearest neighbors calc
    kdtree = spatial.KDTree(points)
    # get neighbors with distance math.sqrt((2e5**2)+(2e5**2))
    x = kdtree.query_ball_tree(kdtree,r=math.sqrt((2e5**2)+(2e5**2)))
    if equal_weights:
        # create empty graph
        G = nx.Graph()
        for i in range(len(x)):
            # remove the node  that correspond to the self 
            index = x[i].pop(np.where(np.array(x[i])==i)[0][0])
            for j in x[i]: 
                # create edges with neighbors with weight 1 
                G.add_edge((np.round(env.map_.loc[index].geometry.x),np.round(env.map_.loc[index].geometry.y)),
                           (np.round(env.map_.loc[j].geometry.x),np.round(env.map_.loc[j].geometry.y)),weight=1)
    else:
        G = nx.Graph()
        for i in range(len(x)):
            index = x[i].pop(np.where(np.array(x[i])==i)[0][0])
            for j in x[i]:
                if math.sqrt((env.map_.loc[index].geometry.x -env.map_.loc[j].geometry.x)**2 +
                             (env.map_.loc[index].geometry.y -env.map_.loc[j].geometry.y)**2) > 2e5:
                    # add to diagonal neighbors weight math.sqrt((2e5**2)+(2e5**2))
                    G.add_edge((np.round(env.map_.loc[index].geometry.x),np.round(env.map_.loc[index].geometry.y)),
                           (np.round(env.map_.loc[j].geometry.x),np.round(env.map_.loc[j].geometry.y)),
                           weight=math.sqrt(2))
                else:
                    # add to diagonal neighbors weight 1
                    G.add_edge((np.round(env.map_.loc[index].geometry.x),np.round(env.map_.loc[index].geometry.y)),
                           (np.round(env.map_.loc[j].geometry.x),np.round(env.map_.loc[j].geometry.y)),weight=1)
    # compute the shortest path                  
    state = nx.shortest_path(G,source=(env.start_longitude, env.start_latitude),target=(env.end_longitude,env.end_latitude),
                             weight='weight', method='dijkstra')
    if visualize:
        fig, ax = plt.subplots(figsize=(20, 20))
        env.map_.plot(ax=ax)


        for i in state:
            ax.scatter(i[0],i[1],color='red',marker='*',s=150)
        ax.scatter(i[0],i[1],color='red',marker='*',s=150,label="path")
        ax.scatter(env.start_longitude, env.start_latitude,color='black',s=150,label='Starting Port')
        ax.scatter(env.end_longitude,env.end_latitude,color='orange',s=150,label='Ending Port')


        ax.set_title("Ocean grid: Robinson Coordinate Reference System,",fontsize = 20)
        ax.set_xlabel("X Coordinates (meters)",fontsize = 20)
        ax.set_ylabel("Y Coordinates (meters)",fontsize = 20)
        ax.legend()

        for axis in [ax.xaxis, ax.yaxis]:
            formatter = ScalarFormatter()
            formatter.set_scientific(False)
            axis.set_major_formatter(formatter)
        plt.show()
    # -1 because len(state) contains the starting point 
    return len(state)-1
