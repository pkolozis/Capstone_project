import numpy as np

def edges(env,up = True,right = True):
    env.reset()
    loc = 0
    end = False
    points = []
# For upward or downward movements
    if up:
        a = 0
    else:
        a = 4
# Finding right or left edges
    if right:
        b = 2
    else:
        b = 6
        
    while True:
        while not end: 
            loc,_,end,_ =  env.step(b)
        _,_,end,_ =  env.step(a)         
        if loc[1] == -200000. or loc[1] == -400000:
            end = True
        points.append(loc)
        if end:
            break
    return points 
def find_edges:
    env = sea(9000000., 8600000.,-5000000.000+2e5, 6e5)
    points1 = edges(env,up = False)
    env = sea(13200000.,  6600000.,-5000000.000+2e5, 6e5)
    points2 = edges(env,up = False)
    env = sea(15600000.0,-4200000.0,-5000000.000+2e5, 6e5)
    points3 = edges(env)
    env = sea(10200000., -8200000.,-5000000.000+2e5, 6e5)
    points4 = edges(env)
    right_edges = points1+points2+points3+points4

    env = sea(-9000000., 8600000.,-5000000.000+2e5, 6e5)
    left1 = edges(env,up=False,right=False)
    env = sea(-12800000.0,6800000.0,-5000000.000+2e5, 6e5)
    left2 = edges(env,up=False,right=False)
    env = sea(-10200000., -8200000.,-5000000.000+2e5, 6e5)
    left3 = edges(env,up=True,right=False)
    left_edges = left1+left2+left3
    with open('edges.npy', 'wb') as fout:
        np.save(fout,{"left_edges":left_edges,"right_edges":right_edges})
