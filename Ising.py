# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 19:25:01 2022

@author: Gabriele
"""

"""A simulation of the 2D Ising model on a square using Metropolis algorithm"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import colors 
from numba import njit


@njit(nogil=True)
def mcmc_m(state, T, steps):
    """A function that returns the absolute value of the average magnetisation

    Args:
        state (list): The initial configuration of the system
        T (float): The temperature at which the magnetisation is calculated
        steps (int): The number of steps in the algorithm

    Returns:
        float: absolute value of the magnetisation at temeprature T
    """

    N = len(state)
    m = np.zeros(steps) #magnetisation at each step
    state = state.copy()
    for step in range(steps):
        i = np.random.randint(0, N)
        j = np.random.randint(0, N)
        s =  state[i, j]
        nn = state[(i+1)%N,j] + state[i,(j+1)%N] + state[(i-1)%N,j] + state[i,(j-1)%N] 
        #nearest neighbour energy
        dE = 2*s*nn
        beta = 1/T
        if dE <= 0:
            s *= -1 #flip the spin
        
        elif np.random.random() < np.exp(-dE*beta): 
            s *= -1 #flip the spin with prob=np.exp(-dE*beta)
        
        state[i, j] = s
        
        m[step] = np.sum(state)/N**2
        
    return np.abs(m)




@njit(nogil=True)
def mcmc_state(steps, state, T):
    """A function that stores the states for each step
    of the Metropolis algorithm

    Args:
        steps (int): The number of steps used for the algorithm
        state (_list): A list that specifies the inital configuration of spins
        T (float): Temperature at which the thermalisation process is performed

    Returns:
        list: A list containing all the configurations indexed by the step
    """

    N = len(state)
    state_c = state.copy()
    n_state = [state_c]
    for step in range(steps):
        i = np.random.randint(0, N)
        j = np.random.randint(0, N)
        s =  state_c[i, j]
        nn = state_c[(i+1)%N,j] + state_c[i,(j+1)%N] + state_c[(i-1)%N,j] + state_c[i,(j-1)%N] 
        #nearest neighbour energy
        dE = 2*s*nn
        beta = 1/T
        if dE <= 0:
            s *= -1 #flip the spin
        
        elif np.random.random() < np.exp(-dE*beta): 
            s *= -1 #flip the spin with prob=np.exp(-dE*beta)
        
        state[i, j] = s
        n_state.append(state.copy())

    return n_state


    
#Toggle the functions

if __name__ == "__main__":

    # Magnetisation

    steps = 10**6 #MCMC steps
    l = 100 #side lenght of the square
    T = np.linspace(0.5, 4, 50) #Temperature range
    initial_state = np.ones([l,l])  

    m_avg = np.zeros(len(T))
    
    for i,j in enumerate(T):
        m = mcmc_m(initial_state, j, steps)
        m_avg[i] = np.mean(m[-10**5:])

    plot1 = plt.figure(1)

    Tc = 2/(np.log(1+np.sqrt(2)))
    plt.plot(T/Tc, m_avg, 'o--')
    plt.xlabel(r"Temperature (T/$T_C$)", fontsize=20)
    plt.ylabel("Magnetization ", fontsize=20)

    # Animation


    # state = mcmc_state(10**4, np.ones([60,60]), 10)
    # fig = plt.figure()
    # cmap = colors.ListedColormap(['orange', 'blue'])
    # im = plt.imshow(state[0],extent=[0,60,0,60], vmin = -1, vmax = 1,cmap = cmap)
    # plt.title('Ising model: thermalisation')
    # plt.colorbar(ticks=[-1, 1])

    # def updatefig(j):
    #     """Function used to update the animation"""
    #     im.set_array(state[j])
    #     return [im]

    # ani = animation.FuncAnimation(fig, updatefig,
    #                           interval=0.01, blit=True)    
    
    plt.show()
    
    
    
    
    
    
    
    
