from random import expovariate # Generate variates from exponential distribution
import numpy as np
import matplotlib.pyplot as plt

def gillespie_ABA(N, B0, beta, gamma, Tmax):

    A = [N-B0]  # We cannot predict how many elements there will be unfortunately
    B = [B0]
    T = [0]
    # Randomly allocate B0 individuals to have state B (state=1), A (state=0) otherwise
    state = np.random.permutation([0]*(N-B0)+[1]*B0)
    # Index of individuals in state B (state=1).
    B_contacts = np.where(state == 1)[0]
    # Set rates to be B0*beta/N (rate for individuals in state A) to all individuals (initialisation).
    rate_vector = B0*beta*np.ones((N, 1))/N
    # Update rate of B_contacts to be gamma (the rate for individuals in state B)
    rate_vector[B_contacts] = gamma

    time = 0
    while time <= Tmax+0.5:  # some (arbitrary) buffer after Tmax
        # Total rate (refer to Gillespie algorithm for details)
        rate = np.sum(rate_vector)
        cumrate = np.cumsum(rate_vector)  # Cumulated sum of rates
        if rate > 0.0000001:  # if rate is sufficiently large
            # Pick an exponentially distributed time. Beware of difference with exprnd in Matlab where it is 1/rate
            tstep = expovariate(rate)
            T.append(T[-1]+tstep)  # Time of next event
            # Find which individual will see its state change
            event = np.where(cumrate > np.random.rand()*rate)[0][0]
            if state[event] == 0:  # individual is in state A
                # this state A individual becomes state B so number of state A individuals is decreased
                A.append(A[-1]-1)
                # obviously, number of state B individuals is increased
                B.append(B[-1]+1)
                state[event] = 1  # Update state vector
                # Change rate of individual to B->A rate, namely gamma
                rate_vector[event] = gamma
                # List of state A individuals after change
                A_contacts = np.where(state == 0)[0]
                # Update rate of state A individuals to account for the extra state B individual
                rate_vector[A_contacts] += beta/N
            else:  # individual is in state B
                # this state B individual becomes state A so number of state B individuals is decreased
                B.append(B[-1]-1)
                # obviously, number of state A individuals is increased
                A.append(A[-1]+1)
                state[event] = 0  # Update state vector
                # List of state A individuals after changes
                A_contacts = np.where(state == 0)[0]
                # Update rate of state A individuals based on number of B individuals
                rate_vector[A_contacts] = beta*len(np.where(state == 1)[0])/N
        else:  # Nothing will happen from now on so we can accelerate the process
            time = T[-1]  # current time
            while time <= Tmax + 0.5:
                A.append(A[-1])  # Just keep things as they are
                B.append(B[-1])
                T.append(T[-1]+0.5)  # arbitrarily add 0.5 to clock
                time = T[-1]
        # Update time and proceed with loop
        time = T[-1]

    return T, A, B


def gillespie_ABA_stable(N , B0, beta, gamma, Tmax):

    A=[N-B0] # We cannot predict how many elements there will be unfortunately
    B=[B0]
    T=[0] 
    state = np.random.permutation([0]*(N-B0)+[1]*B0) # Randomly allocate B0 individuals to have state B (state=1), A (state=0) otherwise 
    
    B_contacts = np.where(state == 1)[0] # Index of individuals in state B (state=1).
    rate_vector = B0 * beta * np.ones((N,1)) / N # Set rates to be B0*beta/N (rate for individuals in state A) to all individuals (initialisation). 
    rate_vector[B_contacts] = gamma # Update rate of B_contacts to be gamma (the rate for individuals in state B)
    
    time = 0
    while time <= Tmax + 0.5: # some (arbitrary) buffer after Tmax
        
        rate = np.sum(rate_vector) # Total rate (refer to Gillespie algorithm for details)
        cumrate = np.cumsum(rate_vector) # Cumulated sum of rates
        
        if rate > 0.000001: # if rate is sufficiently large
            
            tstep = expovariate(rate) # Pick an exponentially distributed time. Beware of difference with exprnd in Matlab where it is 1/rate
            T.append(T[-1] + tstep) # Time of next event
          
            event = np.where(cumrate>np.random.rand()*rate)[0][0] # Find which individual will see its state change 
            
            if state[event] == 0: # individual is in state A 
                
                A.append(A[-1] - 1) # this state A individual becomes state B so number of state A individuals is decreased
                B.append(B[-1] + 1) # obviously, number of state B individuals is increased 
                state[event] = 1 # Update state vector
                rate_vector[event] = gamma # Change rate of individual to B->A rate, namely gamma
                A_contacts = np.where(state == 0)[0] # List of state A individuals after change
                rate_vector[A_contacts] += beta/N # Update rate of state A individuals to account for the extra state B individual
            
            else: # individual is in state B
               
                #implementation to improve average
                ###############################################
                #only iniate the change when [B] > 1
                if (B[-1] > 1 or beta <= gamma):
                    # this state B individual becomes state A so number of state B individuals is decreased
                    B.append(B[-1]-1)
                    # obviously, number of state A individuals is increased
                    A.append(A[-1]+1)
                    state[event] = 0  # Update state vector
                    # List of state A individuals after changes
                    A_contacts = np.where(state == 0)[0]
                    # Update rate of state A individuals based on number of B individuals
                    rate_vector[A_contacts] = beta*len(np.where(state == 1)[0])/N
                    
                #if there is only one left, repeat the previous value
                else:
                    B.append(B[-1])
                    A.append(A[-1])
                ################################################    
                    
                
        else: # Nothing will happen from now on so we can accelerate the process
            
            time = T[-1] # current time
            
            while time <= Tmax + 0.5:
                
                A.append(A[-1]) # Just keep things as they are
                B.append(B[-1])
                T.append(T[-1]+0.5) # arbitrarily add 0.5 to clock
                time = T[-1]
                
        # Update time and proceed with loop 
        time = T[-1]         

    return T,A,B 


#collects and plots the realisations
def simulation(b0, N, beta, gamma, tmax, num_reals, gillfunc, flag):
    
    #realisations are stored
    B_gils = []
    
    #linear time array initialised
    times = np.arange(0, tmax, 0.01)
    
    #loop through the amount of realisations given
    i = 0
    while i < num_reals:

        T_gil, A_gil, B_gil = gillfunc(N, b0, beta, gamma, tmax)
        B_nearest = nearest_gill(B_gil, T_gil, times)     
        B_gils.append([B_nearest])
        
        i += 1
        
    r0 = beta/gamma
    
    #setting up plots
    fig, axs = plt.subplots(1, 1, figsize=(12,7))
    fig.suptitle(f"B Trajectory over time for N={N}, B0={b0}, R0={r0}\n\n", fontweight ="bold") 
    axs.set_ylabel('[B]')
    axs.set_xlabel('Time')
    axs.set_ylim(0, N)
    
    #print(B_gils[0][-1])      
    B_avg, std = cal_stats(B_gils)
    
    #plotting realisations
    axs.scatter(times, B_gils[0], color='r', linewidth=1, s=1, label = "B")
    for i in range(1, num_reals):
        axs.scatter(times, B_gils[i], color='r', linewidth=1, s=1)
     
    #plotting average and error
    if flag == 2 or flag == 3:
        axs.scatter(times, B_avg[0], color='g', linewidth=2, s=0.5, label = "Average")
        axs.errorbar(times, std[0], color='black', linewidth=2, label = "Standard Deviation")
     
    #plotting analytical
    if flag == 3:
        B_analytical = mean_field(b0, times, N, beta, gamma)
        axs.scatter(times, B_analytical, color='b', linewidth=3, s=0.5, label = "Analytical")
    
    axs.legend(loc='best',markerscale=5)
    axs.grid()
    
    plt.show()
    
    
# This functions will be used to find the nearest time to our arange, in the Gillespie time array.
def nearest_gill(b_gil, t_gil, times):

    B = []

    for t in times:

        # calculate the difference array
        difference_array = np.absolute(t_gil - t)

        # find the index of minimum element from the array
        index = difference_array.argmin()

        # Then append the values of B corrresponding to that index into a new array
        B.append(b_gil[index])

    return B


# Mean field equation analytical solution, adapted to return A aswell
def mean_field(B0, t, N, beta, gamma):
    
    #reduce messiness, taken out the common term
    lam = N * (gamma - beta)
    B = B0 * lam / ((lam + B0 * beta) * np.exp(lam * t/N) - B0*beta)
       
    return B

#calculates stats
def cal_stats(b_gil):
   
    B_avg = np.average(b_gil, axis = 0)
    std_err = np.std(b_gil, axis = 0)
    
    return B_avg, std_err

#run on command line with python3 simulation_code.py
if __name__ == "__main__":
    
    
    #edit params to run any b0, r0, N, max time, number of realisations and the flag will determine whether you want 
    #to print average and error bar with or without the analytical plot.
    #simulation(N = 1000, b0 = 500, beta = 2.5, gamma = 1, tmax = 30.1, num_reals = 30, gillfunc = gillespie_ABA, flag = 2)
    
    #testing the considerations of beta = 0.51
    #simulation(N = 1000, b0 = 500, beta = 0.51, gamma = 0.5, tmax = 30.1, num_reals = 100, gillfunc = gillespie_ABA, flag = 3)
    
    
    #improving the average to the mean field
    #simulation(N = 1000, b0 = 1, beta = 0.95, gamma = 0.5, tmax = 30.1, num_reals = 30, gillfunc = gillespie_ABA_stable, flag = 3)
