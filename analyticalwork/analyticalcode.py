import matplotlib.pyplot as plt
import matplotlib
import numpy as np
#import pandas as pd
from scipy.integrate import odeint

class Differential_System:
    
    # This returns dA/dt and dB/dt given current position Y and parameters
    def dydt_AB(self, Y, t, N, b , g):

        A = Y[0];
        B = Y[1];

        dydt = [(-b*B*A)/N + g*B, (b*B*A)/N - g*B]

        return dydt

    #returns mean field equation
    def dydt_B(self, Y, t, N, b, g):

        b0 = Y

        dydt = b0*(b - g - (b * b0/N))

        return dydt

#Q.2 phase portrait

    def plot_phase_portrait_for_system(self, beta, gamma): 
        
        # Plot phase portrait with few sample trajectories for 
        # function f and parameter value para
        x,y = np.linspace(0,1000,100),np.linspace(0,1000,100)
        A,B = np.meshgrid(x,y) # Make a meshgrid from ranges of x and y
        
        #start points
        start = [[200,800],[400,0], [800,50], [0, 1000]]
        [U,V] = self.dydt_AB([A,B],0,1000, beta, gamma)
        
        fig0, ax0 = plt.subplots()
        
        ax0.set_ylabel('[B]')
        ax0.set_xlabel('[A]')
        
       
        strm = ax0.streamplot(A,B,U,V,color=(.75,.90,.93))
        strms = ax0.streamplot(A,B,U,V, start_points=start, color="crimson", linewidth=2)
        plt.show()
        
        
# Q.2 Equilibria

#equilibria in terms of total population for either state A or B
    def equilibria(self, N, r0):

        B_eq = N*(1-1/r0)
        A_eq = N/r0

        return A_eq, B_eq

#Q.3 
    def bifurcation(self, N, r0s):
        
        A_eq = []
        B_eq = [] 
        
        #looping through r0s to determine their equilibria.
        for r in r0s:
            a, b = self.equilibria(N, r)
            A_eq.append(a)
            B_eq.append(b)
            
        #A*
        plt.plot(r0s, A_eq)
        #B*
        plt.plot(r0s, B_eq)

        plt.legend(["A*","B*"])
        plt.ylim([-1000,1500])
        plt.xlabel("R0")
        plt.ylabel("N")

        #limits taken from current total population size.
        plt.plot(r0s,np.full(len(r0s), N), linewidth=1, color = 'red')
        plt.plot(r0s,np.full(len(r0s), 0), linewidth=1, color = 'red')
        plt.show()        
     
 #Q.5 Plot solutions of [B](t) for various values of R0 and B0

    #analytical solution for [B](t)
    def analytical_bt(self, b0, t, N, beta, gamma):
        
        #common expressions within B(t)
        lam = N*(gamma-beta)
    
        return b0 * lam / ((lam + b0 * beta) * np.exp(lam * t/N ) - b0 * beta)
    
    
    def plot_analytical(self, b0, t, N, beta, gamma):
        
        lam = N*(gamma-beta)
    
        bt = b0 * lam / ((lam + b0 * beta) * np.exp(lam/N *t )) - b0 * beta
        
        plt.plot(t, bt)
        plt.show()
    
    #producing visual comparison of Euler and Analytical
    def bt_rep(self, b0s, N, tmax, gamma, r0s):
        
        t = np.linspace(0, tmax, 1000)
        betas = r0s * gamma
        
        for b0 in b0s:
        
            fig, axs = plt.subplots(1, 2, figsize=(12,7))
            fig.suptitle(f"B Trajectory over time for N={N}, B0={b0}\n\n", fontweight ="bold") 
            
            #setting plot titles and axis
            axs[0].set_title('Analytical Solution')
            axs[1].set_title('Euler Solution - Mean Field')

            axs[0].set_ylabel('[B](t)')
            axs[0].set_xlabel('Time')

            #collecting different colours
            cmap = matplotlib.cm.get_cmap()
            colors=np.linspace(0,1,num=len(r0s)) 

            #looping through betas to produce different converges of B0
            for i in range(len(r0s)):

                #As shown before, unrealistic values of B when r0 < 1, only plotting for realistic equilibria
                if r0s[i] > 1:   
                    a_eq, b_eq = self.equilibria(N, r0s[i])                                                       
                else:
                    b_eq = 0

               #plotting the B* converge points given b_eq is B* = N(1-1/r0) with a marker x
                axs[0].plot(tmax, b_eq, marker = 'x', color=cmap(colors[i]))
                axs[1].plot(tmax, b_eq, marker = 'x', color=cmap(colors[i]))

                #Analytical solution
                bt = self.analytical_bt(b0, t, N, betas[i], gamma)
                axs[0].plot(t, bt, color=cmap(colors[i]), label = f"R0 = {r0s[i]:.2f} (beta={betas[i]:.2f}, gamma = {gamma},x=B*={ b_eq:.2f})")

                #Euler solution
                h = 0.001
                t1 = np.arange(0,tmax+h,h)
                e_bt = np.zeros((len(t1),1)) # pre-allocate 
                e_bt[0]=b0

                #perfomring foward euler
                for j in np.arange(len(t1)-1):
                    e_bt[j+1] = e_bt[j]+h*self.dydt_B(e_bt[j], t1[j], N, betas[i], gamma)

                axs[1].plot(t1, e_bt, color=cmap(colors[i]))

            legend = fig.legend(bbox_to_anchor = (0.4,-0.01))
            plt.show()


            
#Q.5, varying gammas with same r0
    def diff_gamma(self, b0, N, t, gammas, r0):
        
        fig, axs = plt.subplots(1, 2, figsize=(12,7))

        fig.suptitle("""B Trajectory over time\n\n""", fontweight ="bold") 
        
        #setting plot titles and axis
        axs[0].set_title('Analytical Solution')
        axs[1].set_title('Euler Solution - Mean Field')
        
        axs[0].set_ylabel('[B](t)')
        axs[0].set_xlabel('Time')
        
        #collecting different colours
        cmap = matplotlib.cm.get_cmap()
        colors=np.linspace(0,1,num=len(gammas)) 
        
        #loop through the gammas
        for i in range(len(gammas)):
            
            beta = r0 * gammas[i]
            
            #Analytical solution
            bt = self.analytical_bt(b0, t, N, beta, gammas[i])
            axs[0].plot(t, bt, color=cmap(colors[i]), label = f"R0 = {r0:.2f} (beta={beta:.2f}, gamma = {gammas[i]:.2f})")
           
            #Euler solution
            tmax = 200  
            h = 0.01
            t1 = np.arange(0,tmax+h,h)
            e_bt = np.zeros((len(t1),1)) # pre-allocate x, x0=0;
            e_bt[0]=b0
            
            for j in np.arange(len(t1)-1):
                e_bt[j+1] = e_bt[j]+h*self.dydt_B(e_bt[j], t1[j], N, beta, gammas[i])
                
            axs[1].plot(t1, e_bt, color=cmap(colors[i]))

            legend = fig.legend(bbox_to_anchor = (0.4,-0.01))
        plt.show()

#run via the command line with python3 analyticalcode.py
if __name__ == "__main__":
    system = Differential_System()
    
    #phase portait in terms of r0

    #greater than 1
    #system.plot_phase_portrait_for_system(beta = 1.1, gamma = 1)

    #less than 1
    #system.plot_phase_portrait_for_system(beta = 1, gamma = 2)

    #equal to 1
    #system.plot_phase_portrait_for_system(beta = 1, gamma = 1)

    #q3 bifurcation plot
    #system.bifurcation(N = 1000, r0s = np.linspace(0.1,5,100))

    #q.5
    #setting constants for comparing [B](t) for various values of B0s and r0s
    b0s = [100, 250, 400, 600, 800, 950]
    N = 1000
    gamma = 0.5
    r0s = np.linspace(0.1,5.0,10)
    betas = r0s * gamma
    tmax = 200

    #produce visual comparison with analytical and euler
    #system.bt_rep(b0s, N, tmax, gamma, r0s)

    #varying gamma
    system.diff_gamma(b0 = 100, N = 1000, t = np.linspace(0,200,100), gammas = np.linspace(0.1,5,10), r0 = 2)

