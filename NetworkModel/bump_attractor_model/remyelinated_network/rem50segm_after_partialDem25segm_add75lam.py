# ---------------- CREATED BY S. IBAÑEZ; JULY 2023.
# CODE TO GENERATE DECODING AND SPIKING DATA FOR A
# SINGLE TRIAL OF A REMYELINATED NETWORK: remyelination of 50%
# of the previously partially demyelination 25% of the segments,
# when adding 75% of the lamellae back. ----------------

from brian2 import *
import sys
import numpy as np
import time
import socket
from scipy.io import savemat
import multiprocessing as mp
import random



start_time = time.time()   # returns the number of seconds passed since epoch
defaultclock.dt = 0.1*ms


n_sims = 1                 # number of simulations to run



# define function to decode angles from spike counts
def decode(firing_rate, N_e): 
    angles = np.arange(0,N_e)*2*np.pi/N_e
    R = np.sum(np.dot(firing_rate,np.exp(1j*angles)))/np.sum(firing_rate)  
    angle = np.angle(R)    # np.angle returns the angle of the complex argument (in radiand by default, betweeen (-pi, pi])
    modulus = np.abs(R)    
    if angle < 0:
        angle += 2*np.pi 
    return angle, modulus 

# define function to get bump center diffusion from full rastergram (i,t)
def readout(i, t, sim_time, N_e):
    w1      = 100*ms                  # time step of the sliding window
    w2      = 250*ms                  # window size
    n_wins  = int((sim_time-w2)/w1)

    decs = []
    mods = []
    for ti in range(int(n_wins)):
        fr  = np.zeros(N_e)
        idx = ((t>ti*w1-w2/2) & (t<ti*w1+w2/2))
        ii  = i[idx]
        for n in range(N_e):
            fr[n] = sum(ii == n)            # number of spikes for each neuron in the time interval defined by idx
        dec, mod = decode(fr, N_e)         
        decs.append(dec)
        mods.append(mod)                   
    return decs, n_wins, mods, w1, w2   


# simulation
def run_sims(i_sims):  
    
    # choose whether to store the state variable values (make_network=True) or restore them (faster)
    make_network = True
    
    # simulation parameters
    nstims=8   # number of possible equidistant locations for the stimulus 
    
    stim_on=2*second                                      
    stim_off=3*second                                    
    runtime=7*second                                      
    
    stimE=0.24*mV           #stimulus amplitude           
    epsE=0.17               #stimulus width               
    stimI=0*mV                                            
    epsI=0                                               
    
    N=20000  # total number of neurons                    
    K=500    # total number of inputs                  
    
    tauE=20*ms                                 
    tauI=10*ms                                
    taua=3*ms  # AMPA synapse decay            
    taun=50*ms # NMDA synapse decay             
    taug=4*ms  # GABA synapse decay            
    
    Vt=20*mV          # spike threshold         
    Vr=-3.33*mV       # reset value            
    refE=0*ms         # refractory period      
    refI=0*ms         # refractory period     
    
    # parameters for short-term plasticity
    U=0.03                                     
    taud=200*ms                                
    tauf=450*ms                                

    # connectivity strengths    
    gEEA=533.3*mV*ms                             
    gEEN=490.64*mV*ms                           
    gEIA=67.2*mV*ms                              
    gEIN=7.4*mV*ms                              
    gIE=-138.6*mV*ms                             
    gII=-90.6*mV*ms                              
    sigmaEE=30    # E-to-E footprint in degrees    
    sigmaEI=35    # E-to-E footprint in degrees     
    sigmaIE=30    # E-to-E footprint in degrees    
    sigmaII=30    # E-to-E footprint in degrees     

    # these are intermediate calculations needed for the equations below
    NE=int(ceil(0.8*N))   # number of excitatory neurons
    NI=int(floor(0.2*N))  # number of inhibitory neurons

    KE=int(ceil(0.8*K))   # number of excitatory inputs
    KI=int(floor(0.2*K))  # number of inhibitory inputs

    sigEE=sigmaEE/360.0
    sigEI=sigmaEI/360.0
    sigIE=sigmaIE/360.0
    sigII=sigmaII/360.0 

    gEEA=gEEA/sqrt(KE)/taua  
    gEEN=gEEN/sqrt(KE)/taun   
    gEIA=gEIA/sqrt(KE)/taua   
    gEIN=gEIN/sqrt(KE)/taun   
    gIE=gIE/sqrt(KI)/taug     
    gII=gII/sqrt(KI)/taug    
    
    stimE=stimE*sqrt(KE)     
    stimI=stimI*sqrt(KE)   

    
    # equations for each neuron
    eqs = '''
    Irec = gea+gen+gi + Ix     : volt
    dV/dt = (Irec-V+Iext)/tau  : volt (unless refractory)
    dgea/dt = -gea/taua        : volt
    dgen/dt = -gen/taun        : volt
    dgi/dt = -gi/taug          : volt
    tau                        : second
    Ix                         : volt
    Iext                       : volt
    transmit_AP                : 1
    probability                : 1 
    ''' 
    
    # reset functions
    reset = '''
    V = Vr
    transmit_AP = (rand() < probability)
    '''    
    
    # switch the order of "synapses" and "reset" in the network schedule 
    Network.schedule = ['start', 'groups', 'thresholds',  'resets', 'synapses', 'end']  # Default schedule: ['start', 'groups', 'thresholds', 'synapses', 'resets', 'end'] 
    
    # generate the two populations
    networkE = NeuronGroup(NE,model=eqs,threshold='V > Vt',reset=reset, refractory=refE, method='exact', name='networkE')   
    networkI = NeuronGroup(NI,model=eqs,threshold='V > Vt',reset='V = Vr', refractory=refI, method='exact', name='networkI') 
    
    # create synapses 
    C1 = Synapses(networkE, networkE,
                  model=''' w    : volt
                            dx/dt=(1-x)/taud : 1 (event-driven)
                            du/dt=(U-u)/tauf : 1 (event-driven) ''',
                  on_pre='''gea += w*u*x*transmit_AP_pre
                            x *= 1-u
                            u += U*(1-u) ''', name='C1') 
    C2 = Synapses(networkE, networkE,
                  model=''' w    : volt
                            dx/dt=(1-x)/taud : 1 (event-driven)
                            du/dt=(U-u)/tauf : 1 (event-driven) ''',
                  on_pre='''gen += w*u*x*transmit_AP_pre
                            x *= 1-u
                            u += U*(1-u) ''', name='C2')                     
    C3 = Synapses(networkE, networkI, 
                  model=''' w    : volt ''',                      
                  on_pre='''gea += w*transmit_AP_pre ''', name='C3')    
    C4 = Synapses(networkE, networkI,
                  model=''' w    : volt ''',   
                  on_pre='''gen += w*transmit_AP_pre ''', name='C4')    
    C5 = Synapses(networkI, networkE, 'w: volt', on_pre='gi += w',name='C5')
    C6 = Synapses(networkI, networkI, 'w: volt', on_pre='gi += w',name='C6')

    if make_network:      
        seed(4670)   
        
        # connect synapses
        fE = float(KE)/float(NE)/np.sqrt(2*np.pi)
        fI = float(KI)/float(NI)/np.sqrt(2*np.pi)
        
        C1.connect(p = 'fE/sigEE  * exp(-(i/NE-j/NE)**2 / (2*sigEE**2)) * (int( (i/NE-j/NE)*sign(i/NE-j/NE) <= 0.5)) + fE/sigEE * exp(-(abs(i/NE-j/NE)-1)**2 / (2*sigEE**2)) * (int( (i/NE-j/NE)*sign(i/NE-j/NE) > 0.5)) ')
        C1.w = gEEA
        C1.u = U
        C1.x = 1
        #C1.delay = 'rand()*40*ms'
            
        C2.connect(p = 'fE/sigEE  * exp(-(i/NE-j/NE)**2 / (2*sigEE**2)) * (int( (i/NE-j/NE)*sign(i/NE-j/NE) <= 0.5)) + fE/sigEE * exp(-(abs(i/NE-j/NE)-1)**2 / (2*sigEE**2)) * (int( (i/NE-j/NE)*sign(i/NE-j/NE) > 0.5)) ')
        C2.w = gEEN
        C2.u = U
        C2.x = 1
        #C2.delay = 'rand()*40*ms'
            
        C3.connect(p = 'fE/sigEI * exp(-(i/NE-j/NI)**2 / (2*sigEI**2)) * (int( (i/NE-j/NI)*sign(i/NE-j/NI) <= 0.5)) + fE/sigEI * exp(-(abs(i/NE-j/NI)-1)**2 / (2*sigEI**2)) * (int( (i/NE-j/NI)*sign(i/NE-j/NI) > 0.5)) ')
        C3.w = gEIA
        #C3.delay = 'rand()*40*ms'
        
        C4.connect(p = 'fE/sigEI * exp(-(i/NE-j/NI)**2 / (2*sigEI**2)) * (int( (i/NE-j/NI)*sign(i/NE-j/NI) <= 0.5)) + fE/sigEI * exp(-(abs(i/NE-j/NI)-1)**2 / (2*sigEI**2)) * (int( (i/NE-j/NI)*sign(i/NE-j/NI) > 0.5)) ')
        C4.w = gEIN
        #C4.delay = 'rand()*40*ms'
        
        C5.connect(p = 'fI/sigIE * exp(-(i/NI-j/NE)**2 / (2*sigIE**2)) * (int( (i/NI-j/NE)*sign(i/NI-j/NE) <= 0.5)) + fI/sigIE * exp(-(abs(i/NI-j/NE)-1)**2 / (2*sigIE**2)) * (int( (i/NI-j/NE)*sign(i/NI-j/NE) > 0.5)) ')
        C5.w = gIE
            
        C6.connect(p = 'fI/sigII * exp(-(i/NI-j/NI)**2 / (2*sigII**2)) * (int( (i/NI-j/NI)*sign(i/NI-j/NI) <= 0.5)) + fI/sigII * exp(-(abs(i/NI-j/NI)-1)**2 / (2*sigII**2)) * (int( (i/NI-j/NI)*sign(i/NI-j/NI) > 0.5)) ')
        C6.w = gII

        seed()
        #store(filename = 'my_network')
        
    else:      
        restore(filename = 'my_network',restore_random_state=False)
        C1.connect(False)
        C2.connect(False)
        C3.connect(False)
        C4.connect(False)
        C5.connect(False)
        C6.connect(False)
    
        
    # initialize parameters for the two populations
    networkE.tau = tauE                       
    networkI.tau = tauI                      
    networkE.Ix = 1.66*sqrt(KE)*mV          
    networkI.Ix = 1.85*0.83*sqrt(KE)*mV       
    networkE.V = Vr + rand(NE)*(Vt - Vr)      # random initial conditions for V in each trial (E neurons)
    networkI.V = Vr + rand(NI)*(Vt - Vr)      # random initial conditions for V in each trial (I neurons)
    networkE.Iext = 0*mV                     
    networkI.Iext = 0*mV                    
    networkE.transmit_AP = 1
    #networkE.probability = 1
    
    #--------- model action potential failure
    perc1 = 80;              # percentage of E neurons with some probability of AP failure 
    perc2 = 2;
    perc3 = 4;
    perc4 = 2;
    perc5 = 2;
    perc6 = 2;
    perc7 = 2;
    perc8 = 2;
    perc9 = 4;
    n1 = int(perc1*NE/100)   # number of E neurons with probability of AP failure
    n2 = int(perc2*NE/100)
    n3 = int(perc3*NE/100)
    n4 = int(perc4*NE/100)
    n5 = int(perc5*NE/100)
    n6 = int(perc6*NE/100)
    n7 = int(perc7*NE/100)
    n8 = int(perc8*NE/100)
    n9 = int(perc9*NE/100)
    p1 = 1                   # probability of AP transmision
    p2 = 0.93
    p3 = 0.90
    p4 = 0.84
    p5 = 0.49
    p6 = 0.47
    p7 = 0.13
    p8 = 0.02
    p9 = 0
    prob1 = ones(n1)*p1
    prob2 = ones(n2)*p2
    prob3 = ones(n3)*p3
    prob4 = ones(n4)*p4
    prob5 = ones(n5)*p5
    prob6 = ones(n6)*p6
    prob7 = ones(n7)*p7
    prob8 = ones(n8)*p8
    prob9 = ones(n9)*p9
    prob = np.append(prob1,prob2)
    prob = np.append(prob,prob3)
    prob = np.append(prob,prob4)
    prob = np.append(prob,prob5)
    prob = np.append(prob,prob6)
    prob = np.append(prob,prob7)
    prob = np.append(prob,prob8)
    prob = np.append(prob,prob9)
    
    random.shuffle(prob)
    probabilities = prob.tolist()
    networkE.probability = probabilities 
    #---------------------------------------------------------
    
    # monitor voltage, currents and spikes
    ME = StateMonitor(networkE, ('V','gea','gen','gi','Ix','Iext','transmit_AP'), record=np.arange(0,NE,800))
    spikesE = SpikeMonitor(networkE)
    spE_i = spikesE.i
    spE_t = spikesE.t 
    spE_count = spikesE.count  
    MI = StateMonitor(networkI, ('V','gea','gen','gi','Ix','Iext'), record=np.arange(0,NI,400))
    spikesI = SpikeMonitor(networkI)
    spI_i = spikesI.i
    spI_t = spikesI.t
    spI_count = spikesI.count
       
    # run the simulation during the pre-cue period
    run(stim_on,report='text')


    ### CUE PERIOD
    # define the stimulus location 
    stimat = randint(0,nstims)/nstims * NE   #stimulus location random within nstims possible equidistant locations
 
    # define the stimulus input
    posE = arange(NE)  
    inpE = stimE * exp(-0.5 * (posE/float(NE)-0.5)**2 / (epsE**2))   
    networkE.Iext = np.roll(inpE,int(stimat-NE/2))                  
    networkI.Iext = stimI                                          
    
    # run the simulation during the cue period
    run(stim_off-stim_on,report='text')
    
     
    ### DELAY PERIOD
    # remove the stimulus input
    networkE.Iext = 0*mV
    networkI.Iext = 0*mV
    
    # monitor spikes during the delay
    # spikesE_d = SpikeMonitor(networkE)
    # spE_count_d = spikesE_d.count  
    # spikesI_d = SpikeMonitor(networkI)
    # spI_count_d = spikesI_d.count
    
    # run the simulation during the delay period
    run(runtime-stim_off,report='text')

  
    ## DO CALCULATIONS
    # get the decoded angle/modulus through the whole simulation time
    popdec, nwins, popmod, window_t_step, window_size = readout(spE_i, spE_t, runtime, NE)  
    popdec = np.array(popdec)
    popmod = np.array(popmod)  
    
    # average firing rate for each E/I neuron during the delay period (option 1)
    # ratesE = spE_count_d/(runtime-stim_off)
    # ratesI = spI_count_d/(runtime-stim_off)
    
    # average firing rate for each E/I neuron during the delay period (option 2)
    fr_delay_E = np.zeros(NE)
    fr_delay_I = np.zeros(NI)
    idxE = ((spE_t > stim_off) & (spE_t <= runtime))
    idxI = ((spI_t > stim_off) & (spI_t <= runtime))
    spE_ii = spE_i[idxE]
    spI_ii = spI_i[idxI]
    for me in range(NE):
        fr_delay_E[me] = sum(spE_ii == me)         # number of spikes for each E neuron during the delay period
    fr_delay_E = fr_delay_E / (runtime-stim_off)    
    for mi in range(NI):
        fr_delay_I[mi] = sum(spI_ii == mi)         # number of spikes for each I neuron during the delay period
    fr_delay_I = fr_delay_I / (runtime-stim_off)

    
    ## SAVE DATA
    data1 = {'N_stim': nstims, 'stim': stimat, 'dec_angle': np.array(popdec), 'dec_modulus': np.array(popmod), 'sliding_window_size': window_size, 'window_time_step': window_t_step, 'number_windows': nwins}    
    savemat("decoding_" + str(i_sims) +  "_" + str(socket.gethostname()) + ".mat", data1) 
    
    data2 = {'spikes_E_cells': np.array(spE_i), 'spike_times_E_cells': np.array(spE_t), 'spike_counts_E_cells': np.array(spE_count), 'spikes_I_cells': np.array(spI_i), 'spike_times_I_cells': np.array(spI_t), 'spike_counts_I_cells': np.array(spI_count)}    
    savemat("spikes_" + str(i_sims) +  "_" + str(socket.gethostname()) + ".mat", data2)       
  
    data3 = {'fr_E' : np.array(fr_delay_E), 'fr_I' : np.array(fr_delay_I)}    
    savemat("rates_" + str(i_sims) +  "_" + str(socket.gethostname()) + ".mat", data3)  
    
    #data4 = {'V_E': ME.V, 'gea_E': ME.gea, 'gen_E': ME.gen, 'gi_E': ME.gi, 'Iext_E': ME.Iext, 'transmit_AP': ME.transmit_AP, 'V_I': MI.V, 'gea_I': MI.gea, 'gen_I': MI.gen, 'gi_I': MI.gi}    
    #savemat("system_variables_" + str(i_sims) +  "_" + str(socket.gethostname()) + ".mat", data4) 



#####################################################################################################
#                                    RUN SIMULATIONS                                                #
#####################################################################################################           
run_sims(n_sims)

print('all sims finished')
print(time.time() - start_time) 











































