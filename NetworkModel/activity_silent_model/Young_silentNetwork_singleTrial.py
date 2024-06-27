from brian2 import *
import sys
import numpy as np
import time
import socket
from scipy.io import savemat
import multiprocessing as mp
import random


start_time = time.time()  # returns the number of seconds passed since epoch
defaultclock.dt = 0.1*ms

n_sims = 1                # number of simulations to run


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
    w2      = 100*ms                  # window size
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
    nstims=8   # number of possible equidistant locations for the stimulus  # by sara  
    
    stim_on=2*second
    stim_off=3*second
    runtime=7*second
    
    stimE=0.27*mV   #stimulus amplitude
    epsE=0.17       #stimulus width
    stimI=0*mV
    epsI=0
    
    N=20000         # total number of neurons
    K=500           # total number of inputs
    
    tauE=20*ms
    tauI=10*ms
    taua=3*ms       # AMPA synapse decay
    taun=50*ms      # NMDA synapse decay
    taug=4*ms       # GABA synapse decay
    
    Vt=20*mV        # spike threshold
    Vr=-3.33*mV     # reset value
    refE=0*ms       # refractory period
    refI=0*ms       # refractory period
    
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
    sigmaEE=30  # E-to-E footprint in degrees
    sigmaEI=35  # E-to-E footprint in degrees
    sigmaIE=30  # E-to-E footprint in degrees
    sigmaII=30  # E-to-E footprint in degrees

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

    '''
    
    reset = '''
    V = Vr
    '''
   
    # generate the two populations
    networkE = NeuronGroup(NE,model=eqs,threshold='V > Vt',reset=reset, refractory=refE, method='exact', name='networkE')
    networkI = NeuronGroup(NI,model=eqs,threshold='V > Vt',reset='V = Vr', refractory=refI, method='exact', name='networkI')
    
    # create synapses 
    C1 = Synapses(networkE, networkE,
                  model=''' w    : volt
                            dx/dt=(1-x)/taud : 1       
                            du/dt=(U-u)/tauf : 1  ''',
                  on_pre='''gea += w*u*x
                            x *= 1-u
                            u += U*(1-u) ''', name='C1')   #(event-driven)
    C2 = Synapses(networkE, networkE,
                  model=''' w    : volt
                            dx/dt=(1-x)/taud : 1 
                            du/dt=(U-u)/tauf : 1  ''',
                  on_pre='''gen += w*u*x
                            x *= 1-u
                            u += U*(1-u) ''', name='C2')                     
    C3 = Synapses(networkE, networkI, 
                  model=''' w    : volt ''',                      
                  on_pre='''gea += w ''', name='C3')    
    C4 = Synapses(networkE, networkI,
                  model=''' w    : volt ''',   
                  on_pre='''gen += w ''', name='C4')                           
    C5 = Synapses(networkI, networkE, 'w: volt', on_pre='gi += w',name='C5')
    C6 = Synapses(networkI, networkI, 'w: volt', on_pre='gi += w',name='C6')

    if make_network:      
        seed(3896)
        
        # connect synapses
        fE = float(KE)/float(NE)/np.sqrt(2*np.pi)
        fI = float(KI)/float(NI)/np.sqrt(2*np.pi)
        
        C1.connect(p = 'fE/sigEE  * exp(-(i/NE-j/NE)**2 / (2*sigEE**2)) * (int( (i/NE-j/NE)*sign(i/NE-j/NE) <= 0.5)) + fE/sigEE * exp(-(abs(i/NE-j/NE)-1)**2 / (2*sigEE**2)) * (int( (i/NE-j/NE)*sign(i/NE-j/NE) > 0.5)) ')
        C1.w = gEEA
        C1.u = U
        C1.x = 1
            
        C2.connect(p = 'fE/sigEE  * exp(-(i/NE-j/NE)**2 / (2*sigEE**2)) * (int( (i/NE-j/NE)*sign(i/NE-j/NE) <= 0.5)) + fE/sigEE * exp(-(abs(i/NE-j/NE)-1)**2 / (2*sigEE**2)) * (int( (i/NE-j/NE)*sign(i/NE-j/NE) > 0.5)) ')
        C2.w = gEEN
        C2.u = U
        C2.x = 1
            
        C3.connect(p = 'fE/sigEI * exp(-(i/NE-j/NI)**2 / (2*sigEI**2)) * (int( (i/NE-j/NI)*sign(i/NE-j/NI) <= 0.5)) + fE/sigEI * exp(-(abs(i/NE-j/NI)-1)**2 / (2*sigEI**2)) * (int( (i/NE-j/NI)*sign(i/NE-j/NI) > 0.5)) ')
        C3.w = gEIA
        
        C4.connect(p = 'fE/sigEI * exp(-(i/NE-j/NI)**2 / (2*sigEI**2)) * (int( (i/NE-j/NI)*sign(i/NE-j/NI) <= 0.5)) + fE/sigEI * exp(-(abs(i/NE-j/NI)-1)**2 / (2*sigEI**2)) * (int( (i/NE-j/NI)*sign(i/NE-j/NI) > 0.5)) ')
        C4.w = gEIN
        
        C5.connect(p = 'fI/sigIE * exp(-(i/NI-j/NE)**2 / (2*sigIE**2)) * (int( (i/NI-j/NE)*sign(i/NI-j/NE) <= 0.5)) + fI/sigIE * exp(-(abs(i/NI-j/NE)-1)**2 / (2*sigIE**2)) * (int( (i/NI-j/NE)*sign(i/NI-j/NE) > 0.5)) ')
        C5.w = gIE
            
        C6.connect(p = 'fI/sigII * exp(-(i/NI-j/NI)**2 / (2*sigII**2)) * (int( (i/NI-j/NI)*sign(i/NI-j/NI) <= 0.5)) + fI/sigII * exp(-(abs(i/NI-j/NI)-1)**2 / (2*sigII**2)) * (int( (i/NI-j/NI)*sign(i/NI-j/NI) > 0.5)) ')
        C6.w = gII

        # C1.delay = 'rand()*100*ms'
        # C2.delay = 'rand()*100*ms'
        # C3.delay = 'rand()*100*ms'
        # C4.delay = 'rand()*100*ms'
        
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
    networkE.Ix = 1.60*sqrt(KE)*mV
    networkI.Ix = 1.85*0.83*sqrt(KE)*mV
    networkE.V = Vr + rand(NE)*(Vt - Vr)      # random initial conditions for V in each trial (E neurons)
    networkI.V = Vr + rand(NI)*(Vt - Vr)      # random initial conditions for V in each trial (I neurons)
    networkE.Iext = 0*mV
    networkI.Iext = 0*mV

    # monitor voltage, currents and spikes
    MEa = StateMonitor(C1, ('u','x'), record=np.arange(0,NE*KE,500),dt=10*ms)
    MEn = StateMonitor(C2, ('u','x'), record=np.arange(0,NE*KE,500),dt=10*ms)
    spikesE = SpikeMonitor(networkE)
    spE_i = spikesE.i
    spE_t = spikesE.t 
    spE_count = spikesE.count  
    spikesI = SpikeMonitor(networkI)
    spI_i = spikesI.i
    spI_t = spikesI.t
    spI_count = spikesI.count
       
    # run the simulation during the pre-cue period
    run(stim_on,report='text')


    ### CUE PERIOD
    # define the stimulus location
    Stim_loc = 4      #stimulus at 180º (0 <= stim_loc <= 7)
    stimat = Stim_loc/nstims * NE
    #stimat = randint(0,nstims)/nstims * NE   #stimulus location random within nstims possible equidistant locations
 
    # define the stimulus input
    posE = arange(NE)  
    inpE = stimE * exp(-0.5 * (posE/float(NE)-0.5)**2 / (epsE**2))
    networkE.Iext = np.roll(inpE,int(stimat-NE/2))
    networkI.Iext = stimI
    
    # run the simulation during the cue period
    run(stim_off-stim_on,report='text')
    
     
    ### DELAY PERIOD
    ti1 = 4.1*second 
    tf1 = 4.4*second   
    ti2 = 5.4*second   
    tf2 = 5.7*second
    amp = 11*mV
    
    networkE.Iext = 0*mV
    networkI.Iext = 0*mV
    # run the simulation during the delay period
    run(ti1-stim_off,report='text')  
    
    networkE.Iext = amp
    networkI.Iext = 0*mV   
    # run the simulation during the delay period
    run(tf1-ti1,report='text')
    
    networkE.Iext = 0*mV
    networkI.Iext = 0*mV  
    # run the simulation during the delay period
    run(ti2-tf1,report='text')
    
    networkE.Iext = amp + 2*mV
    networkI.Iext = 0*mV  
    # run the simulation during the delay period
    run(tf2-ti2,report='text')

    networkE.Iext = 0*mV
    networkI.Iext = 0*mV   
    # run the simulation during the delay period
    run(runtime-tf2,report='text')
    
    
    
    ## DO CALCULATIONS
    # get the decoded angle/modulus of the population vector through the whole simulation time
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

    # average firing rate for each E neuron during the cue period
    fr_cue = np.zeros(NE)
    idxEc = ((spE_t > stim_on) & (spE_t <= stim_off))
    spE_iic = spE_i[idxEc]
    for mec in range(NE):
        fr_cue[mec] = sum(spE_iic == mec)          # number of spikes for each E neuron during the cue period
    fr_cue = fr_cue / (stim_off-stim_on)   
     
    # average firing rate for each E neuron during the first Iext injection period
    fr_DelayBump1 = np.zeros(NE)
    idxE1 = ((spE_t > ti1) & (spE_t <= tf1))
    spE_ii1 = spE_i[idxE1]
    for me1 in range(NE):
        fr_DelayBump1[me1] = sum(spE_ii1 == me1)    # number of spikes for each E neuron during the first Iext injection period
    fr_DelayBump1 = fr_DelayBump1 / (tf1-ti1)    
    
    # average firing rate for each E neuron during the second Iext injection period
    fr_DelayBump2 = np.zeros(NE)
    idxE2 = ((spE_t > ti2) & (spE_t <= tf2))
    spE_ii2 = spE_i[idxE2]
    for me2 in range(NE):
        fr_DelayBump2[me2] = sum(spE_ii2 == me2)     # number of spikes for each E neuron during the second Iext injection period
    fr_DelayBump2 = fr_DelayBump2 / (tf2-ti2)   
    
    
    
    ## SAVE DATA
    data1 = {'N_stim': nstims, 'stim': stimat, 'dec_angle': np.array(popdec), 'dec_modulus': np.array(popmod), 'sliding_window_size': window_size, 'window_time_step': window_t_step, 'number_windows': nwins}    
    savemat("decoding_" + str(i_sims) +  "_" + str(socket.gethostname()) + ".mat", data1) 
    
    data2 = {'spikes_E_cells': np.array(spE_i), 'spike_times_E_cells': np.array(spE_t), 'spike_counts_E_cells': np.array(spE_count), 'spikes_I_cells': np.array(spI_i), 'spike_times_I_cells': np.array(spI_t), 'spike_counts_I_cells': np.array(spI_count)}    
    savemat("spikes_" + str(i_sims) +  "_" + str(socket.gethostname()) + ".mat", data2)       
  
    data3 = {'fr_E' : np.array(fr_delay_E), 'fr_E_DelayBump1' : np.array(fr_DelayBump1), 'fr_E_DelayBump2' : np.array(fr_DelayBump2),  'fr_cue' : np.array(fr_cue)}
    savemat("rates_" + str(i_sims) +  "_" + str(socket.gethostname()) + ".mat", data3)  
    
    data4 = {'u_a': MEa.u, 'x_a': MEa.x, 'u_n': MEn.u, 'x_n': MEn.x}
    savemat("system_variables_" + str(i_sims) +  "_" + str(socket.gethostname()) + ".mat", data4)
    
    #data5 = {'delays_C1': np.array(C1.delay), 'delays_C2': np.array(C2.delay), 'delays_C3': np.array(C3.delay), 'delays_C4': np.array(C4.delay)}
    #savemat("delays_" + str(i_sims) +  "_" + str(socket.gethostname()) + ".mat", data5)
    
    



#####################################################################################################
#                                    RUN SIMULATIONS                                                #
#####################################################################################################           
run_sims(n_sims)

print('all sims finished')
print(time.time() - start_time) 





































