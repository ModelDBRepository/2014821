```
Filename: readme.md
Description: Contains instructions for initiating simulation(s) through the GUI.
Edit History: Created by Nilapratim Sengupta in June 2023. Formatting edits by ModelDB, March 2024.
```


## Step 1: Compilation of NMODL (.mod) files from the terminal

```
> nrnivmodl
```

## Step 2: Launching the GUI from the terminal
```
> nrngui beginSimulation.hoc
```


## Step 3: Setting up the simulation from the GUI

i) Edit the fields in the Date Panel (output files get saved accordingly)

ii) Select Model (Default: Model with Less Susceptible Axon)

iii) Select Perturbation (Default: No Perturbation)

iv) Select either the "200 ms Test Run" or "2000 ms Protocol" on the Control Panel

Note: These durations specify the duration of the current clamp.

In either case, total simulation duration would include additional 15 ms (following the empirical protocol)
prior to the current clamp and 100 ms after the current clamp to ensure complete propagation of elicited 
action potentials to the distal end of the axon (for calculation of firing rate and conduction velocity).


## Step 4: Launching the simulation from the GUI

i) Click the "Run" button on the Control Panel

Note: Prior to onset of simulation the model needs sufficient time to initialize all the compartments using a 
2 step initialization protocol (Rumble et al., 2016).


## Step 5: Observing/accessing results

i) The graphs plot membrane potential traces from different locations 
	along the axon (refer to the index on the graphs). 
	The upper panel plots traces from the proximal (close to the soma) end of the axon.
	The lower panel plots traces from the distal end of the axon.

ii) The Control Panel tracks progress of the simulation. 
	Simulations run 100 ms longer than specified duration to ensure propagation of spikes along the axon 
	before computing the firing rate (FR) and conduction velocity (CV).

iii) The terminal/console displays once a simulation is complete!

iv) Two output files get saved.
	'simulationOutput' file saves just the key parameters and outcome.
	'detailedOutput' file lists other simulation parameters as well.
	


Queries, if any, can be emailed to Nilapratim Sengupta (nilapratim.sengupta@gmail.com).


