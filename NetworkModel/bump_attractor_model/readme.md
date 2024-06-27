**Read-me file for the bump attractor network model. To generate Figure 5B.**


This folder contains:

- The python codes to generate a single trial for a young/control network, and for the same network under a demyelination condition and under a remyelination condition (same conditions as in Figure 5B). 
Details:

  - The python codes use the Brian2 simulator to simulate the networks.

  - To generate other demyelination/remyelination conditions, the subsection *"model action potential failure"* needs to be modified accordingly.  

- Data:
  - The *rates* and *spikes* data files contain the information for 1 trial (to generate Figure 5B, left and middle columns).

  - The *decoding* data files are for the 280 trials (to generate Figure 5B, right column).
    
    The decoding data would also allow to generate figures 5C-E (for 1 network).

-	The matlab file `analysis.m` that generates Figures 5B i, ii, and iii. 


Queries, if any, can be emailed to Sara Ibañez (sara.i.solas@gmail.com).
