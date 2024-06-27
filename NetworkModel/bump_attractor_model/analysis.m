
% ---------------- CREATED BY S. IBAÑEZ; JULY 2023.
% CODE TO GENERATE FIGURE 6B IN "Myelin dystrophy in the aging
% prefrontal cortex leads to impaired signal transmission and
% working memory decline: a multiscale computational study" ----------------

clear 
%close all


% --------- UNCOMMENT TO GENERATE FIGURE 6B,i ---------
folder = 'young_control_network/data_controlNetwork';   
yaxis = [2.3 2.45];                              % Figure 6B,i (right panel): y-axis limits
yaxis_ticks = [2.3213, 2.3562, 2.3911, 2.4260];  % Figure 6B,i (right panel): y-axis ticks
yaxis_ticklabels = {'133','135','137','139'};    % Figure 6B,i (right panel): y-axis tick labels
color = [0,0,1];                                 % Figure 6B,i: color
% -----------------------------------------------------

% --------- UNCOMMENT TO GENERATE FIGURE 6B,ii --------- 
% folder = 'demyelinated_network/data_dem25segm_remove75lam'; 
% yaxis = [-1 3];                              % Figure 6B,ii (right panel): y-axis limits
% yaxis_ticks = [-0.7854, 0.7854, 2.3562];     % Figure 6B,ii (right panel): y-axis ticks
% yaxis_ticklabels = {'-45','45','135'};       % Figure 6B,ii (right panel): y-axis tick labels
% color = [0.8,0,0];                           % Figure 6B,ii: color
% -----------------------------------------------------
 
% --------- UNCOMMENT TO GENERATE FIGURE 6B,iii --------- 
% folder = 'remyelinated_network/data_rem50segm_after_partialDem25segm_add75lam';   
% yaxis = [2.14 2.59];                         % Figure 6B,iii (right panel): y-axis limits
% yaxis_ticks = [2.1817, 2.3562, 2.5307];      % Figure 6B,iii (right panel): y-axis ticks
% yaxis_ticklabels = {'125','135','145'};      % Figure 6B,iii (right panel): y-axis tick labels
% color = [0.60,0.07,0.93];                    % Figure 6B,iii: color
% -----------------------------------------------------


%%% Import data  
fnames_dec = dir(sprintf('%s/decoding_*.mat',folder));      % import file names
fnames_rates = dir(sprintf('%s/rates_*.mat',folder));       % import file names
fnames_spikes = dir(sprintf('%s/spikes_*.mat',folder));     % import file names
Ntrials = length(fnames_dec);                               % total number of trials

file_rates = fnames_rates.name;
file_spikes = fnames_spikes.name;
rates = load(sprintf('%s/%s', folder,file_rates));          % load the firing rates for 1 trial
spikes = load(sprintf('%s/%s', folder,file_spikes));        % load the spikes for 1 trial

decoding = cell(1,Ntrials);
for i=1:Ntrials
    file_dec = fnames_dec(i).name;  
    decoding{i} = load(sprintf('%s/%s', folder,file_dec));  % load the decoding info for all trials
end


%%% Parameters
NE =length(rates.fr_E);       % number of excitatory neurons
Nstims = 8;                   % number of stimulus locations
runtime = 7;
stim_on = 2;
stim_off = 3;
time = runtime/67:runtime/67:runtime;  % n_windows = 67 

         
%%% Average firing rate for each E neuron during the delay period (for 1 trial) 
rates_E = rates.fr_E; 

%%% Spike trains for each individual neuron (for 1 trial) 
spksE_id = spikes.spikes_E_cells;       % indices of E neurons that fire at each time point
spksE_t = spikes.spike_times_E_cells;   % time points when an E neuron fires a spike

%%% Angle of the stimulus location
stimulus = zeros(Ntrials,1);
for i=1:Ntrials
    stimulus(i) = decoding{1,i}.stim;            % stimulus location for each trial (defined between 0 and NE)
end
stimulus_rad = stimulus*2*pi/NE;                 % stimulus location in rad
ind2 = find(stimulus_rad > pi);
stimulus_rad(ind2) = stimulus_rad(ind2) - 2*pi;  % stimulus location defined between -pi and pi

%%% Define the 8 posible stimulus locations 
stims = ((0:Nstims-1)/Nstims) * 2*pi;            % angle of the stimuli in rad
ind4 = find(stims > pi);
stims(ind4) = stims(ind4) - 2*pi;                % angle of the stimuli defined between -pi and pi
stims = sort(stims);

%%% Decoding: angle and modulus of the population vector (for each trial)
n_windows = decoding{1,1}.number_windows;
dec_angle = zeros(Ntrials,n_windows);
for i=1:Ntrials
    dec_angle(i,:) = decoding{1,i}.dec_angle;      % decoded angle (in rad) at all time windows (for each trial) 
    ind1 = find(dec_angle(i,:) > pi);
    dec_angle(i,ind1) = dec_angle(i,ind1) - 2*pi;  % decoded angle defined between -pi and pi
end


%%% FIGURES

% Figure 6B, left panels: rasterplot for E neurons (single trial)
f = figure;
f.Position(3:4)=[1500,300];
subplot(1,3,1)
x = [2; 3; 3;     2];
y = [0; 0; 16000; 16000];
patch( x,y, [0.8 0.8 0.8] );
hold on
scatter( spksE_t(1:50:end), spksE_id(1:50:end), 100, '.', 'MarkerEdgeColor', color )  
xlim([0 7])
ylim([-10 16000])
xticks([1 3 5 7])
yticks([0  4000  8000  12000  16000])
xticklabels({'-2', '0', '2', '4'})
yticklabels({'0', '90', '180', '270', '360'})
xlabel('time from delay onset (s)')
ylabel('preferred angle (deg)')
fontsize(gca,24,"pixels")

% Figure 6B, middle panels: Average FR for the E neurons during the delay period (single trial)
subplot(1,3,2)
scatter( rates_E(1:10:end), 1:10:NE, 100, '.', 'MarkerEdgeColor', color )  
C = convn(rates_E, ones(1,500)/500, 'same');
hold on
plot( C(1:end), 1:NE, 'k', 'LineWidth', 3 )
xlim([0,70])
yticklabels({''})
xlabel('firing rate (Hz)')
fontsize(gca,24,"pixels")

% Figure 6B, right panels: stimulus location & decoded angle for 1 cue
subplot(1,3,3)
x = [stim_on stim_off stim_off stim_on];
y = [-2*pi -2*pi  2*pi 2*pi];
patch( x,y, [0.8 0.8 0.8] );  
hold on
for i=7:7 %Nstims    
    ind = find(stimulus_rad == stims(i));         
    plot( 0:runtime, stims(i)*ones(size(0:runtime)) , '--k', 'LineWidth', 1.5)   
    plot ( time, dec_angle(ind,:), 'Color', color )  
end
plot( time, mean(dec_angle(ind,:)), 'Color', color, 'LineWidth', 4 )

xlim([2.5 runtime])
ylim(yaxis) 
xticks(3:runtime)
yticks(yaxis_ticks) 
xticklabels({'0','1','2','3','4'})
yticklabels(yaxis_ticklabels)
xlabel('time from delay onset')
ylabel('bump center (deg)')
fontsize(gca,18,"pixels")










