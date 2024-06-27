
clear 
close all


%% Import single trials for the silent network
folder_y = 'young_silent_network_singleTrial/stimloc_4'; 

fnames_rates_y = dir(sprintf('%s/rates_*.mat',folder_y));      % import file names
fnames_spikes_y = dir(sprintf('%s/spikes_*.mat',folder_y));    % import file names

Ntrials = length(fnames_rates_y);                              % number of trials

rates_y = cell(1,Ntrials);
spikes_y = cell(1,Ntrials);
for i=1:Ntrials
    file_rates_y = fnames_rates_y(i).name;
    file_spikes_y = fnames_spikes_y(i).name;
    rates_y{i} = load(sprintf('%s/%s', folder_y,file_rates_y));         % load the FRs for all trials
    spikes_y{i} = load(sprintf('%s/%s', folder_y,file_spikes_y));       % load the spikes for all trials
end


%%% Average firing rate for each E/I neuron during the delay period (for each trial)
NE =length(rates_y{1,1}.fr_E);           % number of excitatory neurons

rates_E_y = zeros(Ntrials,NE);
rates_E_cue_y = zeros(Ntrials,NE);
rates_E_DelayBump1_y = zeros(Ntrials,NE);
rates_E_DelayBump2_y = zeros(Ntrials,NE);
for i=1:Ntrials
    rates_E_y(i,:) = rates_y{1,i}.fr_E;                        % average FR for each E neuron during the delay period (for each trial) 
    rates_E_cue_y(i,:) = rates_y{1,i}.fr_cue;                  % average FR for each E neuron during the cue period (for each trial)   
    rates_E_DelayBump1_y(i,:) = rates_y{1,i}.fr_E_DelayBump1;  % average FR for each E neuron during the first Ixt injection period (for each trial) 
    rates_E_DelayBump2_y(i,:) = rates_y{1,i}.fr_E_DelayBump2;  % average FR for each E neuron during the second Ixt injection period (for each trial) 
end


%%% Number of spikes during all the task duration for each E/I neuron 
spkE_counts_y = zeros(Ntrials,NE);
for i=1:Ntrials
    spkE_counts_y(i,:) = spikes_y{1,i}.spike_counts_E_cells;   % number of spikes for each E neuron (for each trial) 
end


%%% Spike trains for each individual neuron
spksE_id_y = spikes_y{1,1}.spikes_E_cells;          % indices of E neurons that fire at each time point
spksE_t_y = spikes_y{1,1}.spike_times_E_cells;      % time points when an E neuron fires a spike


%%% MAKE FIGURES FOR single trials
runtime = 7;
n_windows = 69; 
time = runtime/n_windows:runtime/n_windows:runtime;    

f=figure;
f.Position = [100 100 1600 200];

% Rasterplot for E neurons (1 trial)
subplot(1,3,1)
x = [2; 3; 3; 2];
y = [0; 0; 16000; 16000];
patch( x,y, [0.8 0.8 0.8], 'EdgeColor', 'none' );
hold on
scatter(spksE_t_y(1:50:end),spksE_id_y(1:50:end),100,'.', 'MarkerEdgeColor',[0,0,1])  
x = [4.1; 4.4; 4.4; 4.1];
patch( x,y, [0.8 0.8 0.8], 'FaceColor','none','EdgeColor', [1,0.8,0.5], 'LineWidth', 2);
x = [5.4; 5.7; 5.7; 5.4];
patch( x,y, [0.8 0.8 0.8], 'FaceColor','none','EdgeColor',[0,0.9,0], 'LineWidth', 2);
xlim([0 runtime])
xticks([1  3  5  7])
xticklabels({'-2', '0', '2', '4'})
ylim([-10 16000])
yticks([0  4000  8000  12000  16000])
yticklabels({'0', '90', '180', '270', '360'})
xlabel('time from delay onset (s)')
ylabel('preferred angle (deg)')
fontsize(gca,15,"pixels")

% Average FR for the E neurons during the cue period, and the first and second reactivation periods (for 1 trial)
subplot(1,3,2)
C = convn(rates_E_cue_y(1,:), ones(1,500)/500, 'same');
C1 = convn(rates_E_DelayBump1_y(1,:), ones(1,500)/500, 'same');
C2 = convn(rates_E_DelayBump2_y(1,:), ones(1,500)/500, 'same');
hold on
plot(C(1:end), 1:NE, 'Color', [0 0 0], LineWidth=1)
plot(C1(1:end), 1:NE, 'Color', [1,0.8,0.5], LineWidth=1)
plot(C2(1:end), 1:NE, 'Color', [0,0.8,0], LineWidth=1)
ylim([200 15800])
yticklabels({'', '', '', '', ''})
xlim([0,26])
xticks([0, 10, 20])
xlabel('firing rate (Hz)')
fontsize(gca,15,"pixels")



%% Import 280 trials for the silent network
folder_y = 'young_silent_network_280trials'; 

fnames_dec_y = dir(sprintf('%s/decoding_*.mat',folder_y));       

Ntrials = length(fnames_dec_y); 

decoding_y = cell(1,Ntrials);
for i=1:Ntrials
    file_dec_y = fnames_dec_y(i).name;   
    decoding_y{i} = load(sprintf('%s/%s', folder_y,file_dec_y));    % load the decoding info for all trials
end

%%% Decoding: angle and modulus of the (E) population vector (for each trial)
n_windows = decoding_y{1,1}.number_windows;

dec_angle_y = zeros(Ntrials,n_windows);
dec_modulus_y = zeros(Ntrials,n_windows);
for i=1:Ntrials
    dec_angle_y(i,:) = decoding_y{1,i}.dec_angle;          % decoded angle (in rad) at all time windows (for each trial)    
    ind1_y = find(dec_angle_y(i,:) > pi);
    dec_modulus_y(i,:) = decoding_y{1,i}.dec_modulus;  % decoded modulus at all time windows (for each trial) 
end
aver_dec_modulus_y = mean(dec_modulus_y);               % decoded modulus averaged for all trials (for each network j)


%%% MAKE FIGURES FOR memory strength vs time
subplot(1,3,3)
plot ( time, aver_dec_modulus_y , 'b', 'LineWidth', 2); hold on
xlim([0 runtime])
xticks([1  3  5  7])
xticklabels({'-2', '0', '2', '4'})
ylim([0 0.85])
xlabel('time from delay onset (s)')
ylabel('memory strength')
fontsize(gca,15,"pixels")




