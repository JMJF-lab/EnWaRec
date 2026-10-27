%% =========================================================
% DATASET TRAIN — AUTOMATED PHREEQC-MATLAB COUPLING
%% =========================================================
clc;
clear;
close all;

N = 9000;
L = 100;
Nx = 50;
x = linspace(0, L, Nx);
time = (0:N-1)';

%% =========================================================
% 1. AUTOMATED IMPORT AND CLEANING OF PHREEQC DATA
%% =========================================================
phreeqc_file = 'simulacion_termica_sstp.tsv';
if exist(phreeqc_file, 'file')
    % Preserve original column names exported by PHREEQC
    opts = detectImportOptions(phreeqc_file, 'FileType', 'text', 'VariableNamingRule', 'preserve');
    data_pqc = readtable(phreeqc_file, opts);
    
    % Extract key columns respecting the exact PHREEQC format
    raw_Temp   = data_pqc.temp;            % Temperature (°C)
    raw_SI     = data_pqc.si_Calcite;      % Calcite Saturation Index
    raw_Precip = abs(data_pqc.d_Calcite);    % Precipitated mass (d_Calcite)
    
    % Ensure temperature points are strictly UNIQUE 
    % to prevent interpolation errors in MATLAB.
    [Temp_PHREEQC, unique_idx] = unique(raw_Temp, 'stable');
    SI_PHREEQC     = raw_SI(unique_idx);
    Precip_PHREEQC = raw_Precip(unique_idx);
    
    disp('PHREEQC thermodynamic data imported and filtered successfully!');
else
    error(['File not found: ', phreeqc_file, ...
           '. Run the thermal simulation in PHREEQC first and save the .tsv file.']);
end

%% =========================================================
% 2. OPERATIONAL VARIABLES AND REALISTIC FLUCTUATIONS
%% =========================================================
Flow_in = 40 + 20*sin(time/400) + 4*randn(N,1);
Pressure_in = 2.8 + 0.4*sin(time/600) + 0.05*randn(N,1);

% Operational temperature oscillating within the range validated by PHREEQC (20-60 °C)
Temp = 38 + 12*sin(time/800) + 0.8*randn(N,1);
Temp = max(min(Temp_PHREEQC), min(max(Temp_PHREEQC), Temp));

% Concentrations of the SSTP base solution with operational noise
Ca = 3.197 + 0.3*sin(time/500) + 0.05*randn(N,1); 
CO3 = 3.797 + 0.3*sin(time/550) + 0.05*randn(N,1); 
pH = 7.0 + 0.2*sin(time/700) + 0.02*randn(N,1);

%% =========================================================
% 3. PIPELINE GEOMETRY (Bends Effect)
%% =========================================================
curvature = 0.5 * ones(1, Nx);
curvature(x > 30 & x < 40) = 2.0;  % Bend 1
curvature(x > 70 & x < 75) = 1.5;  % Bend 2

%% =========================================================
% 4. INITIAL CONDITIONS
%% =========================================================
Scaling = zeros(Nx, N);
Flow_profile = zeros(Nx, N);
Scaling(:, 1) = 0.02 * rand(Nx, 1);
k_flow = 0.02;

%% =========================================================
% 5. COUPLED DYNAMIC SIMULATION
%% =========================================================
for t = 2:N
    % Real-time interpolation over cleaned PHREEQC data
    SI_current = interp1(Temp_PHREEQC, SI_PHREEQC, Temp(t), 'linear', 'extrap');
    precip_potential = interp1(Temp_PHREEQC, Precip_PHREEQC, Temp(t), 'linear', 'extrap');
    
    for i = 2:Nx-1
        Flow_profile(i, t) = Flow_in(t) / (1 + 0.15 * Scaling(i, t-1));
        
        % Kinetics driven by SI and real PHREEQC precipitation
        nucleation = max(0, SI_current - min(SI_PHREEQC)); 
        shear = 1 / (1 + k_flow * Flow_profile(i, t));
        geometry_factor = (1 + 0.18 * curvature(i));
        
        deposition = precip_potential * (1 + nucleation^2) * shear * geometry_factor * 0.1;
        diffusion = 0.03 * (Scaling(i+1, t-1) + Scaling(i-1, t-1) - 2 * Scaling(i, t-1));
        
        Scaling(i, t) = Scaling(i, t-1) + deposition + diffusion;
        Scaling(i, t) = max(0, min(0.6, Scaling(i, t)));
    end
end

%% =========================================================
% 6. SENSOR EXTRACTION AND EXPORT FOR LSTM
%% =========================================================
sensor_pos = round([10, 25, 35, 60, 80, 95] / 100 * Nx);
sensor_pos = max(1, min(Nx, sensor_pos));
n_feat = 7;
X = zeros(N, length(sensor_pos) * n_feat);

for t = 1:N
    for s = 1:length(sensor_pos)
        i = sensor_pos(s);
        b = (s-1) * n_feat;
        X(t, b+1) = Flow_profile(i, t);
        X(t, b+2) = Temp(t);
        X(t, b+3) = Pressure_in(t);
        X(t, b+4) = Ca(t);
        X(t, b+5) = CO3(t);
        X(t, b+6) = pH(t);
        X(t, b+7) = curvature(i);
    end
end

Y = Scaling(sensor_pos, :)';
Y_mean = mean(Y(:));
Y_std = std(Y(:));
Y = (Y - Y_mean) / Y_std;

save('y_norm.mat', 'Y_mean', 'Y_std');
save('dataset_phreeqc.mat', 'X', 'Y', 'time');
disp('LSTM dataset successfully generated and linked with PHREEQC.');
