%% =========================================================
% ENTRENAMIENTO LSTM MIMO — CONFIGURACIÓN GANADORA (R2 = 0.8294)
%% =========================================================
clc;
clear;
close all;
rng(42,'twister');

%% 1. CARGAR DATASET
load('dataset_phreeqc.mat')
windowSize = 60;  % Ventana óptima validada

%% 2. INGENIERÍA DE CARACTERÍSTICAS (Derivadas temporales)
X_diff = [zeros(1, size(X,2)); diff(X)]; 
X_augmented = [X, X_diff];

%% 3. SPLIT TEMPORAL
Ntotal = size(X_augmented,1);
idx_split = floor(0.7*Ntotal);

X_train_raw = X_augmented(1:idx_split,:);
Y_train_raw = Y(1:idx_split,:);
X_val_raw   = X_augmented(idx_split+1:end,:);
Y_val_raw   = Y(idx_split+1:end,:);

%% 4. NORMALIZAR ÚNICAMENTE LAS ENTRADAS AUMENTADAS (X)
mu = mean(X_train_raw, 1);
sigma = std(X_train_raw, 0, 1);
sigma(sigma == 0) = 1;

X_train_n = (X_train_raw - mu) ./ sigma;
X_val_n   = (X_val_raw - mu)   ./ sigma;

% Guardar parámetros con los nombres originales
save('mu_aug.mat', 'mu');
save('sigma_aug.mat', 'sigma');
save('windowSize.mat', 'windowSize');

%% 5. DIMENSIONES
numFeatures  = size(X_train_n, 2);
numResponses = size(Y_train_raw, 2);
numTrain     = size(X_train_n, 1) - windowSize;
numVal       = size(X_val_n, 1) - windowSize;

%% 6. CREAR SECUENCIAS (Tensores 3D)
XTrain = zeros(windowSize, numFeatures, numTrain, 'single');
TTrain = zeros(numTrain, numResponses, 'single');
for i = 1:numTrain
    XTrain(:,:,i) = single(X_train_n(i:i+windowSize-1, :));
    TTrain(i,:)   = single(Y_train_raw(i+windowSize, :));
end

XValidation = zeros(windowSize, numFeatures, numVal, 'single');
TValidation = zeros(numVal, numResponses, 'single');
for i = 1:numVal
    XValidation(:,:,i) = single(X_val_n(i:i+windowSize-1, :));
    TValidation(i,:)   = single(Y_val_raw(i+windowSize, :));
end

%% 7. RED STACKED LSTM PROFUNDA
layers = [
    sequenceInputLayer(numFeatures)
    
    lstmLayer(256, OutputMode="sequence")
    dropoutLayer(0.2)
    
    lstmLayer(128, OutputMode="last")
    dropoutLayer(0.15)
    
    fullyConnectedLayer(128)
    reluLayer
    dropoutLayer(0.1)
    
    fullyConnectedLayer(64)
    reluLayer
    
    fullyConnectedLayer(numResponses)
];

%% 8. OPCIONES DE ENTRENAMIENTO (Configuración óptima)
options = trainingOptions("adam", ...
    MaxEpochs = 150, ...                  
    MiniBatchSize = 32, ...               
    InitialLearnRate = 4e-4, ...          
    LearnRateSchedule = "piecewise", ...
    LearnRateDropFactor = 0.5, ...
    LearnRateDropPeriod = 40, ...
    GradientThreshold = 1, ...
    L2Regularization = 1e-4, ...          
    Shuffle = "never", ...
    Metrics = "rmse", ...
    ValidationData = {XValidation, TValidation}, ...
    ValidationFrequency = 20, ...
    OutputNetwork = "best-validation", ...
    Verbose = 0, ...
    Plots = "training-progress");

%% 9. ENTRENAMIENTO
disp('Iniciando entrenamiento con configuración óptima (R2 = 0.8294)...');
[net, info] = trainnet(XTrain, TTrain, layers, "mse", options);

%% 10. GUARDAR MODELO (Con el nombre original exacto)
save('trainnet_phreeqc_aug.mat', 'net', 'info', 'mu', 'sigma', 'windowSize');
disp('Modelo guardado correctamente como trainnet_phreeqc_aug.mat.');
