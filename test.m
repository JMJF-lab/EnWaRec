%% =========================================================
% TEST EXTERNO — LSTM MIMO CON CARACTERÍSTICAS AUMENTADAS (PHREEQC)
%% =========================================================
clc;
clear;
close all;

%% =========================================================
% 1. CARGAR MODELO Y PARÁMETROS AUMENTADOS
%% =========================================================
load('trainnet_phreeqc_aug.mat');  % Tu modelo entrenado con 84 features
load('mu_aug.mat');               % Medias de las 84 variables
load('sigma_aug.mat');            % Desviaciones de las 84 variables
load('windowSize.mat');           % Tamaño de ventana (60)

%% =========================================================
% 2. CARGAR DATASET DE TEST Y APLICAR INGENIERÍA DE CARACTERÍSTICAS
%% =========================================================
load('dataset_phreeqc.mat');      % Carga X e Y originales

% Aplicar exactamente la misma transformación que en el train (Derivadas)
X_diff = [zeros(1, size(X,2)); diff(X)]; 
X_augmented = [X, X_diff];        % Pasa de 42 a 84 canales

% Seleccionar el 30% final para test externo (datos no vistos)
Ntotal = size(X_augmented,1);
idx_split = floor(0.7*Ntotal);
X_test_raw = X_augmented(idx_split+1:end, :);
Y_test_raw = Y(idx_split+1:end, :);

%% =========================================================
% 3. NORMALIZAR X DE TEST
%% =========================================================
Xn = (X_test_raw - mu) ./ sigma;

%% =========================================================
% 4. DIMENSIONES Y SECUENCIAS
%% =========================================================
numFeatures  = size(Xn, 2);
numResponses = size(Y_test_raw, 2);
numTest      = size(Xn, 1) - windowSize;

XTest = zeros(windowSize, numFeatures, numTest, 'single');
YTest = zeros(numTest, numResponses, 'single');
for i = 1:numTest
    XTest(:,:,i) = single(Xn(i:i+windowSize-1, :));
    YTest(i,:)   = Y_test_raw(i+windowSize, :);
end

%% =========================================================
% 5. PREDICCIÓN
%% =========================================================
YPred = predict(net, XTest);
YPred = double(YPred);

%% =========================================================
% 6. MÉTRICAS (Escala Real Física)
%% =========================================================
rmse = sqrt(mean((YPred - YTest).^2, 'all'));
mae  = mean(abs(YPred - YTest), 'all');
ss_tot = sum((YTest - mean(YTest, 'all')).^2, 'all');
ss_res = sum((YTest - YPred).^2, 'all');
R2     = 1 - (ss_res / ss_tot);

fprintf('\n===== RESULTADOS TEST EXTERNO AUMENTADO (PHREEQC) =====\n');
fprintf('RMSE: %.6f\n', rmse);
fprintf('MAE : %.6f\n', mae);
fprintf('R2  : %.6f\n', R2);

%% =========================================================
% 7. VISUALIZACIÓN SENSOR 1
%% =========================================================
figure('Name', 'Sensor 1 - PHREEQC Test Aumentado', 'Color', 'w');
plot(YTest(:,1), 'b-', 'LineWidth', 1.5);
hold on;
plot(YPred(:,1), 'r--', 'LineWidth', 1.5);
legend('Real (PHREEQC)', 'Predicción LSTM Aumentada', 'Location', 'best');
xlabel('Muestras de Test');
ylabel('Masa de Incrustación (Scaling)');
title(sprintf('Sensor 1 | R^2 = %.4f | RMSE = %.4f', R2, rmse));
grid on;

%% =========================================================
% 8. SCATTER GLOBAL
%% =========================================================
figure('Name', 'Correlación Global', 'Color', 'w');
scatter(YTest(:), YPred(:), 12, 'filled');
xlabel('Valores Reales (PHREEQC)');
ylabel('Valores Predichos (LSTM)');
title(sprintf('Correlación Global (R^2 = %.4f)', R2));
grid on;
lsline;

%% =========================================================
% 9. ERROR TEMPORAL
%% =========================================================
figure('Name', 'Error Temporal - Sensor 1', 'Color', 'w');
plot(YTest(:,1) - YPred(:,1), 'k-', 'LineWidth', 1.0);
xlabel('Muestras');
ylabel('Error (Real - Predicho)');
title('Error Temporal - Sensor 1');
grid on;
