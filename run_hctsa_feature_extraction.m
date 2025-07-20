%% HCTSA Feature Extraction Script
% Handles only the computationally intensive feature extraction:
% 1. Initialize HCTSA dataset from time series input
% 2. Compute features using parallel processing
% 3. Save raw feature matrix to HCTSA.mat

%% SETUP: Clear workspace and initialize
clear; clc; close all;
set(0, 'DefaultFigureVisible', 'off');  % Disable figure display
set(0, 'DefaultFigureWindowStyle', 'normal'); % Prevent docked figures
startup; % Initialize HCTSA environment

%% CONFIG: Feature Extraction Configuration
NUM_WORKERS = 10;                        % Number of parallel workers
FEATURE_SET = 'hctsa';                   % Feature set to use: 'hctsa', 'catch22', 'catch24'
TS_DATA_FILE = 'INP_gait_hctsa_chs0_all_patients.mat';            % Time series data file
HCTSA_OUTPUT_FILE = 'HCTSA.mat';         % Output feature matrix file

fprintf('=== HCTSA Feature Extraction Pipeline ===\n');

%% STEP 1: Initialize Parallel Pool
poolobj = gcp('nocreate');
if ~isempty(poolobj)
    delete(poolobj);
end
parpool(NUM_WORKERS);
fprintf('Parallel pool created with %d workers\n', NUM_WORKERS);

%% STEP 2: Initialize HCTSA Dataset
fprintf('Initializing HCTSA dataset...\n');

% Set additional global variables to suppress figures and prompts
global TS_PlotDataMatrix_force
TS_PlotDataMatrix_force = true;
set(0, 'DefaultFigureCreateFcn', @(~,~) set(gcf, 'Visible', 'off'));

% Initialize with specified feature set (verbose off to reduce output)
TS_Init(TS_DATA_FILE, FEATURE_SET, [0,0,0], HCTSA_OUTPUT_FILE);

[~, TimeSeries, Operations] = TS_LoadData(HCTSA_OUTPUT_FILE);
fprintf('Dataset initialized: %d time series × %d operations\n', height(TimeSeries), height(Operations));

%% STEP 3: Feature Computation with Parallel Processing
fprintf('Computing features using parallel processing...\n');

% Compute all features
% TS_Compute(true, [], [], 'missing', HCTSA_OUTPUT_FILE, 'minimal');
TS_Compute(true);

fprintf('Feature computation completed\n');

%% STEP 4: Results Summary
[TS_DataMat, TimeSeries, Operations] = TS_LoadData(HCTSA_OUTPUT_FILE);
completion_rate = (sum(~isnan(TS_DataMat(:))) / numel(TS_DataMat)) * 100;

fprintf('=== EXTRACTION COMPLETE ===\n');
fprintf('Feature matrix: %d × %d\n', size(TS_DataMat, 1), size(TS_DataMat, 2));
fprintf('Completion rate: %.1f%%\n', completion_rate);
fprintf('Output saved to: %s\n', HCTSA_OUTPUT_FILE);
