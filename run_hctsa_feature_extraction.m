%% HCTSA Feature Extraction Script
% Handles only the computationally intensive feature extraction:
% 1. Initialize HCTSA dataset from time series input
% 2. Compute features using parallel processing
% 3. Save raw feature matrix to HCTSA.mat
%
% For analysis (normalization, quality checks, etc.), use run_hctsa_analysis.m

%% SETUP: Clear workspace and initialize
clear; clc; close all;
set(0, 'DefaultFigureVisible', 'off');  % Headless mode
startup; % Initialize HCTSA environment

%% CONFIG: Configuration Parameters
INPUT_FILE = 'INP_gait_hctsa_chs0_PW_EM59.mat';
HCTSA_FILENAME = 'HCTSA.mat';
FEATURE_SET = 'hctsa';  % Options: 'hctsa', 'catch22', 'catch24'
NUM_WORKERS = 10;                        % Number of parallel workers
SAVE_INCREMENT = 5;                      % Save progress every N time series
LOG_FOLDER = 'logs';                     % Folder to store all log files

% Initialize parallel pool and logs
parpool(NUM_WORKERS);
mkdir(LOG_FOLDER);

fprintf('=== HCTSA Feature Extraction ===\n');
fprintf('Input: %s → Output: %s\n', INPUT_FILE, HCTSA_FILENAME);

%% STEP1: Initialize HCTSA Dataset
TS_Init(INPUT_FILE, FEATURE_SET, [0,0,0], HCTSA_FILENAME);

%% STEP2: Compute Features
start_time = tic;
sample_runscript_matlab(true, SAVE_INCREMENT, HCTSA_FILENAME);
total_time = toc(start_time);

%% STEP3: Save Summary
summary_info = struct();
summary_info.input_file = INPUT_FILE;
summary_info.output_file = HCTSA_FILENAME;
summary_info.feature_set = FEATURE_SET;
summary_info.num_workers = NUM_WORKERS;
summary_info.computation_time_minutes = total_time/60;
summary_info.completion_time = datestr(now);
summary_info.log_folder = LOG_FOLDER;

save('hctsa_extraction_summary.mat', 'summary_info');
save(fullfile(LOG_FOLDER, 'hctsa_extraction_summary.mat'), 'summary_info');

% Cleanup
poolobj = gcp('nocreate');
if ~isempty(poolobj)
    delete(poolobj);
end

fprintf('✓ Feature extraction completed in %.1f minutes\n', total_time/60);
fprintf('Results saved to: %s\n', HCTSA_FILENAME);

