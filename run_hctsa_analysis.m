%% HCTSA Analysis Script
% This script performs quality control, normalization, and analysis on HCTSA features:
% 1. Quality inspection of raw features
% 2. Data normalization and filtering
% 3. Exploratory data analysis and visualization
% 4. Group labeling and classification
% 5. Feature analysis and interpretation
%
% Run this script AFTER running run_hctsa_feature_extraction.m

%% SETUP: Clear workspace and initialize
clear; clc; close all;
% Note: Don't set figures invisible initially - some HCTSA functions need visible figures
startup; % Initialize HCTSA environment

%% CONFIG: Analysis Configuration
NUM_WORKERS = 10;                        % Number of parallel workers
TIMESTAMP = datestr(now, 'yyyymmdd_HHMMSS'); % Timestamp for all files
TIMESTAMP_READABLE = datestr(now, 'dd-mmm-yyyy HH:MM:SS'); % Human-readable timestamp
HCTSA_FILENAME = 'HCTSA.mat';             % Raw feature matrix file
HCTSA_NORM_FILE = 'HCTSA_N.mat';         % Normalized feature matrix file
NORM_METHOD = 'mixedSigmoid';             % Normalization method
FILTER_THRESHOLD = [0.7, 1.0];           % [time series threshold, operation threshold]
CLASS_VAR_FILTER = true;                 % Filter on class variance
GROUP_KEYWORDS = {'normal_walking', 'gait_modulation'}; % Group labeling keywords

NUM_TOP_FEATURES = 40;
NUM_FEATURES_DISTR = 10;
NUM_NULLS = 20;
RESULTS_PREFIX = 'hctsa_analysis';

% Initialize parallel pool and logs
poolobj = gcp('nocreate');
if ~isempty(poolobj)
    delete(poolobj);
end
parpool(NUM_WORKERS);

% Create structured results folders
RESULTS_FOLDER = 'results';              % Main results folder
mkdir(RESULTS_FOLDER);

TXT_FOLDER = fullfile(RESULTS_FOLDER, 'analyzed_txt');
CSV_FOLDER = fullfile(RESULTS_FOLDER, 'csv_files');
FIGURES_FOLDER = fullfile(RESULTS_FOLDER, 'figures');
LOG_FOLDER = fullfile(RESULTS_FOLDER, 'logs')

mkdir(TXT_FOLDER);
mkdir(CSV_FOLDER);
mkdir(FIGURES_FOLDER);
mkdir(LOG_FOLDER);                       % Create logs folder under results

% Create timestamped subfolders within each category
TXT_TIMESTAMP_FOLDER = fullfile(TXT_FOLDER, TIMESTAMP);
CSV_TIMESTAMP_FOLDER = fullfile(CSV_FOLDER, TIMESTAMP);
FIGURES_TIMESTAMP_FOLDER = fullfile(FIGURES_FOLDER, TIMESTAMP);
mkdir(TXT_TIMESTAMP_FOLDER);
mkdir(CSV_TIMESTAMP_FOLDER);
mkdir(FIGURES_TIMESTAMP_FOLDER);

% Verify folder creation
fprintf('Created timestamped folders:\n');
fprintf('  TXT: %s (exists: %s)\n', TXT_TIMESTAMP_FOLDER, string(exist(TXT_TIMESTAMP_FOLDER, 'dir') == 7));
fprintf('  CSV: %s (exists: %s)\n', CSV_TIMESTAMP_FOLDER, string(exist(CSV_TIMESTAMP_FOLDER, 'dir') == 7));
fprintf('  FIGURES: %s (exists: %s)\n', FIGURES_TIMESTAMP_FOLDER, string(exist(FIGURES_TIMESTAMP_FOLDER, 'dir') == 7));

% Setup logging
log_file = fullfile(LOG_FOLDER, sprintf('hctsa_analysis_log_%s.txt', TIMESTAMP));

fid = fopen(log_file, 'w');
fprintf(fid, '=== HCTSA Analysis Log Started: %s ===\n', TIMESTAMP_READABLE);
fclose(fid);

fprintf('=== HCTSA Analysis Pipeline ===\n');
fprintf('HCTSA Analysis Pipeline Started: %s\n', TIMESTAMP_READABLE);
fprintf('Created results structure:\n');
fprintf('  %s/\n', RESULTS_FOLDER);
fprintf('    logs/\n');
fprintf('    analyzed_txt/%s/\n', TIMESTAMP);
fprintf('    csv_files/%s/\n', TIMESTAMP);
fprintf('    figures/%s/\n', TIMESTAMP);


%% STEP1: Load HCTSA Data [TS_LoadData, TS_GetFromData]
[TS_DataMat_raw, TimeSeries, Operations] = TS_LoadData(HCTSA_FILENAME);
MasterOperations = TS_GetFromData(HCTSA_FILENAME, 'MasterOperations');
fprintf('STEP1 [TS_LoadData, TS_GetFromData]: Dataset loaded: %d time series × %d features\n', height(TimeSeries), height(Operations));

% Export raw data tables immediately after loading
data_output_dir = fullfile('data', 'hctsa_output_data');
if ~exist(data_output_dir, 'dir')
    mkdir(data_output_dir);
end

% Export the raw data tables (overwrites existing files)
writetable(TimeSeries, fullfile(data_output_dir, 'TimeSeries.csv'));
writetable(Operations, fullfile(data_output_dir, 'Operations.csv'));
writetable(MasterOperations, fullfile(data_output_dir, 'MasterOperations.csv'));
fprintf('STEP1 [TS_LoadData, TS_GetFromData]: Raw data tables exported to: %s\n', data_output_dir);

fprintf('-----------------------------\n');

%% STEP1b: Filtering Only (No Normalization) [TS_FilterData]
% Save a version of HCTSA.mat with only filtering (no normalization)

% Create data output directory
data_output_dir = fullfile('data', 'hctsa_output_data');
if ~exist(data_output_dir, 'dir')
    mkdir(data_output_dir);
end

% Use the same filtering thresholds as for normalization, but skip normalization
% Use TS_Normalize with 'none' normalization
TS_Normalize('none', FILTER_THRESHOLD, HCTSA_FILENAME, false); % creates HCTSA_N.mat
movefile('HCTSA_N.mat', 'HCTSA_F.mat'); % renames the file
HCTSA_FILTERED_FILE = 'HCTSA_F.mat';

fprintf('STEP1b [TS_FilterData/TS_Normalize]: Filtered-only HCTSA saved to: %s\n', HCTSA_FILTERED_FILE);

% Export filtered data tables to filtered folder
[TS_DataMat_filt, TimeSeries_filt, Operations_filt] = TS_LoadData(HCTSA_FILTERED_FILE);
MasterOperations_filt = TS_GetFromData(HCTSA_FILTERED_FILE, 'MasterOperations');

writetable(TimeSeries_filt, fullfile(data_output_dir, 'TimeSeries_F.csv'));
writetable(Operations_filt, fullfile(data_output_dir, 'Operations_F.csv'));
writetable(MasterOperations_filt, fullfile(data_output_dir, 'MasterOperations_F.csv'));

fprintf('STEP1b [TS_FilterData/TS_Normalize]: Filtered data tables exported to: %s\n', data_output_dir);

fprintf('-----------------------------\n');

%% STEP2: Quality Inspection [TS_InspectQuality, TS_WhichProblemTS, TS_FeatureSummary]
TS_InspectQuality('summary', HCTSA_FILENAME);
png_file = fullfile(FIGURES_TIMESTAMP_FOLDER, '01_TS_InspectQuality_summary.png');
saveFigureIfExists(png_file, 'Quality Inspection Summary');

TS_InspectQuality('master', HCTSA_FILENAME);
png_file = fullfile(FIGURES_TIMESTAMP_FOLDER, '02_TS_InspectQuality_master.png');
saveFigureIfExists(png_file, 'Quality Inspection Master');

TS_InspectQuality('reduced', HCTSA_FILENAME);
png_file = fullfile(FIGURES_TIMESTAMP_FOLDER, '03_TS_InspectQuality_reduced.png');
saveFigureIfExists(png_file, 'Quality Inspection Reduced');

TS_InspectQuality('full', HCTSA_FILENAME);
png_file = fullfile(FIGURES_TIMESTAMP_FOLDER, '04_TS_InspectQuality_full.png');
saveFigureIfExists(png_file, 'Quality Inspection Full');

% Identify problematic time series
fprintf('Identifying problematic time series...\n');
try
    % Check for time series causing NaN outputs
    problemTS_NaN = TS_WhichProblemTS('NaN', HCTSA_FILENAME);
    fprintf('Time series causing NaN outputs: %d\n', length(problemTS_NaN));
    
    % Check for time series causing Inf outputs
    problemTS_Inf = TS_WhichProblemTS('Inf', HCTSA_FILENAME);
    fprintf('Time series causing Inf outputs: %d\n', length(problemTS_Inf));
    
    % Check for time series causing error outputs
    problemTS_Error = TS_WhichProblemTS('error', HCTSA_FILENAME);
    fprintf('Time series causing error outputs: %d\n', length(problemTS_Error));
    
    % Save problem time series information
    problemTS_info = struct();
    problemTS_info.NaN_producers = problemTS_NaN;
    problemTS_info.Inf_producers = problemTS_Inf;
    problemTS_info.Error_producers = problemTS_Error;
    problemTS_info.analysis_timestamp = TIMESTAMP;
    
    % Save to timestamped folder
    problemTS_file = fullfile(TXT_TIMESTAMP_FOLDER, sprintf('problem_timeseries_%s.mat', TIMESTAMP));
    save(problemTS_file, 'problemTS_info');
    
    % Create detailed text report
    problemTS_txt = fullfile(TXT_TIMESTAMP_FOLDER, sprintf('problem_timeseries_report_%s.txt', TIMESTAMP));
    fid = fopen(problemTS_txt, 'w');
    if fid > 0
        fprintf(fid, 'PROBLEM TIME SERIES ANALYSIS REPORT\n');
        fprintf(fid, '===================================\n');
        fprintf(fid, 'Analysis completed: %s\n', TIMESTAMP_READABLE);
        fprintf(fid, 'Source file: %s\n\n', HCTSA_FILENAME);
        
        fprintf(fid, 'SUMMARY:\n');
        fprintf(fid, 'Time series causing NaN outputs: %d\n', length(problemTS_NaN));
        fprintf(fid, 'Time series causing Inf outputs: %d\n', length(problemTS_Inf));
        fprintf(fid, 'Time series causing error outputs: %d\n\n', length(problemTS_Error));
        
        if ~isempty(problemTS_NaN)
            fprintf(fid, 'NaN-PRODUCING TIME SERIES IDs:\n');
            fprintf(fid, '%s\n\n', mat2str(problemTS_NaN));
        end
        
        if ~isempty(problemTS_Inf)
            fprintf(fid, 'INF-PRODUCING TIME SERIES IDs:\n');
            fprintf(fid, '%s\n\n', mat2str(problemTS_Inf));
        end
        
        if ~isempty(problemTS_Error)
            fprintf(fid, 'ERROR-PRODUCING TIME SERIES IDs:\n');
            fprintf(fid, '%s\n\n', mat2str(problemTS_Error));
        end
        
        % Get unique problematic time series
        allProblemTS = unique([problemTS_NaN; problemTS_Inf; problemTS_Error]);
        if ~isempty(allProblemTS)
            fprintf(fid, 'UNIQUE PROBLEMATIC TIME SERIES: %d total\n', length(allProblemTS));
            fprintf(fid, 'IDs: %s\n\n', mat2str(allProblemTS));
            
            % Try to get time series names if available
            try
                if ~isempty(TimeSeries)
                    fprintf(fid, 'PROBLEMATIC TIME SERIES DETAILS:\n');
                    for i = 1:length(allProblemTS)
                        tsID = allProblemTS(i);
                        tsIdx = find(TimeSeries.ID == tsID);
                        if ~isempty(tsIdx)
                            if ismember('Name', TimeSeries.Properties.VariableNames)
                                fprintf(fid, 'ID %d: %s\n', tsID, TimeSeries.Name{tsIdx(1)});
                            else
                                fprintf(fid, 'ID %d: (name not available)\n', tsID);
                            end
                        end
                    end
                end
            catch
                fprintf(fid, 'Could not retrieve time series details\n');
            end
        else
            fprintf(fid, 'No problematic time series found - data quality is good!\n');
        end
        
        fclose(fid);
    end
    
    fprintf('Problem time series information saved to: %s\n', problemTS_file);
    fprintf('Problem time series report saved to: %s\n', problemTS_txt);
    
catch ME
    fprintf('Warning: Could not analyze problem time series: %s\n', ME.message);
end

fprintf('STEP2 [TS_InspectQuality, TS_WhichProblemTS, TS_FeatureSummary]: Quality inspection completed\n');

if ~exist('myColors','var')
    myColors = {'b', 'r', 'g', 'm', 'c', 'y', 'k'};
end


% Feature summary of raw data
% Use already loaded Operations data
for i = 1:min(3, height(Operations))
    opID = Operations.ID(i);
    fprintf('Generating feature summary for operation ID %d...\n', opID);
    fprintf('HCTSA File: %s\n', HCTSA_FILENAME);
    TS_FeatureSummary(opID, HCTSA_FILENAME);
    png_file = fullfile(FIGURES_TIMESTAMP_FOLDER, sprintf('05_%02d_TS_FeatureSummary_raw_op%d.png', i, opID));
    saveFigureIfExists(png_file, sprintf('Feature Summary Raw Op %d', opID));
end
fprintf('STEP2 [TS_InspectQuality, TS_WhichProblemTS, TS_FeatureSummary]: Feature summary of raw data completed\n');

%% STEP3: Group Labeling [TS_LabelGroups]
TS_LabelGroups(HCTSA_FILENAME, GROUP_KEYWORDS);
fprintf('STEP3 [TS_LabelGroups]: Group labeling completed\n');

fprintf('-----------------------------\n');

%% STEP4: Data Normalization [TS_Normalize, TS_LoadData, TS_GetFromData, TS_FeatureSummary]
normalizedDataFile = TS_Normalize(NORM_METHOD, FILTER_THRESHOLD, HCTSA_FILENAME, CLASS_VAR_FILTER);
fprintf('STEP4 [TS_Normalize]: Data normalized and saved to: %s\n', normalizedDataFile);

% Export key data tables immediately after normalization
[TS_DataMat_norm, TimeSeries_norm, Operations_norm] = TS_LoadData(normalizedDataFile);
MasterOperations_norm = TS_GetFromData(normalizedDataFile, 'MasterOperations');

% Export the data tables (overwrites existing files)
writetable(TimeSeries_norm, fullfile(data_output_dir, 'TimeSeries_N.csv'));
writetable(Operations_norm, fullfile(data_output_dir, 'Operations_N.csv'));
writetable(MasterOperations_norm, fullfile(data_output_dir, 'MasterOperations_N.csv'));
fprintf('STEP4 [TS_LoadData, TS_GetFromData]: Key data tables exported to: %s\n', data_output_dir);

% Feature summary of normalized data
% Use already loaded Operations_norm data
for i = 1:min(3, height(Operations_norm))
    opID = Operations_norm.ID(i);
    TS_FeatureSummary(opID, normalizedDataFile);
    png_file = fullfile(FIGURES_TIMESTAMP_FOLDER, sprintf('06_%02d_TS_FeatureSummary_norm_op%d.png', i, opID));
    saveFigureIfExists(png_file, sprintf('Feature Summary Normalized Op %d', opID));
end
fprintf('STEP4 [TS_FeatureSummary]: Feature summary of normalized data completed\n');

fprintf('-----------------------------\n');

%% STEP5: Time Series and Data Matrix Visualization [TS_PlotTimeSeries, TS_PlotDataMatrix]
TS_PlotTimeSeries('norm');
png_file = fullfile(FIGURES_TIMESTAMP_FOLDER, '07_TS_PlotTimeSeries_norm.png');
saveFigureIfExists(png_file, 'Time Series Plot Normalized');

TS_PlotDataMatrix('whatData', normalizedDataFile, 'addTimeSeries', 1);
png_file = fullfile(FIGURES_TIMESTAMP_FOLDER, '08_TS_PlotDataMatrix_addTS.png');
saveFigureIfExists(png_file, 'Data Matrix with Time Series');

TS_PlotDataMatrix('whatData', normalizedDataFile, 'addTimeSeries', 1, 'colorGroups', 1);
png_file = fullfile(FIGURES_TIMESTAMP_FOLDER, '09_TS_PlotDataMatrix_addTS_colorGroups.png');
saveFigureIfExists(png_file, 'Data Matrix with Color Groups');
fprintf('STEP5 [TS_PlotTimeSeries, TS_PlotDataMatrix]: Visualization plots completed\n');
fprintf('-----------------------------\n');

%% STEP6: Clustering Analysis [TS_Cluster, TS_PlotDataMatrix]
TS_Cluster('euclidean', 'average', 'corr_fast', 'average');
png_file = fullfile(FIGURES_TIMESTAMP_FOLDER, '10_TS_Cluster_euclidean_average.png');
saveFigureIfExists(png_file, 'Clustering Analysis');

TS_PlotDataMatrix('whatData', normalizedDataFile, 'addTimeSeries', 1);
png_file = fullfile(FIGURES_TIMESTAMP_FOLDER, '11_TS_PlotDataMatrix_clustered_addTS.png');
saveFigureIfExists(png_file, 'Data Matrix Clustered with Time Series');

TS_PlotDataMatrix('whatData', normalizedDataFile, 'addTimeSeries', 1, 'colorGroups', 1);
png_file = fullfile(FIGURES_TIMESTAMP_FOLDER, '12_TS_PlotDataMatrix_clustered_colorGroups.png');
saveFigureIfExists(png_file, 'Data Matrix Clustered with Color Groups');
fprintf('STEP6 [TS_Cluster, TS_PlotDataMatrix]: Clustering analysis completed\n');
fprintf('-----------------------------\n');

%% STEP7: Low-Dimensional Visualizations [TS_PlotLowDim]
TS_PlotLowDim(normalizedDataFile, 'pca');
png_file = fullfile(FIGURES_TIMESTAMP_FOLDER, '13_TS_PlotLowDim_pca.png');
saveFigureIfExists(png_file, 'PCA Low-Dimensional Visualization');

TS_PlotLowDim(normalizedDataFile, 'tsne');
png_file = fullfile(FIGURES_TIMESTAMP_FOLDER, '14_TS_PlotLowDim_tsne.png');
saveFigureIfExists(png_file, 't-SNE Low-Dimensional Visualization');
fprintf('STEP7 [TS_PlotLowDim]: Low-dimensional visualizations completed\n');
fprintf('-----------------------------\n');

%% STEP8: Similarity Search Analysis [TS_SimSearch]
% Use already loaded normalized data
TS_SimSearch('targetId', TimeSeries_norm.ID(1), 'whatPlots', {'matrix', 'scatter'}, ...
           'tsOrOps', 'ts', 'numNeighbors', height(TimeSeries_norm) - 1);
png_file = fullfile(FIGURES_TIMESTAMP_FOLDER, sprintf('15_TS_SimSearch_ts_target%d_matrix.png', TimeSeries_norm.ID(1)));
saveFigureIfExists(png_file, 'Similarity Search for Time Series');

TS_SimSearch('targetId', Operations_norm.ID(1), 'whatPlots', {'matrix', 'scatter'}, ...
           'tsOrOps', 'ops', 'numNeighbors', min(100, height(Operations_norm)));
png_file = fullfile(FIGURES_TIMESTAMP_FOLDER, sprintf('16_TS_SimSearch_ops_target%d_matrix.png', Operations_norm.ID(1)));
saveFigureIfExists(png_file, 'Similarity Search for Operations');
fprintf('STEP8 [TS_SimSearch]: Similarity search analysis completed\n');
fprintf('-----------------------------\n');

%% STEP9: Individual Feature Analysis [TS_SingleFeature]
for i = 1:min(3, height(Operations_norm))
    TS_SingleFeature(normalizedDataFile, Operations_norm.ID(i), false);
    png_file = fullfile(FIGURES_TIMESTAMP_FOLDER, sprintf('%02d_TS_SingleFeature_ID%d.png', 16+i, Operations_norm.ID(i)));
    saveFigureIfExists(png_file, sprintf('Single Feature Analysis for Operation %d', Operations_norm.ID(i)));
end
fprintf('STEP9 [TS_SingleFeature]: Individual feature analysis completed\n');
fprintf('-----------------------------\n');

%% STEP10: Classification Analysis [GiveMeDefaultClassificationParams, TS_Classify, TS_CompareFeatureSets, TS_ClassifyLowDim]
cfnParams = GiveMeDefaultClassificationParams(normalizedDataFile);
[meanAcc, nullStats] = TS_Classify(normalizedDataFile, cfnParams, NUM_NULLS, 'doParallel', true);
png_file = fullfile(FIGURES_TIMESTAMP_FOLDER, sprintf('20_TS_Classify_nulls%d.png', NUM_NULLS));
saveFigureIfExists(png_file, 'Classification Analysis');

TS_CompareFeatureSets();
png_file = fullfile(FIGURES_TIMESTAMP_FOLDER, '21_TS_CompareFeatureSets.png');
saveFigureIfExists(png_file, 'Feature Sets Comparison');

TS_ClassifyLowDim(normalizedDataFile, cfnParams, 3, false);
png_file = fullfile(FIGURES_TIMESTAMP_FOLDER, '22_TS_ClassifyLowDim_3PCs.png');
saveFigureIfExists(png_file, 'Low-Dimensional Classification');

fprintf('STEP10 [TS_Classify, TS_CompareFeatureSets, TS_ClassifyLowDim]: Classification accuracy: %.2f%%\n', meanAcc*100);
fprintf('-----------------------------\n');

%% STEP11: Top Features Analysis [TS_TopFeatures]
fprintf('STEP11 [TS_TopFeatures]: Running top features analysis...\n');
TS_TopFeatures(normalizedDataFile, 'classification', cfnParams, ...
              'whatPlots', {'histogram', 'distributions', 'cluster','datamatrix'}, ...
              'numTopFeatures', NUM_TOP_FEATURES, 'numFeaturesDistr', NUM_FEATURES_DISTR, 'numNulls', NUM_NULLS);
png_file = fullfile(FIGURES_TIMESTAMP_FOLDER, '23_TS_TopFeatures_classification.png');
saveFigureIfExists(png_file, 'Top Features Analysis');
fprintf('STEP11 [TS_TopFeatures]: TS_TopFeatures completed\n');
fprintf('-----------------------------\n');

%% STEP12: Data Export [Data Export Functions]
% Export to CSV with timestamps in structured folders
raw_csv_file = fullfile(CSV_TIMESTAMP_FOLDER, sprintf('HCTSA_raw_%s.csv', TIMESTAMP));
norm_csv_file = fullfile(CSV_TIMESTAMP_FOLDER, sprintf('HCTSA_normalized_%s.csv', TIMESTAMP));

fprintf('STEP12 [Data Export]: Exporting data to CSV files...\n');

% Export raw data directly to timestamped CSV file
writetable([TimeSeries, array2table(TS_DataMat_raw)], raw_csv_file);

% Export normalized data directly to timestamped CSV file
writetable([TimeSeries_norm, array2table(TS_DataMat_norm)], norm_csv_file);

fprintf('STEP12 [Data Export]: Data exported to CSV files in %s\n', CSV_TIMESTAMP_FOLDER);
fprintf('-----------------------------\n');

%% STEP13: Final Analysis Summary [Summary Functions]
% Use already loaded normalized data for final summary
fprintf('STEP13 [Summary]: HCTSA ANALYSIS COMPLETION SUMMARY\n');
fprintf('STEP13 [Summary]: Normalized dataset: %s\n', normalizedDataFile);
fprintf('STEP13 [Summary]: Number of time series: %d\n', height(TimeSeries_norm));
fprintf('STEP13 [Summary]: Number of features: %d\n', height(Operations_norm));
fprintf('STEP13 [Summary]: Data matrix size: %d × %d\n', size(TS_DataMat_norm,1), size(TS_DataMat_norm,2));
fprintf('STEP13 [Summary]: Missing values: %.2f%%\n', 100*sum(isnan(TS_DataMat_norm(:)))/numel(TS_DataMat_norm));

if ismember('Group', TimeSeries_norm.Properties.VariableNames)
    groups = categorical(TimeSeries_norm.Group);
    group_counts = countcats(groups);
    group_names = categories(groups);
    unique_groups = unique(TimeSeries_norm.Group);
    fprintf('STEP13 [Summary]: Groups identified: %d\n', length(unique_groups));
    for i = 1:length(group_names)
        fprintf('STEP13 [Summary]: Group %s: %d time series (%.1f%%)\n', group_names{i}, group_counts(i), ...
                100*group_counts(i)/sum(group_counts));
    end
    fprintf('STEP13 [Summary]: Classification accuracy: %.2f%%\n', meanAcc*100);
else
    fprintf('STEP13 [Summary]: Groups identified: No\n');
end
fprintf('-----------------------------\n');

%% STEP14: Save Analysis Results [Save Functions]
% Save analysis workspace with timestamp
analysis_workspace_file = fullfile(RESULTS_FOLDER, sprintf('%s_results_%s.mat', RESULTS_PREFIX, TIMESTAMP));
save(analysis_workspace_file);

% Create analysis summary text file with timestamp
analysis_txt_file = fullfile(TXT_TIMESTAMP_FOLDER, sprintf('analysis_summary_%s.txt', TIMESTAMP));
analysis_csv_file = fullfile(CSV_TIMESTAMP_FOLDER, sprintf('analysis_summary_%s.csv', TIMESTAMP));
fid = fopen(analysis_txt_file, 'w');
if fid > 0
    fprintf(fid, 'HCTSA ANALYSIS SUMMARY\n');
    fprintf(fid, '======================\n');
    fprintf(fid, 'Analysis completed: %s\n', TIMESTAMP_READABLE);
    fprintf(fid, 'Timestamp: %s\n', TIMESTAMP);
    fprintf(fid, '\nDataset Information:\n');
    fprintf(fid, 'Normalized dataset: %s\n', normalizedDataFile);
    fprintf(fid, 'Number of time series: %d\n', height(TimeSeries_norm));
    fprintf(fid, 'Number of features: %d\n', height(Operations_norm));
    fprintf(fid, 'Data matrix size: %d × %d\n', size(TS_DataMat_norm,1), size(TS_DataMat_norm,2));
    fprintf(fid, 'Missing values: %.2f%%\n', 100*sum(isnan(TS_DataMat_norm(:)))/numel(TS_DataMat_norm));
    
    if ismember('Group', TimeSeries_norm.Properties.VariableNames)
        groups = TimeSeries_norm.Group;
        unique_groups = unique(groups);
        fprintf(fid, '\nClassification Results:\n');
        fprintf(fid, 'Groups identified: %d\n', length(unique_groups));
        fprintf(fid, 'Classification accuracy: %.2f%%\n', meanAcc*100);
        
        % Group statistics
        group_counts = countcats(categorical(groups));
        group_names = categories(categorical(groups));
        fprintf(fid, '\nGroup Statistics:\n');
        for i = 1:length(group_names)
            fprintf(fid, '  %s: %d time series (%.1f%%)\n', group_names{i}, group_counts(i), ...
                    100*group_counts(i)/sum(group_counts));
        end
    else
        fprintf(fid, '\nClassification Results:\n');
        fprintf(fid, 'Groups identified: No\n');
    end
    
    fprintf(fid, '\nGenerated Files:\n');
    fprintf(fid, 'Analysis workspace: %s\n', analysis_workspace_file);
    fprintf(fid, 'Raw data CSV: %s\n', raw_csv_file);
    fprintf(fid, 'Normalized data CSV: %s\n', norm_csv_file);
    fprintf(fid, 'Log file: %s\n', log_file);
    
    fclose(fid);
end

% Create analysis summary CSV file with timestamp
fid_csv = fopen(analysis_csv_file, 'w');
if fid_csv > 0
    % Write CSV header
    fprintf(fid_csv, 'Metric,Value\n');
    fprintf(fid_csv, 'Analysis Completed,%s\n', TIMESTAMP_READABLE);
    fprintf(fid_csv, 'Timestamp,%s\n', TIMESTAMP);
    fprintf(fid_csv, 'Normalized Dataset,%s\n', normalizedDataFile);
    fprintf(fid_csv, 'Number of Time Series,%d\n', height(TimeSeries_norm));
    fprintf(fid_csv, 'Number of Features,%d\n', height(Operations_norm));
    fprintf(fid_csv, 'Data Matrix Rows,%d\n', size(TS_DataMat_norm,1));
    fprintf(fid_csv, 'Data Matrix Columns,%d\n', size(TS_DataMat_norm,2));
    fprintf(fid_csv, 'Missing Values Percentage,%.2f\n', 100*sum(isnan(TS_DataMat_norm(:)))/numel(TS_DataMat_norm));
    
    if ismember('Group', TimeSeries_norm.Properties.VariableNames)
        groups = TimeSeries_norm.Group;
        unique_groups = unique(groups);
        fprintf(fid_csv, 'Groups Identified,%d\n', length(unique_groups));
        fprintf(fid_csv, 'Classification Accuracy,%.2f\n', meanAcc*100);
        
        % Group statistics
        group_counts = countcats(categorical(groups));
        group_names = categories(categorical(groups));
        for i = 1:length(group_names)
            fprintf(fid_csv, 'Group %s Count,%d\n', group_names{i}, group_counts(i));
            fprintf(fid_csv, 'Group %s Percentage,%.1f\n', group_names{i}, ...
                    100*group_counts(i)/sum(group_counts));
        end
    else
        fprintf(fid_csv, 'Groups Identified,0\n');
        fprintf(fid_csv, 'Classification Accuracy,N/A\n');
    end
    
    fprintf(fid_csv, 'Analysis Workspace,%s\n', analysis_workspace_file);
    fprintf(fid_csv, 'CSV Files Folder,%s\n', CSV_TIMESTAMP_FOLDER);
    fprintf(fid_csv, 'Figures Folder,%s\n', FIGURES_TIMESTAMP_FOLDER);
    fprintf(fid_csv, 'Text Reports Folder,%s\n', TXT_TIMESTAMP_FOLDER);
    fprintf(fid_csv, 'Log File,%s\n', log_file);
    
    fclose(fid_csv);
end

analysis_summary = struct();
analysis_summary.normalized_data_file = normalizedDataFile;
analysis_summary.num_time_series = height(TimeSeries_norm);
analysis_summary.num_operations = height(Operations_norm);
analysis_summary.analysis_completion_time = TIMESTAMP_READABLE;
analysis_summary.timestamp = TIMESTAMP;
analysis_summary.num_workers = NUM_WORKERS;
analysis_summary.log_file = log_file;
analysis_summary.log_folder = LOG_FOLDER;
analysis_summary.results_folder = RESULTS_FOLDER;
analysis_summary.txt_folder = TXT_TIMESTAMP_FOLDER;
analysis_summary.csv_folder = CSV_TIMESTAMP_FOLDER;
analysis_summary.figures_folder = FIGURES_TIMESTAMP_FOLDER;
analysis_summary.analysis_txt_file = analysis_txt_file;
analysis_summary.analysis_csv_file = analysis_csv_file;

if ismember('Group', TimeSeries_norm.Properties.VariableNames)
    groups = TimeSeries_norm.Group;
    unique_groups = unique(groups);
    analysis_summary.groups_found = true;
    analysis_summary.num_groups = length(unique_groups);
    analysis_summary.group_names = cellstr(unique_groups);
    analysis_summary.classification_accuracy = meanAcc;
else
    analysis_summary.groups_found = false;
end

% Save summary structure with timestamp
analysis_summary_file = fullfile(LOG_FOLDER, sprintf('%s_summary_%s.mat', RESULTS_PREFIX, TIMESTAMP));
save(analysis_summary_file, 'analysis_summary');

fprintf('STEP14 [Save Functions]: Analysis results saved to: %s\n', RESULTS_FOLDER);

% Cleanup
poolobj = gcp('nocreate');
if ~isempty(poolobj)
    delete(poolobj);
end

fprintf('STEP14 [Pipeline Complete]: HCTSA Analysis Pipeline Completed: %s\n', TIMESTAMP_READABLE);
fprintf('-----------------------------\n');

%% Local Functions
function saveFigureIfExists(filename, figDescription)
    % Helper function to save figures with proper error handling
    try
        figHandles = get(0, 'Children');
        if ~isempty(figHandles)
            % Get the most recent figure
            fig = figHandles(1);
            % Make sure the figure is current
            figure(fig);
            % Ensure figure is properly formatted
            set(fig, 'PaperPositionMode', 'auto');
            set(fig, 'Color', 'white');
            % Make figure invisible for headless operation
            set(fig, 'Visible', 'off');
            % Save with explicit format and high resolution
            print(fig, filename, '-dpng', '-r300');
            fprintf('✓ Saved %s: %s\n', figDescription, filename);
            % Close the figure
            close(fig);
        else
            fprintf('⚠ WARNING: No figure to save for %s\n', figDescription);
        end
    catch ME
        fprintf('✗ ERROR saving %s: %s\n', figDescription, ME.message);
        % Close all figures in case of error
        try
            close all;
        catch
            % Ignore close errors
        end
    end
end