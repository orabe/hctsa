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
set(0, 'DefaultFigureVisible', 'off');  % Headless mode
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
GROUP_KEYWORDS = {'normalWalk', 'gaitMod'}; % Group labeling keywords
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


%% STEP1: Load HCTSA Data
[TS_DataMat_raw, TimeSeries, Operations] = TS_LoadData(HCTSA_FILENAME);
MasterOperations = TS_GetFromData(HCTSA_FILENAME, 'MasterOperations');
fprintf('STEP1: Dataset loaded: %d time series × %d features\n', height(TimeSeries), height(Operations));

% Export raw data tables immediately after loading
data_output_dir = fullfile('data', 'hctsa_output_data');
if ~exist(data_output_dir, 'dir')
    mkdir(data_output_dir);
end

% Export the raw data tables (overwrites existing files)
writetable(TimeSeries, fullfile(data_output_dir, 'TimeSeries.csv'));
writetable(Operations, fullfile(data_output_dir, 'Operations.csv'));
writetable(MasterOperations, fullfile(data_output_dir, 'MasterOperations.csv'));
fprintf('STEP1: Raw data tables exported to: %s\n', data_output_dir);

fprintf('-----------------------------\n');

%% STEP2: Quality Inspection
TS_InspectQuality('summary', HCTSA_FILENAME);
TS_InspectQuality('master', HCTSA_FILENAME);
TS_InspectQuality('reduced', HCTSA_FILENAME);
TS_InspectQuality('full', HCTSA_FILENAME);
fprintf('STEP2: Quality inspection completed\n');
fprintf('-----------------------------\n');

%% STEP3: Group Labeling
TS_LabelGroups(HCTSA_FILENAME, GROUP_KEYWORDS);
fprintf('STEP3: Group labeling completed\n');
fprintf('-----------------------------\n');

%% STEP4: Data Normalization
outputFileName = TS_Normalize(NORM_METHOD, FILTER_THRESHOLD, HCTSA_FILENAME, CLASS_VAR_FILTER);
fprintf('STEP4: Data normalized and saved to: %s\n', outputFileName);

% Export key data tables immediately after normalization
[TS_DataMat_norm, TimeSeries_norm, Operations_norm] = TS_LoadData(outputFileName);
MasterOperations_norm = TS_GetFromData(outputFileName, 'MasterOperations');

% Create data output directory
data_output_dir = fullfile('data', 'hctsa_output_data');
if ~exist(data_output_dir, 'dir')
    mkdir(data_output_dir);
end

% Export the data tables (overwrites existing files)
writetable(TimeSeries_norm, fullfile(data_output_dir, 'TimeSeries_N.csv'));
writetable(Operations_norm, fullfile(data_output_dir, 'Operations_N.csv'));
writetable(MasterOperations_norm, fullfile(data_output_dir, 'MasterOperations_N.csv'));
fprintf('STEP4: Key data tables exported to: %s\n', data_output_dir);

fprintf('-----------------------------\n');

%% STEP5: Time Series and Data Matrix Visualization
TS_PlotTimeSeries('norm');
% Save time series plot
png_file = fullfile(FIGURES_TIMESTAMP_FOLDER, sprintf('timeseries_plot_%s.png', TIMESTAMP));
print(png_file, '-dpng', '-r300');

TS_PlotDataMatrix('whatData', outputFileName, 'addTimeSeries', 1);
% Save data matrix plot
png_file = fullfile(FIGURES_TIMESTAMP_FOLDER, sprintf('datamatrix_plot_%s.png', TIMESTAMP));
print(png_file, '-dpng', '-r300');

TS_PlotDataMatrix('whatData', outputFileName, 'addTimeSeries', 1, 'colorGroups', 1);
% Save colored data matrix plot
png_file = fullfile(FIGURES_TIMESTAMP_FOLDER, sprintf('datamatrix_colored_%s.png', TIMESTAMP));
print(png_file, '-dpng', '-r300');
fprintf('STEP5: Visualization plots completed\n');
fprintf('-----------------------------\n');

%% STEP6: Clustering Analysis
TS_Cluster('euclidean', 'average', 'corr_fast', 'average');

TS_PlotDataMatrix('whatData', outputFileName, 'addTimeSeries', 1);
% Save clustered data matrix plot
png_file = fullfile(FIGURES_TIMESTAMP_FOLDER, sprintf('clustered_datamatrix_%s.png', TIMESTAMP));
print(png_file, '-dpng', '-r300');

TS_PlotDataMatrix('whatData', outputFileName, 'addTimeSeries', 1, 'colorGroups', 1);
% Save clustered colored data matrix plot
png_file = fullfile(FIGURES_TIMESTAMP_FOLDER, sprintf('clustered_datamatrix_colored_%s.png', TIMESTAMP));
print(png_file, '-dpng', '-r300');
fprintf('STEP6: Clustering analysis completed\n');
fprintf('-----------------------------\n');

%% STEP7: Low-Dimensional Visualizations
TS_PlotLowDim(outputFileName, 'pca');
% Save PCA plot
png_file = fullfile(FIGURES_TIMESTAMP_FOLDER, sprintf('pca_plot_%s.png', TIMESTAMP));
print(png_file, '-dpng', '-r300');

TS_PlotLowDim('norm', 'tsne');
% Save t-SNE plot
png_file = fullfile(FIGURES_TIMESTAMP_FOLDER, sprintf('tsne_plot_%s.png', TIMESTAMP));
print(png_file, '-dpng', '-r300');
fprintf('STEP7: Low-dimensional visualizations completed\n');
fprintf('-----------------------------\n');

%% STEP8: Similarity Search Analysis
[~, TimeSeries_norm, Operations_norm] = TS_LoadData(outputFileName);

TS_SimSearch('targetId', TimeSeries_norm.ID(1), 'whatPlots', {'matrix', 'scatter'}, ...
           'tsOrOps', 'ts', 'numNeighbors', height(TimeSeries_norm) - 1);
% Save time series similarity search plots
png_file = fullfile(FIGURES_TIMESTAMP_FOLDER, sprintf('ts_similarity_search_%s.png', TIMESTAMP));
print(png_file, '-dpng', '-r300');

TS_SimSearch('targetId', Operations_norm.ID(1), 'whatPlots', {'matrix', 'scatter'}, ...
           'tsOrOps', 'ops', 'numNeighbors', min(100, height(Operations_norm)));
% Save operations similarity search plots
png_file = fullfile(FIGURES_TIMESTAMP_FOLDER, sprintf('ops_similarity_search_%s.png', TIMESTAMP));
print(png_file, '-dpng', '-r300');
fprintf('STEP8: Similarity search analysis completed\n');
fprintf('-----------------------------\n');

%% STEP9: Individual Feature Analysis
for i = 1:min(3, height(Operations_norm))
    TS_SingleFeature(outputFileName, Operations_norm.ID(i), false);
    % Save individual feature plot
    png_file = fullfile(FIGURES_TIMESTAMP_FOLDER, sprintf('single_feature_%d_%s.png', Operations_norm.ID(i), TIMESTAMP));
    print(png_file, '-dpng', '-r300');
end
fprintf('STEP9: Individual feature analysis completed\n');
fprintf('-----------------------------\n');

%% STEP10: Classification Analysis
cfnParams = GiveMeDefaultClassificationParams(outputFileName);
[meanAcc, nullStats] = TS_Classify(outputFileName, cfnParams, NUM_NULLS, 'doParallel', true);
% Save classification plots
png_file = fullfile(FIGURES_TIMESTAMP_FOLDER, sprintf('classification_results_%s.png', TIMESTAMP));
print(png_file, '-dpng', '-r300');

TS_CompareFeatureSets();
% Save feature sets comparison
png_file = fullfile(FIGURES_TIMESTAMP_FOLDER, sprintf('feature_sets_comparison_%s.png', TIMESTAMP));
print(png_file, '-dpng', '-r300');

TS_ClassifyLowDim(outputFileName, cfnParams, 5, false);  % Stop after 5 PCs, don't search for perfect accuracy
% Save low-dimensional classification
png_file = fullfile(FIGURES_TIMESTAMP_FOLDER, sprintf('classification_lowdim_%s.png', TIMESTAMP));
print(png_file, '-dpng', '-r300');

fprintf('STEP10: Classification accuracy: %.2f%%\n', meanAcc*100);
fprintf('-----------------------------\n');

%% STEP11: Top Features Analysis
fprintf('STEP11: Running top features analysis...\n');
try
    TS_TopFeatures(outputFileName, 'classification', cfnParams, ...
                  'whatPlots', {'histogram', 'distributions', 'cluster','datamatrix'}, ...
                  'numTopFeatures', NUM_TOP_FEATURES, 'numFeaturesDistr', NUM_FEATURES_DISTR, 'numNulls', NUM_NULLS);
    % Save top features plots
    png_file = fullfile(FIGURES_TIMESTAMP_FOLDER, sprintf('top_features_official_%s.png', TIMESTAMP));
    print(png_file, '-dpng', '-r300');
    fprintf('STEP11: TS_TopFeatures completed successfully\n');
catch ME
    fprintf('STEP11: TS_TopFeatures failed: %s\n', ME.message);
    fprintf('STEP11: Running manual top features analysis...\n');
    manualTopFeatures(outputFileName, cfnParams, NUM_TOP_FEATURES, TXT_TIMESTAMP_FOLDER, FIGURES_TIMESTAMP_FOLDER, TIMESTAMP);
end
fprintf('-----------------------------\n');

%% STEP12: Data Export
% Export to CSV with timestamps in structured folders
raw_csv_file = fullfile(CSV_TIMESTAMP_FOLDER, sprintf('HCTSA_raw_%s.csv', TIMESTAMP));
norm_csv_file = fullfile(CSV_TIMESTAMP_FOLDER, sprintf('HCTSA_normalized_%s.csv', TIMESTAMP));

fprintf('STEP12: Exporting data to CSV files...\n');

% Export raw data directly to timestamped CSV file
[TS_DataMat_raw, TimeSeries_raw, Operations_raw] = TS_LoadData(HCTSA_FILENAME);
writetable([TimeSeries_raw, array2table(TS_DataMat_raw)], raw_csv_file);

% Export normalized data directly to timestamped CSV file
[TS_DataMat_norm, TimeSeries_norm, Operations_norm] = TS_LoadData(outputFileName);
writetable([TimeSeries_norm, array2table(TS_DataMat_norm)], norm_csv_file);

fprintf('STEP12: Data exported to CSV files in %s\n', CSV_TIMESTAMP_FOLDER);
fprintf('-----------------------------\n');

%% STEP13: Group Summary Statistics
[~, TimeSeries, ~] = TS_LoadData(outputFileName);
if ismember('Group', TimeSeries.Properties.VariableNames)
    groups = categorical(TimeSeries.Group);
    group_counts = countcats(groups);
    group_names = categories(groups);
    for i = 1:length(group_names)
        fprintf('STEP13: Group %s: %d time series (%.1f%%)\n', group_names{i}, group_counts(i), ...
                100*group_counts(i)/sum(group_counts));
    end
else
    fprintf('STEP13: No groups found in the data\n');
end
fprintf('-----------------------------\n');

%% STEP14: Final Analysis Summary
[TS_DataMat, TimeSeries, Operations] = TS_LoadData(HCTSA_NORM_FILE);
fprintf('STEP14: HCTSA ANALYSIS COMPLETION SUMMARY\n');
fprintf('STEP14: Normalized dataset: %s\n', HCTSA_NORM_FILE);
fprintf('STEP14: Number of time series: %d\n', height(TimeSeries));
fprintf('STEP14: Number of features: %d\n', height(Operations));
fprintf('STEP14: Data matrix size: %d × %d\n', size(TS_DataMat,1), size(TS_DataMat,2));
fprintf('STEP14: Missing values: %.2f%%\n', 100*sum(isnan(TS_DataMat(:)))/numel(TS_DataMat));

if ismember('Group', TimeSeries.Properties.VariableNames)
    groups = TimeSeries.Group;
    unique_groups = unique(groups);
    fprintf('STEP14: Groups identified: %d\n', length(unique_groups));
    fprintf('STEP14: Classification accuracy: %.2f%%\n', meanAcc*100);
else
    fprintf('STEP14: Groups identified: No\n');
end
fprintf('-----------------------------\n');

%% STEP15: Save Analysis Results
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
    fprintf(fid, 'Normalized dataset: %s\n', HCTSA_NORM_FILE);
    fprintf(fid, 'Number of time series: %d\n', height(TimeSeries));
    fprintf(fid, 'Number of features: %d\n', height(Operations));
    fprintf(fid, 'Data matrix size: %d × %d\n', size(TS_DataMat,1), size(TS_DataMat,2));
    fprintf(fid, 'Missing values: %.2f%%\n', 100*sum(isnan(TS_DataMat(:)))/numel(TS_DataMat));
    
    if ismember('Group', TimeSeries.Properties.VariableNames)
        groups = TimeSeries.Group;
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
    fprintf(fid_csv, 'Normalized Dataset,%s\n', HCTSA_NORM_FILE);
    fprintf(fid_csv, 'Number of Time Series,%d\n', height(TimeSeries));
    fprintf(fid_csv, 'Number of Features,%d\n', height(Operations));
    fprintf(fid_csv, 'Data Matrix Rows,%d\n', size(TS_DataMat,1));
    fprintf(fid_csv, 'Data Matrix Columns,%d\n', size(TS_DataMat,2));
    fprintf(fid_csv, 'Missing Values Percentage,%.2f\n', 100*sum(isnan(TS_DataMat(:)))/numel(TS_DataMat));
    
    if ismember('Group', TimeSeries.Properties.VariableNames)
        groups = TimeSeries.Group;
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
analysis_summary.normalized_data_file = HCTSA_NORM_FILE;
analysis_summary.num_time_series = height(TimeSeries);
analysis_summary.num_operations = height(Operations);
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

if ismember('Group', TimeSeries.Properties.VariableNames)
    groups = TimeSeries.Group;
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

fprintf('STEP15: Analysis results saved to: %s\n', RESULTS_FOLDER);

% Cleanup
poolobj = gcp('nocreate');
if ~isempty(poolobj)
    delete(poolobj);
end

fprintf('STEP15: HCTSA Analysis Pipeline Completed: %s\n', TIMESTAMP_READABLE);
fprintf('-----------------------------\n');

%% Local Functions
function writeToLogFile(filename, message)
    fid = fopen(filename, 'a');
    if fid > 0
        fprintf(fid, '%s\n', message);
        fclose(fid);
    end
end

function saveAndCloseFigure(png_path, description)
    try
        print(png_path, '-dpng', '-r300');
        close all; % Close all figures to free memory
    catch ME
        fprintf('Error saving %s: %s\n', description, ME.message);
    end
end

function manualTopFeatures(dataFile, cfnParams, numTopFeatures, txtFolder, figuresFolder, timestamp)
    % Manual implementation of top features analysis when TS_TopFeatures fails
    
    fprintf('Starting manual top features analysis...\n');
    
    try
        % Load normalized data
        [TS_DataMat, TimeSeries, Operations] = TS_LoadData(dataFile);
        
        % Check if we have groups for classification
        if ~ismember('Group', TimeSeries.Properties.VariableNames)
            fprintf('No groups found - skipping feature discrimination analysis\n');
            return;
        end
        
        % Get group information
        groups = TimeSeries.Group;
        uniqueGroups = unique(groups);
        fprintf('Found %d groups: %s\n', length(uniqueGroups), strjoin(cellstr(uniqueGroups), ', '));
        
        if length(uniqueGroups) ~= 2
            fprintf('Manual analysis requires exactly 2 groups, found %d\n', length(uniqueGroups));
            return;
        end
        
        % Perform Mann-Whitney U test for each feature
        fprintf('Computing feature discriminability (Mann-Whitney U test) for %d features...\n', height(Operations));
        
        pValues = zeros(height(Operations), 1);
        testStats = zeros(height(Operations), 1);
        
        % Use parallel or sequential loop
        poolobj = gcp('nocreate');
        if ~isempty(poolobj)
            fprintf('Using parallel computing for feature analysis (%d workers)\n', poolobj.NumWorkers);
            parfor i = 1:height(Operations)
                [pValues(i), testStats(i)] = computeFeatureTest(TS_DataMat(:, i), groups, uniqueGroups);
            end
        else
            fprintf('Using sequential computation for feature analysis\n');
            for i = 1:height(Operations)
                [pValues(i), testStats(i)] = computeFeatureTest(TS_DataMat(:, i), groups, uniqueGroups);
            end
        end
        
        % Sort by test statistic (higher is better for discrimination)
        [~, sortIdx] = sort(testStats, 'descend');
        
        % Display top discriminating features
        numTopToShow = min(numTopFeatures, height(Operations));
        fprintf('Top %d discriminating features (manual analysis):\n', numTopToShow);
        fprintf('Rank\tFeature ID\tTest Stat\tp-value\tFeature Name\n');
        fprintf('----\t----------\t---------\t-------\t------------\n');
        
        for i = 1:numTopToShow
            idx = sortIdx(i);
            fprintf('%d\t%d\t\t%.2f\t\t%.4f\t%s\n', i, Operations.ID(idx), ...
                    testStats(idx), pValues(idx), Operations.Name{idx});
        end
        
        % Save top features information with timestamp
        topFeatures = struct();
        topFeatures.OperationIDs = Operations.ID(sortIdx(1:numTopToShow));
        topFeatures.OperationNames = Operations.Name(sortIdx(1:numTopToShow));
        topFeatures.TestStatistics = testStats(sortIdx(1:numTopToShow));
        topFeatures.PValues = pValues(sortIdx(1:numTopToShow));
        topFeatures.Groups = uniqueGroups;
        topFeatures.Method = 'Manual Mann-Whitney U test';
        topFeatures.AnalysisTime = TIMESTAMP_READABLE;
        topFeatures.Timestamp = timestamp;
        
        % Save detailed text report
        topFeatures_txt = fullfile(txtFolder, sprintf('top_features_analysis_%s.txt', timestamp));
        fid = fopen(topFeatures_txt, 'w');
        if fid > 0
            fprintf(fid, 'MANUAL TOP FEATURES ANALYSIS\n');
            fprintf(fid, '============================\n');
            fprintf(fid, 'Analysis completed: %s\n', TIMESTAMP_READABLE);
            fprintf(fid, 'Timestamp: %s\n', timestamp);
            fprintf(fid, 'Method: Mann-Whitney U test\n');
            fprintf(fid, 'Groups analyzed: %s vs %s\n', char(uniqueGroups(1)), char(uniqueGroups(2)));
            fprintf(fid, '\nTop %d Discriminating Features:\n', numTopToShow);
            fprintf(fid, 'Rank\tFeature ID\tTest Stat\tp-value\tFeature Name\n');
            fprintf(fid, '----\t----------\t---------\t-------\t------------\n');
            
            for i = 1:numTopToShow
                idx = sortIdx(i);
                fprintf(fid, '%d\t%d\t\t%.2f\t\t%.4f\t%s\n', i, Operations.ID(idx), ...
                        testStats(idx), pValues(idx), Operations.Name{idx});
            end
            fclose(fid);
        end
        
        % Save MATLAB data file
        topFeatures_mat = fullfile(txtFolder, sprintf('top_features_data_%s.mat', timestamp));
        save(topFeatures_mat, 'topFeatures');
        
        fprintf('Top features saved to text report and data file\n');
        
        % Create plots with timestamp
        createTopFeaturesPlots(TS_DataMat, TimeSeries, Operations, sortIdx, numTopToShow, uniqueGroups, figuresFolder, timestamp);
        
    catch ME
        fprintf('Error in manual top features analysis: %s\n', ME.message);
        fprintf('Error details: %s\n', ME.getReport());
    end
end

function [pValue, testStat] = computeFeatureTest(feature_data, groups, uniqueGroups)
    % Compute Mann-Whitney U test for a single feature
    
    % Remove NaN values
    validIdx = ~isnan(feature_data);
    if sum(validIdx) < 4  % Need at least 4 valid points
        pValue = 1;
        testStat = 0;
        return;
    end
    
    data_valid = feature_data(validIdx);
    groups_valid = groups(validIdx);
    
    group1_data = data_valid(groups_valid == uniqueGroups(1));
    group2_data = data_valid(groups_valid == uniqueGroups(2));
    
    if length(group1_data) < 2 || length(group2_data) < 2
        pValue = 1;
        testStat = 0;
        return;
    end
    
    try
        [pValue, ~, stats] = ranksum(group1_data, group2_data);
        testStat = stats.ranksum;
    catch
        pValue = 1;
        testStat = 0;
    end
end

function createTopFeaturesPlots(TS_DataMat, TimeSeries, Operations, sortIdx, numTopFeatures, uniqueGroups, figuresFolder, timestamp)
    % Create basic plots for top features with timestamp
    
    try
        % Plot top 5 features
        numToPlot = min(5, numTopFeatures);
        
        figure('Visible', 'off');
        for i = 1:numToPlot
            subplot(ceil(numToPlot/2), 2, i);
            
            featureIdx = sortIdx(i);
            feature_data = TS_DataMat(:, featureIdx);
            groups = TimeSeries.Group;
            
            % Create group-based histogram
            group1_data = feature_data(groups == uniqueGroups(1));
            group2_data = feature_data(groups == uniqueGroups(2));
            
            % Remove NaN values
            group1_data = group1_data(~isnan(group1_data));
            group2_data = group2_data(~isnan(group2_data));
            
            if ~isempty(group1_data) && ~isempty(group2_data)
                hold on;
                histogram(group1_data, 'FaceAlpha', 0.5, 'DisplayName', char(uniqueGroups(1)));
                histogram(group2_data, 'FaceAlpha', 0.5, 'DisplayName', char(uniqueGroups(2)));
                legend;
                title(sprintf('Feature %d: %s', Operations.ID(featureIdx), Operations.Name{featureIdx}), ...
                      'Interpreter', 'none', 'FontSize', 8);
                xlabel('Feature Value');
                ylabel('Count');
                hold off;
            end
        end
        
        % Save figure as PNG
        pngFile = fullfile(figuresFolder, sprintf('top_features_histograms_%s.png', timestamp));
        print(pngFile, '-dpng', '-r300');
        
        fprintf('Top features plots saved\n');
        
        close all;
        
    catch ME
        fprintf('Error creating plots: %s\n', ME.message);
    end
end
