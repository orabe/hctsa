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

% Function to log messages both to terminal and file
logAndPrint = @(msg) logAndPrintToFile(log_file, msg);

% Simple function to log messages (file only)
logToFile = @(msg) writeToLogFile(log_file, msg);

% Function to save and close current figure
saveFigure = @(png_path, description) saveAndCloseFigure(png_path, description);

logAndPrint('=== HCTSA Analysis Pipeline ===');
logAndPrint(sprintf('HCTSA Analysis Pipeline Started: %s', TIMESTAMP_READABLE));
logAndPrint('Created results structure:');
logAndPrint(sprintf('  %s/', RESULTS_FOLDER));
logAndPrint('    logs/');
logAndPrint(sprintf('    analyzed_txt/%s/', TIMESTAMP));
logAndPrint(sprintf('    csv_files/%s/', TIMESTAMP));
logAndPrint(sprintf('    figures/%s/', TIMESTAMP));
saveFigure = @(png_path, description) saveAndCloseFigure(png_path, description);


%% STEP1: Load HCTSA Data
[TS_DataMat_raw, TimeSeries, Operations] = TS_LoadData(HCTSA_FILENAME);
msg = sprintf('Dataset loaded: %d time series × %d features', height(TimeSeries), height(Operations));
logAndPrint(msg);

%% STEP2: Quality Inspection
TS_InspectQuality('summary', HCTSA_FILENAME);
TS_InspectQuality('master', HCTSA_FILENAME);
TS_InspectQuality('reduced', HCTSA_FILENAME);
TS_InspectQuality('full', HCTSA_FILENAME);

%% STEP3: Group Labeling
TS_LabelGroups(HCTSA_FILENAME, GROUP_KEYWORDS);

%% STEP4: Data Normalization
outputFileName = TS_Normalize(NORM_METHOD, FILTER_THRESHOLD, HCTSA_FILENAME, CLASS_VAR_FILTER);
msg = sprintf('Data normalized and saved to: %s', outputFileName);
logAndPrint(msg);

%% STEP5: Time Series and Data Matrix Visualization
TS_PlotTimeSeries('norm');
% Save time series plot
png_file = fullfile(FIGURES_TIMESTAMP_FOLDER, sprintf('timeseries_plot_%s.png', TIMESTAMP));
print(png_file, '-dpng', '-r300');
logAndPrint(sprintf('Time series plot saved: %s', png_file));

TS_PlotDataMatrix('whatData', outputFileName, 'addTimeSeries', 1);
% Save data matrix plot
png_file = fullfile(FIGURES_TIMESTAMP_FOLDER, sprintf('datamatrix_plot_%s.png', TIMESTAMP));
print(png_file, '-dpng', '-r300');
logAndPrint(sprintf('Data matrix plot saved: %s', png_file));

TS_PlotDataMatrix('whatData', outputFileName, 'addTimeSeries', 1, 'colorGroups', 1);
% Save colored data matrix plot
png_file = fullfile(FIGURES_TIMESTAMP_FOLDER, sprintf('datamatrix_colored_%s.png', TIMESTAMP));
print(png_file, '-dpng', '-r300');
logAndPrint(sprintf('Colored data matrix plot saved: %s', png_file));

%% STEP6: Clustering Analysis
TS_Cluster('euclidean', 'average', 'corr_fast', 'average');

TS_PlotDataMatrix('whatData', outputFileName, 'addTimeSeries', 1);
% Save clustered data matrix plot
png_file = fullfile(FIGURES_TIMESTAMP_FOLDER, sprintf('clustered_datamatrix_%s.png', TIMESTAMP));
print(png_file, '-dpng', '-r300');
logAndPrint(sprintf('Clustered data matrix plot saved: %s', png_file));

TS_PlotDataMatrix('whatData', outputFileName, 'addTimeSeries', 1, 'colorGroups', 1);
% Save clustered colored data matrix plot
png_file = fullfile(FIGURES_TIMESTAMP_FOLDER, sprintf('clustered_datamatrix_colored_%s.png', TIMESTAMP));
print(png_file, '-dpng', '-r300');
logAndPrint(sprintf('Clustered colored data matrix plot saved: %s', png_file));

%% STEP7: Low-Dimensional Visualizations
TS_PlotLowDim(outputFileName, 'pca');
% Save PCA plot
png_file = fullfile(FIGURES_TIMESTAMP_FOLDER, sprintf('pca_plot_%s.png', TIMESTAMP));
print(png_file, '-dpng', '-r300');
logAndPrint(sprintf('PCA plot saved: %s', png_file));

TS_PlotLowDim('norm', 'tsne');
% Save t-SNE plot
png_file = fullfile(FIGURES_TIMESTAMP_FOLDER, sprintf('tsne_plot_%s.png', TIMESTAMP));
print(png_file, '-dpng', '-r300');
logAndPrint(sprintf('t-SNE plot saved: %s', png_file));

%% STEP8: Similarity Search Analysis
[~, TimeSeries_norm, Operations_norm] = TS_LoadData(outputFileName);

TS_SimSearch('targetId', TimeSeries_norm.ID(1), 'whatPlots', {'matrix', 'scatter'}, ...
           'tsOrOps', 'ts', 'numNeighbors', height(TimeSeries_norm) - 1);
% Save time series similarity search plots
png_file = fullfile(FIGURES_TIMESTAMP_FOLDER, sprintf('ts_similarity_search_%s.png', TIMESTAMP));
print(png_file, '-dpng', '-r300');
logAndPrint(sprintf('Time series similarity search plot saved: %s', png_file));

TS_SimSearch('targetId', Operations_norm.ID(1), 'whatPlots', {'matrix', 'scatter'}, ...
           'tsOrOps', 'ops', 'numNeighbors', min(100, height(Operations_norm)));
% Save operations similarity search plots
png_file = fullfile(FIGURES_TIMESTAMP_FOLDER, sprintf('ops_similarity_search_%s.png', TIMESTAMP));
print(png_file, '-dpng', '-r300');
logAndPrint(sprintf('Operations similarity search plot saved: %s', png_file));

%% STEP9: Individual Feature Analysis
for i = 1:min(3, height(Operations_norm))
    TS_SingleFeature(outputFileName, Operations_norm.ID(i), false);
    % Save individual feature plot
    png_file = fullfile(FIGURES_TIMESTAMP_FOLDER, sprintf('single_feature_%d_%s.png', Operations_norm.ID(i), TIMESTAMP));
    print(png_file, '-dpng', '-r300');
    logAndPrint(sprintf('Single feature plot %d saved: %s', Operations_norm.ID(i), png_file));
end

%% STEP10: Classification Analysis
cfnParams = GiveMeDefaultClassificationParams(outputFileName);
[meanAcc, nullStats] = TS_Classify(outputFileName, cfnParams, NUM_NULLS, 'doParallel', true);
% Save classification plots
png_file = fullfile(FIGURES_TIMESTAMP_FOLDER, sprintf('classification_results_%s.png', TIMESTAMP));
print(png_file, '-dpng', '-r300');
logAndPrint(sprintf('Classification results plot saved: %s', png_file));

TS_CompareFeatureSets();
% Save feature sets comparison
png_file = fullfile(FIGURES_TIMESTAMP_FOLDER, sprintf('feature_sets_comparison_%s.png', TIMESTAMP));
print(png_file, '-dpng', '-r300');
logAndPrint(sprintf('Feature sets comparison plot saved: %s', png_file));

TS_ClassifyLowDim(outputFileName, cfnParams, 5);
% Save low-dimensional classification
png_file = fullfile(FIGURES_TIMESTAMP_FOLDER, sprintf('classification_lowdim_%s.png', TIMESTAMP));
print(png_file, '-dpng', '-r300');
logAndPrint(sprintf('Low-dimensional classification plot saved: %s', png_file));

msg = sprintf('Classification accuracy: %.2f%%', meanAcc*100);
logAndPrint(msg);
logToFile(msg);

%% STEP11: Top Features Analysis
logAndPrint('Running top features analysis...');
try
    TS_TopFeatures(outputFileName, 'classification', cfnParams, ...
                  'whatPlots', {'histogram', 'distributions', 'cluster','datamatrix'}, ...
                  'numTopFeatures', NUM_TOP_FEATURES, 'numFeaturesDistr', NUM_FEATURES_DISTR, 'numNulls', NUM_NULLS);
    % Save top features plots
    png_file = fullfile(FIGURES_TIMESTAMP_FOLDER, sprintf('top_features_official_%s.png', TIMESTAMP));
    print(png_file, '-dpng', '-r300');
    logAndPrint(sprintf('Official top features plot saved: %s', png_file));
    logAndPrint('TS_TopFeatures completed successfully');
catch ME
    logAndPrint(sprintf('TS_TopFeatures failed: %s', ME.message));
    logAndPrint('Running manual top features analysis...');
    manualTopFeatures(outputFileName, cfnParams, NUM_TOP_FEATURES, TXT_TIMESTAMP_FOLDER, FIGURES_TIMESTAMP_FOLDER, TIMESTAMP);
end

%% STEP12: Data Export
% Export to CSV with timestamps in structured folders
raw_csv_file = fullfile(CSV_TIMESTAMP_FOLDER, sprintf('HCTSA_raw_%s.csv', TIMESTAMP));
norm_csv_file = fullfile(CSV_TIMESTAMP_FOLDER, sprintf('HCTSA_normalized_%s.csv', TIMESTAMP));

logAndPrint('Exporting data directly to timestamped CSV files...');

% Export raw data directly to timestamped CSV file
[TS_DataMat_raw, TimeSeries_raw, Operations_raw] = TS_LoadData(HCTSA_FILENAME);
writetable([TimeSeries_raw, array2table(TS_DataMat_raw)], raw_csv_file);
logAndPrint(sprintf('Raw data CSV saved to: %s', raw_csv_file));

% Export normalized data directly to timestamped CSV file
[TS_DataMat_norm, TimeSeries_norm, Operations_norm] = TS_LoadData(outputFileName);
writetable([TimeSeries_norm, array2table(TS_DataMat_norm)], norm_csv_file);
logAndPrint(sprintf('Normalized data CSV saved to: %s', norm_csv_file));

msg = sprintf('Data exported to CSV files in %s', CSV_TIMESTAMP_FOLDER);
logAndPrint(msg);

%% STEP13: Group Summary Statistics
[~, TimeSeries, ~] = TS_LoadData(outputFileName);
if ismember('Group', TimeSeries.Properties.VariableNames)
    groups = categorical(TimeSeries.Group);
    group_counts = countcats(groups);
    group_names = categories(groups);
    for i = 1:length(group_names)
        msg = sprintf('Group %s: %d time series (%.1f%%)', group_names{i}, group_counts(i), ...
                100*group_counts(i)/sum(group_counts));
        logAndPrint(msg);
    end
else
    msg = 'No groups found in the data';
    logAndPrint(msg);
end

%% STEP14: Final Analysis Summary
[TS_DataMat, TimeSeries, Operations] = TS_LoadData(HCTSA_NORM_FILE);
logAndPrint('HCTSA ANALYSIS COMPLETION SUMMARY');

msg = sprintf('Normalized dataset: %s', HCTSA_NORM_FILE);
logAndPrint(msg);

msg = sprintf('Number of time series: %d', height(TimeSeries));
logAndPrint(msg);

msg = sprintf('Number of features: %d', height(Operations));
logAndPrint(msg);

msg = sprintf('Data matrix size: %d × %d', size(TS_DataMat,1), size(TS_DataMat,2));
logAndPrint(msg);

msg = sprintf('Missing values: %.2f%%', 100*sum(isnan(TS_DataMat(:)))/numel(TS_DataMat));
logAndPrint(msg);

if ismember('Group', TimeSeries.Properties.VariableNames)
    groups = TimeSeries.Group;
    unique_groups = unique(groups);
    msg = sprintf('Groups identified: %d', length(unique_groups));
    logAndPrint(msg);
    
    msg = sprintf('Classification accuracy: %.2f%%', meanAcc*100);
    logAndPrint(msg);
else
    msg = 'Groups identified: No';
    logAndPrint(msg);
end

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

logAndPrint('Analysis results saved:');
logAndPrint(sprintf('  Workspace: %s', analysis_workspace_file));
logAndPrint(sprintf('  Summary text: %s', analysis_txt_file));
logAndPrint(sprintf('  Summary CSV: %s', analysis_csv_file));
logAndPrint(sprintf('  Summary data: %s', analysis_summary_file));

% Cleanup
poolobj = gcp('nocreate');
if ~isempty(poolobj)
    delete(poolobj);
end

msg = sprintf('HCTSA Analysis Pipeline Completed: %s', TIMESTAMP_READABLE);
logAndPrint(msg);

msg = sprintf('Results saved to structured folders in: %s', RESULTS_FOLDER);
logAndPrint(msg);

%% Local Functions
function writeToLogFile(filename, message)
    fid = fopen(filename, 'a');
    if fid > 0
        fprintf(fid, '%s\n', message);
        fclose(fid);
    end
end

function logAndPrintToFile(filename, message)
    % Log to both terminal and file
    fprintf('%s\n', message);  % Print to terminal
    fid = fopen(filename, 'a');
    if fid > 0
        fprintf(fid, '%s\n', message);  % Write to file
        fclose(fid);
    end
end

function saveAndCloseFigure(png_path, description)
    % Create a basic log file path for error reporting
    log_file = fullfile('results', 'logs', sprintf('hctsa_analysis_log_%s.txt', datestr(now, 'yyyymmdd_HHMMSS')));
    
    try
        print(png_path, '-dpng', '-r300');
        % Note: This function doesn't have access to the log file directly
        % But the calling code will log the save operations
        close all; % Close all figures to free memory
    catch ME
        % Log error to both terminal and file
        error_msg = sprintf('Error saving %s: %s', description, ME.message);
        fprintf('%s\n', error_msg);
        if exist(log_file, 'file')
            logAndPrintToFile(log_file, error_msg);
        end
    end
end

function manualTopFeatures(dataFile, cfnParams, numTopFeatures, txtFolder, figuresFolder, timestamp)
    % Manual implementation of top features analysis when TS_TopFeatures fails
    
    % Create a local logging function that can access the log file
    log_file = fullfile('results', 'logs', sprintf('hctsa_analysis_log_%s.txt', timestamp));
    localLogAndPrint = @(msg) logAndPrintToFile(log_file, msg);
    
    localLogAndPrint('Starting manual top features analysis...');
    
    try
        % Load normalized data
        [TS_DataMat, TimeSeries, Operations] = TS_LoadData(dataFile);
        
        % Check if we have groups for classification
        if ~ismember('Group', TimeSeries.Properties.VariableNames)
            localLogAndPrint('No groups found - skipping feature discrimination analysis');
            return;
        end
        
        % Get group information
        groups = TimeSeries.Group;
        uniqueGroups = unique(groups);
        localLogAndPrint(sprintf('Found %d groups: %s', length(uniqueGroups), strjoin(cellstr(uniqueGroups), ', ')));
        
        if length(uniqueGroups) ~= 2
            localLogAndPrint(sprintf('Manual analysis requires exactly 2 groups, found %d', length(uniqueGroups)));
            return;
        end
        
        % Perform Mann-Whitney U test for each feature
        localLogAndPrint(sprintf('Computing feature discriminability (Mann-Whitney U test) for %d features...', height(Operations)));
        
        pValues = zeros(height(Operations), 1);
        testStats = zeros(height(Operations), 1);
        
        % Use parallel or sequential loop
        poolobj = gcp('nocreate');
        if ~isempty(poolobj)
            localLogAndPrint(sprintf('Using parallel computing for feature analysis (%d workers)', poolobj.NumWorkers));
            parfor i = 1:height(Operations)
                [pValues(i), testStats(i)] = computeFeatureTest(TS_DataMat(:, i), groups, uniqueGroups);
            end
        else
            localLogAndPrint('Using sequential computation for feature analysis');
            for i = 1:height(Operations)
                [pValues(i), testStats(i)] = computeFeatureTest(TS_DataMat(:, i), groups, uniqueGroups);
            end
        end
        
        % Sort by test statistic (higher is better for discrimination)
        [~, sortIdx] = sort(testStats, 'descend');
        
        % Display top discriminating features
        numTopToShow = min(numTopFeatures, height(Operations));
        localLogAndPrint(sprintf('Top %d discriminating features (manual analysis):', numTopToShow));
        localLogAndPrint('Rank	Feature ID	Test Stat	p-value	Feature Name');
        localLogAndPrint('----	----------	---------	-------	------------');
        
        for i = 1:numTopToShow
            idx = sortIdx(i);
            localLogAndPrint(sprintf('%d	%d		%.2f		%.4f	%s', i, Operations.ID(idx), ...
                    testStats(idx), pValues(idx), Operations.Name{idx}));
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
        
        localLogAndPrint('Top features saved to:');
        localLogAndPrint(sprintf('  Text report: %s', topFeatures_txt));
        localLogAndPrint(sprintf('  Data file: %s', topFeatures_mat));
        
        % Create plots with timestamp
        createTopFeaturesPlots(TS_DataMat, TimeSeries, Operations, sortIdx, numTopToShow, uniqueGroups, figuresFolder, timestamp);
        
    catch ME
        localLogAndPrint(sprintf('Error in manual top features analysis: %s', ME.message));
        localLogAndPrint(sprintf('Error details: %s', ME.getReport()));
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
    
    % Create a local logging function that can access the log file
    log_file = fullfile('results', 'logs', sprintf('hctsa_analysis_log_%s.txt', timestamp));
    localLogAndPrint = @(msg) logAndPrintToFile(log_file, msg);
    
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
        
        localLogAndPrint(sprintf('Top features plots saved to: %s', pngFile));
        
        close all;
        
    catch ME
        localLogAndPrint(sprintf('Error creating plots: %s', ME.message));
    end
end
