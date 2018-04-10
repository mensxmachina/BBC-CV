% Code for the simulations' results presented in Section 5.1 of the paper
% titled "Bootstrapping the Out-of-sample Predictions for Efficient and 
% Accurate Cross-Validation" by Tsamardinos, Greasidou, and Borboudakis,
% published in the Machine Learning Journal.

% To produce the exact plots shown in the paper run the following commands
% in Matlab:
% machineLearningJournalSimulationsCode(9, 6, 10, 1000, 1000, 0.99);
% plotSimulationsBiasResults(simulations_bias_results, [-0.1, 0.2]);

% beta_a, beta_b: 1st and 2nd shape parameters of the Beta distribution
% K: number of folds for the CVT, NCV, TT, BBC-CV, and BBCD-CV methods
% num_bootstraps_BBC_CV: number of bootstraps for the BBC-CV method
% num_bootstraps_BBCD_CV: number of bootstraps for the BBCD-CV method
% alpha: significance threshold for the BBCD-CV method
function simulations_bias_results = machineLearningJournalSimulationsCode...
        (beta_a, beta_b, K, num_bootstraps_BBC_CV, num_bootstraps_BBCD_CV, ...
        alpha)
    
    beta_distr = makedist('Beta', 'a', beta_a, 'b', beta_b);
    
    num_configurations_range = [50 100 200 300 500 1000 2000];
    sample_size_range = [20, 40, 60, 80, 100, 500, 1000];
    
    % number of iterations of the simulations
    num_smlts_iterations = 500;
    
    % bias of the K-Fold CV performance estimate
    performance_bias_CV = zeros(num_smlts_iterations, 1);
    % bias of the BBC-CV performance estimate
    performance_bias_BBC_CV = zeros(num_smlts_iterations, 1);
    % bias of the TT performance estimate
    performance_bias_TT = zeros(num_smlts_iterations, 1);
    % bias of the nested CV performance estimate
    performance_bias_NCV = zeros(num_smlts_iterations, 1);
    % bias of the BBCD-CV performance estimate
    performance_bias_BBCD_CV = zeros(num_smlts_iterations, 1);
    
    c = length(num_configurations_range);
    s = length(sample_size_range);
    
    mean_performance_bias_CV = zeros(c, s);
    mean_performance_bias_BBC_CV = zeros(c, s);
    mean_performance_bias_TT = zeros(c, s);
    mean_performance_bias_NCV = zeros(c, s);
    mean_performance_bias_BBCD_CV = zeros(c, s);
    
    rng(1);  % set random seed
    i = 1;
    for num_configurations = num_configurations_range
        fprintf('\nNumber of models: %i\n', num_configurations);
        j = 1;
        for sample_size = sample_size_range
            fprintf('Sample size: %i\n', sample_size);
            for current_smltn = 1:num_smlts_iterations
                % each column contains the indices to the corresponding
                % fold of the data
                folds = reshape(1:sample_size, sample_size/K, K);
                
                % true simulated accuracies for each model
                true_accuracies = random(beta_distr, num_configurations, 1);
                
                predictions = zeros(sample_size, num_configurations);
                for current_cnfg = 1:num_configurations
                    % simulate sample_size number of predictions with given
                    % accuracy
                    predictions(:, current_cnfg) = rand(sample_size, 1);
                    predictions(:, current_cnfg) = predictions...
                        (:, current_cnfg) < true_accuracies(current_cnfg);
                end
                
                % performance and best configuration returned by
                % K-Fold Cross-Validation (with pooling)
                [performance_CV, best_configuration_CV] = ...
                    max(mean(predictions, 1));
                performance_bias_CV(current_smltn) = ...
                    performance_CV - true_accuracies(best_configuration_CV);
                
                % BBC-CV performance estimate
                [performance_BBC_CV, ~, ~] = ...
                    BBC_CV(sample_size, predictions, num_bootstraps_BBC_CV);
                % bias of the BBC-CV estimate of performance
                performance_bias_BBC_CV(current_smltn) = performance_BBC_CV...
                    - true_accuracies(best_configuration_CV);
                
                % compute the bias of the TT correction method
                performance_bias_TT_tmp = TTCorrection(predictions, folds, ...
                    K, best_configuration_CV);
                performance_bias_TT(current_smltn) = ...
                    performance_bias_CV(current_smltn) - performance_bias_TT_tmp;
                
                % compute the NCV performance estimate
                performance_NCV = NCV(true_accuracies, sample_size, ...
                    num_configurations, folds, K);
                performance_bias_NCV(current_smltn) = ...
                    performance_NCV - true_accuracies(best_configuration_CV);
                
                % bootstrap-based dropping - BBCD-CV
                [performance_BBCD_CV, best_configuration_BBCD_CV, ~, ~] = ...
                    BBCD_CV(num_configurations, sample_size, K, ...
                    num_bootstraps_BBC_CV, folds, predictions, ...
                    num_bootstraps_BBCD_CV, alpha);
                % bias estimation of the corrected performance of BBCD-CV
                performance_bias_BBCD_CV(current_smltn) = performance_BBCD_CV ...
                    - true_accuracies(best_configuration_BBCD_CV);
            end
            
            mean_performance_bias_CV(i, j) = mean(performance_bias_CV);
            mean_performance_bias_BBC_CV(i, j) = mean(performance_bias_BBC_CV);
            mean_performance_bias_TT(i, j) = mean(performance_bias_TT);
            mean_performance_bias_NCV(i, j) = mean(performance_bias_NCV);
            mean_performance_bias_BBCD_CV(i, j) = mean(performance_bias_BBCD_CV);
            j = j + 1;
        end
        i = i + 1;
    end
    
    simulations_bias_results.mean_performance_bias_TT = ...
        mean_performance_bias_TT;
    simulations_bias_results.mean_performance_bias_NCV = ...
        mean_performance_bias_NCV;
    simulations_bias_results.mean_performance_bias_CV = ...
        mean_performance_bias_CV;
    simulations_bias_results.mean_performance_bias_BBC_CV = ...
        mean_performance_bias_BBC_CV;
    simulations_bias_results.mean_performance_bias_BBCD_CV = ...
        mean_performance_bias_BBCD_CV;
    
    save('./simulations_bias_results.mat', 'simulations_bias_results');
    
    %plotSimulationsBiasResults(simulations_bias_results, [-0.1, 0.2]);
end

function [corrected_performance, out_samples_performances, test_boot] = ...
        BBC_CV(sample_size, predictions, num_bootstraps_BBC_CV)
    out_samples_performances = zeros(1, num_bootstraps_BBC_CV);
    boot_out = zeros(1, num_bootstraps_BBC_CV);
    
    for current_bootstrap = 1:num_bootstraps_BBC_CV
        % We create bootstrap versions of the predictions
        % Select samples with resampling
        in_samples = randi(sample_size, sample_size, 1);
        out_samples = predictions(setdiff(1:sample_size, in_samples), :);
        
        [~, best_index] = max(mean(predictions(in_samples, :), 1));
        out_samples_perf = mean(out_samples(:, best_index));
        
        out_samples_performances(current_bootstrap) = out_samples_perf;
        boot_out(current_bootstrap) = max(mean(out_samples, 1)) - ...
            out_samples_perf;
    end
    
    corrected_performance = mean(out_samples_performances);
    test_boot = mean(boot_out);
end

function [performance_BBCD_CV, best_configuration, active_configurations, ...
        trained_models] = BBCD_CV(num_configurations, sample_size, K, ...
        num_bootstraps_BBC_CV, folds, predictions, num_bootstraps_BBCD_CV, ...
        alpha)
    trained_models = 0;
    active_configurations = 1:num_configurations;
    
    for current_fold = 1:K
        trained_models = trained_models + length(active_configurations);
        % select the configuration with the best performance so far
        [~, best_config_index] = max(mean(predictions(...
            folds(:, 1:current_fold), active_configurations)));
        imin = folds(1, 1);
        imax = folds(end, current_fold);
        worse_count = zeros(num_bootstraps_BBCD_CV, ...
            length(active_configurations));
        
        for current_bootstrap = 1:num_bootstraps_BBCD_CV
            in_indices = randi([imin, imax], current_fold*sample_size/K, 1);
            in_samples = predictions(in_indices, active_configurations);
            mean_performances = mean(in_samples);
            worse_count(current_bootstrap, :) = ...
                mean_performances(best_config_index) > mean_performances;
        end
        
        percentage_worse = mean(worse_count);
        drop = percentage_worse > alpha; %% configurations to drop
        active_configurations(drop) = [];
    end
    
    % performance and best configuration returned
    [~, best_configuration_index] = ...
        max(mean(predictions(:, active_configurations), 1));
    best_configuration = active_configurations(best_configuration_index);
    
    % bias correction of the performance of the best configuration
    [performance_BBCD_CV, ~, ~] = BBC_CV(sample_size, ...
        predictions(:, active_configurations), num_bootstraps_BBC_CV);
end

function performance_TT = TTCorrection(predictions, folds, K, ...
        best_configuration_CV)
    performances = zeros(K, 1);
    
    for i = 1:K
        performances(i) = mean(-predictions(folds(:, i), ...
            best_configuration_CV)) + max(mean(predictions(folds(:, i), :)));
    end
    
    performance_TT = mean(performances);
end

function performance_NCV = NCV(true_accuracies, sample_size, ...
        num_configurations, folds, K)
    performances = zeros(K, 1);
    
    for current_fold = 1:K
        predictions = zeros(sample_size, num_configurations);
        
        for current_cnfg = 1:num_configurations
            predictions(:, current_cnfg) = rand(sample_size, 1);
            predictions(:, current_cnfg) = ...
                predictions(:, current_cnfg) < true_accuracies(current_cnfg);
        end
        
        inner_preds = predictions(folds(:, setdiff(1:K, current_fold)), :);
        outer_preds = predictions(folds(:, current_fold), :);
        [~, best_model] = max(mean(inner_preds, 1));
        performances(current_fold) = mean(outer_preds(:, best_model));
    end
    performance_NCV = mean(performances);
end
