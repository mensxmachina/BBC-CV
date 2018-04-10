function plotSimulationsBiasResults(simulations_bias_results, y_limits)
    x_axis_range = [20, 40, 60, 80, 100, 120, 140];
    sample_size_range = [20, 40, 60, 80, 100, 500, 1000];
    
    markers = {'-o', '-.x', '--s', '--o', '-x', '-.s', '--s'};
    
    % CVT
    plotBias(simulations_bias_results.mean_performance_bias_CV, ...
        sample_size_range, x_axis_range, markers, y_limits, 'CVT', ...
        'sample size', 'average bias (acc)', './CVT_bias');
    
    % TT
    plotBias(simulations_bias_results.mean_performance_bias_TT, ...
        sample_size_range, x_axis_range, markers, y_limits, 'TT', ...
        'sample size', 'average bias (acc)', './TT_bias');
    
    % NCV
    plotBias(simulations_bias_results.mean_performance_bias_NCV, ...
        sample_size_range, x_axis_range, markers, y_limits, 'NCV', ...
        'sample size', 'average bias (acc)', './NCV_bias');
    
    % BBC-CV
    plotBias(simulations_bias_results.mean_performance_bias_BBC_CV, ...
        sample_size_range, x_axis_range, markers, y_limits, 'BBC-CV', ...
        'sample size', 'average bias (acc)', 'BBC-CV_bias');
    
    % BBCD-CV
    plotBias(simulations_bias_results.mean_performance_bias_BBCD_CV, ...
        sample_size_range, x_axis_range, markers, y_limits, 'BBCD-CV', ...
        'sample size', 'average bias (acc)', 'BBCD-CV_bias')
end

function plotBias(bias, sample_size_range, x_axis_range, markers, y_limits, ...
        title_str, xlabel_str, ylabel_str, name_to_save)
    figure();
    plot(x_axis_range, bias(1, :), markers{1}, 'Color', 'b', 'LineWidth', 1)
    hold on
    plot(x_axis_range, bias(2, :), markers{2}, 'Color', 'b', 'LineWidth', 1)
    plot(x_axis_range, bias(3, :), markers{3}, 'Color', 'b', 'LineWidth', 1)
    plot(x_axis_range, bias(4, :), markers{4}, 'Color', [0, 0.6, 0], 'LineWidth', 1)
    plot(x_axis_range, bias(5, :), markers{5}, 'Color', [0, 0.6, 0], 'LineWidth', 1)
    plot(x_axis_range, bias(6, :), markers{6}, 'Color', 'r', 'LineWidth', 1)
    plot(x_axis_range, bias(7, :), markers{7}, 'Color', 'r', 'LineWidth', 1)
    plot(x_axis_range, zeros(7, 1), '--.k', 'LineWidth', 1)
    hold off
    grid on
    ylim(y_limits)
    set(gca, 'XTick', x_axis_range)
    set(gca, 'XTickLabel', sample_size_range)
    hXLabel = xlabel(xlabel_str);
    hYLabel = ylabel(ylabel_str);
    hTitle = title(title_str);
    
    set([hXLabel, hYLabel], 'FontSize', 14);
    set(hTitle, 'FontSize', 16, 'FontWeight', 'bold');
    set(gcf, 'PaperPositionMode', 'auto');
    print(sprintf('%s.eps', name_to_save),'-depsc2');
end




