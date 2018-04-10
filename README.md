Code for the simulations' results presented in Section 5.1 of the paper titled "Bootstrapping the Out-of-sample Predictions for Efficient and Accurate Cross-Validation" by Tsamardinos, Greasidou, and Borboudakis,
published in the Machine Learning Journal.

To produce the exact plots shown in the paper run the following commands in Matlab:

simulations_bias_results = machineLearningJournalSimulationsCode(9, 6, 10, 1000, 1000, 0.99);
plotSimulationsBiasResults(simulations_bias_results, [-0.1, 0.2]);