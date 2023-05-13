# Proximal-Mean-Field-Learning
The folder BinaryClassificationWDBCDataSet contains all the material necessary to complete the binary classification of the WDBC dataset using our algorithm. The necessary data is contained in wbcd.mat. MFL.py performs 10^6 proximal recursions and saves the weighted and unweighted risk functions for both test and training data at each step, as well as the final theta and rho values, as .dat files. For reference, we have included the weighted and unweighted risk values for the test data for each of the three beta values tested. Running Figures.py plots these results.

The folder BinaryClassificationOnJetson contains all the material necessary to complete the four binary classifications run on the Jetson TX2. wdbc_classification.py repeats the experiment run via the code in BinaryClassificationWDBCDataSet, with a vastly improved runtime. In the case of banana_classification.py, diabetes_classification.py, and two_norm_classification.py, running the provided files on a Jetson TX2 completes the corresponding classification problem, reporting the weighted and unweighted accuracy in the terminal.

The folder MultiClassTestCaseOnJetson likewise contains all the needed material to run the test case described in Section 5. Running semeion_classification.py stores the weighted and unweighted risk values at each step and the final values of theta and rho.

Please note that the code in BinaryClassificationOnJetson and MultiClassTestCaseOnJetson is not designed to run on systems other than the Jetson TX2.
