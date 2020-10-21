# ConditionalMetaLearning
We propose a conditional Meta-Learning approach to Biased Regularization and Fine Tuning for heterogeneous tasks.

This repository is the official implementation of the paper:

'The Advantage of Conditional Meta-Learning for Biased Regularization and Fine Tuning' (ID: 6479)

The scripts are organized in the following folders/files.
1) 'data' folder: it contains the Schools and the Lenk dataset we used for our real experiments.
2) 'saved_results': it contains the results we got.
3) 'src' folder: it contains the following files.
    A) 'data_management.py': it generates the data for the different experimental settings.
    B) 'general_functions.py': it contains the basic functions used (such as loss, loss subgradient, feature map).
    C) 'inner_algorithm.py': it contains the implementation of the online inner algorithm (fine tuning variant).
    D) 'methods.py': it contains the implementation of the inner algorithm with a fixed meta-parameter vector (constant
       conditioning function), the implementation of the unconditional meta-learning approach and the implementation of
       the conditional meta-learning approach.
4) 'post_processing.py': it allows to plot the results memorized in 'saved_results' folder.
5) 'main_script.py': it allows to run the methods on the different experimental setting.
