# Matrix Representation Stored - Neural Turing Machine.

An experimental project under Deep Matrix Representation Learning [Gao et al, 2017 and Do et al 2017] applied to Neural Turing Machine [Graves et al, 2014] to increase the representation power of NTMs.

The proposal here is to replace the usual vector representation based RNN controller of an NTM with a possibly more powerful, Matrix RNNs. 

To load a MatrixNTM with desired parameters, instantiate the class MatNTM in MatrixNTM.py

Currently, 3 experiments have been conducted on Matrix NTMs on the tasks (all tasks are on matrix sequences) 

1. Copy Task
2. Repeat Copy Task
3. Associative Recall Task

Dissection Notebooks can be found for all the tasks in their respective directores in SavedModels.

Learning Curves for these tasks are as shown below:

1. **Copy Task** :

    a) Cross Entropy Loss,
    
    b) num_sequences from Unif(1,20),
    
    c) Each sequence is a 5x5 matrix.

![Experimental Copy Task Learning Curve](https://github.com/sydney-machine-learning/Matrix_NeuralTuringMachine/blob/master/Experiments'%20LC/MatCopy_first_train.png "Experimental Copy Task Learning Curve")

2. **Repeat Copy** : 

    a) Cross Entropy Loss,
    
    b) num_sequences from Unif(1,10)
    
    c) num_repeats from Unif(1,10)
    
    d) Each Sequence is a 5x5 matrix.
    
![Experimental Repeat Copy Learning Curve](https://github.com/sydney-machine-learning/Matrix_NeuralTuringMachine/blob/master/Experiments'%20LC/MatRepeatCopy_ReducedLR_first_train.png "Experimental Repeat Copy Learning Curve")

**NOTE**: During the experiment, the learning rate was reduced accidently by 1000 times after 26,000 iterations, that's why it actually didn't converge. This will be corrected in future. However, in it's Dissection Notebook in SavedModels, you can see the MatNTM learns the Repeat Copy scheme.


3. **Associative Recall** :

    a) Cross Entropy Loss, Error reported in Bits Error Per Sequence
    
    b) num_items from Unif(2,15)
    
    c) Each item is two sequence long
    
    d) Each sequence is a 5x5 matrix.
    
![Experimental Associative Recall Learning Curve](https://github.com/sydney-machine-learning/Matrix_NeuralTuringMachine/blob/master/Experiments'%20LC/MatAssociativeRecall_SeqErr_first_train.png "Experimental Associative Recall Learning Curve")


**NOTE :** There is a massive scope of improvement and further modifications in terms of hyperparameter tuning because, as visible in MatrixNTM.py, MatNTM contains several free hyperparameters due to inclusion of new axis in terms of representation dimensions. The above experiments were run only on an *educated guess* of these hyperparameter values, and yet it performs better in these tasks compared to a vanilla NTM. Also, these high amount of perturbations and unstable learning dynamics is expected due to the quadratic natre of left and right parameter matrix multplication in the proposed Matrix FNN and RNNs.

