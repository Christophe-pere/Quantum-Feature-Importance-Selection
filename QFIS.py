#############################
# Author: Christophe Pere   #
# Creation Date: 2023-01-21 #
# Modified: 2023-01-21      #
# Version: 0.01             #
# Project: ML-Hybride       #
# Sub: Fraud Detection      #
#############################


from qiskit.primitives import Sampler
from qiskit.algorithms.state_fidelities import ComputeUncompute
from qiskit.circuit.library import ZZFeatureMap
from qiskit_ibm_runtime import QiskitRuntimeService, Session,  Options, Sampler
from qiskit.utils import algorithm_globals
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from sklearn import metrics
from sklearn.metrics import roc_curve
from qiskit.primitives import Sampler as Sampler_prim

import itertools
from tqdm import tqdm
import numpy as np
import pandas as pd



# -----------------------------------------------------------------------------------
# Quantum Part


def quantum_feature_importance(X: np.array, y: np.array, X_test: np.array, y_test: np.array, backend=None, 
                               options=None, service=None, max_features=None, n_reps=2, env='local'):
    '''
    Function to select the most important features by comparing QSVC performances (metrics)
    
    :param X: Matrix containing the train dataset 
    :type X: numpy array 
    :param y: Array containing the labels for each sample in the train dataset
    :type y: numpy array
    :param X_test: Matrix containing the test dataset 
    :type X_test: numpy array 
    :param y_test: Array containing the labels for each sample in the test dataset
    :type y_test: numpy array
    :param backend: backend to run the QSVC (simulator or real system) 
    :type backend: service or string
    :param options: object option to pass to the sampler (optimisation level, mitigation, number of shots...)
    :type options: object
    :param service: parameters for the backend
    :type service: object
    :param max_features: number of features max 
    :type max_features: None or int 
    :param nb_reps: Number of reps to initialize the feature map
    :type nb_reps: int (default 2)
    :param env: Determine if the algorithm will be run locally or via the IBM cloud
    :type: str 
    
    :return: List of feature ids, KPIs
    :rtype: np.array, np.array
    '''
    # unitary tests - assert .... 
    assert X.shape[1] > 3                       # error if X shape is inferior or equal to 3 
    assert X.shape[0] == y.shape[0]             # error if X and y have not the same number of samples
    assert X_test.shape[0] == y_test.shape[0]   # error if X_test and y_test have not the same number of samples
    assert X.shape[1] == X_test.shape[1]        # error if X and X_test have not the same number of features 
    assert backend is not None                  # error if backend isn't provided 
    assert service is not None                  # error if service isn't provided 
    assert options is not None                  # error if options isn't provided 
    
    # --------------------------------------------------------------------
    # possible metrics: balanced acc, AUC, prec, recall, f1-score -> balanced acc and AUC implemented
    
    # --------------------------------------------------------------------
    # Steps of the feature selection 
    # 1 - initialization -> select the 3 first features of the data 
    # 2 - run QSVC -> with circuit runner or primitives 
    # 3 - evaluation with specific metrics -> put them in parameter? (predict the test to evaluate the model)
    # 4 - record KPI and feature ids 
    # 5 - if there is more feature, select other features in permutation without repetition 
    # 6 - start with the best set of features; add one feature
    # 7 - repeat 2, 3, 4, 5 
    # 8 - compare the KPI with one feature added to the KPI obtained with the previous number of feature
    # 9 - if there is no improvement, stop the algorithm and output the KPI and ids of the best features 
    # 10 - if improvement repeat 6, 7, 8, 9, 10
    
    # --------------------------------------------------------------------
    # variables initialization 
    n_features = X.shape[1] # number of features contained in X 
    nb_features = 3         # start with 3 initial features 
    _auc_ = 0               # initialization of AUC storing variable 
    _bal_acc_ = 0           # initialization of balanced accuracy storing variable 
    results = pd.DataFrame()
    _bal_acc_ref_ = 0
    _auc_ref_ = 0
    # --------------------------------------------------------------------
    # quantum features selection process 
    
    # first step with 3 features permutation
    # permutations possibilities
    
    n_permutations = list(range(n_features))
    permutations = set(itertools.combinations(n_permutations, 3))
    
    ######## Improvements ##########
    # can be paralelized to save 
    # computation time
    ################################
    print(f'\nThe number of permutations is {len(permutations)}\n')
    print(f'Step 1\n')
    
    for combination in tqdm(permutations):      # loop on all the unique permutations  
                
        _X_ = X[:, combination]           # select the corresponding columns in the train dataset 
        _X_test_ = X_test[:, combination] # select the corresponding columns in the test dataset
                
        # run and evaluate the model with the combination of feature
        _bal_acc_, _auc_, qsvm_model  =  run_evaluate_model(_X_, y, _X_test_, y_test, len(combination),n_reps, service, backend, options, _bal_acc_ref_, _auc_ref_, env=env)
                
        # comparison between permutations 
        _bal_acc_ref_, _auc_ref_, _features_, results = eval_quantum_fs(_bal_acc_, _bal_acc_ref_, _auc_, _auc_ref_, combination, results)
        
    # --------------------------------------------------------------------
    # extract best combination 
    best_features = np.array(_features_)  # best features combination
    rest_features = np.delete(n_permutations, _features_) # delete the best combination from the list 
    

    print('Step 2\n')
    
    # --------------------------------------------------------------------
    # number of best features max 
    if not max_features:
        max_features = n_features
    nb_features +=1 
    while nb_features <= max_features:
        
        # steps 2, 3, 4, 5
        ######## Improvements ##########
        # can be paralelized to save 
        # computation time
        ################################ 
        for feature in tqdm(rest_features):  # progress bar
            
            new_combi = np.append(_features_, feature) # add one feature 
            
            _X_ = X[:, new_combi]           # select the corresponding columns in the train dataset 
            _X_test_ = X_test[:, new_combi] # select the corresponding columns in the test dataset
                
            # run and evaluate the model with the combination of feature
            _bal_acc_, _auc_, qsvm_model =  run_evaluate_model(_X_, y, _X_test_, y_test, len(new_combi),n_reps,service, backend, options, _bal_acc_ref_, _auc_ref_, env=env)
            
            # -----------------------------------------  
            # comparison between permutations 
            _bal_acc_ref_, _auc_ref_, _features_, results = eval_quantum_fs(_bal_acc_, _bal_acc_ref_, _auc_, _auc_ref_, new_combi, results)
            
        # --------------------------------------------------------------------
        # Break conditions
          
        # test on the number of features, if no improvement with adding one feature, break and print the result
        if len(_features_)<nb_features:
            print(f'The best combination of features is: {_features_} \
            the QSVC obtained an AUC of: {round(100*_auc_ref_,2)}% and a balanced accuracy of: {round(100*_bal_acc_ref_,2)}%') 
            return _features_, results
        else:    
            best_features = np.array(_features_)  # best features combination
            rest_features = np.delete(n_permutations, _features_) # delete the best combination from the list 
            
        # --------------------------------------------------------------------
        # update the number of features by 1 
        nb_features += 1 # step 6 
    print(f'The best combination of features is: {_features_} \
            the QSVC obtained an AUC of: {round(100*_auc_ref_,2)}% and a balanced accuracy of: {round(100*_bal_acc_ref_,2)}%') 
    return _features_, results



def run_evaluate_model(X, y, X_test, y_test, nb_features=0, n_reps=2, service=None, backend=None, options=None, bal_acc_ref=0., auc_ref=0., env='local', train_only=False):
    '''
    Function to run a QSVC algorithm with Primitives 
    
    :param X: Matrix containing the train dataset 
    :type X: numpy array 
    :param y: Array containing the labels for each sample in the train dataset
    :type y: numpy array
    :param X_test: Matrix containing the test dataset 
    :type X_test: numpy array 
    :param y_test: Array containing the labels for each sample in the test dataset
    :type y_test: numpy array
    :param nb_features: Number of feature to initialize the feature map and number of qubits
    :type nb_features: int (default 0)
    :param nb_reps: Number of reps to initialize the feature map
    :type nb_reps: int (default 2)
    :param service: parameters for the backend
    :type service: object
    :param backend: backend to run the QSVC (simulator or real system) 
    :type backend: service or string
    :param options: object option to pass to the sampler (optimisation level, mitigation, number of shots...)
    :type options: object
    :param bal_acc_ref: previous balanced accuracy
    :type bal_acc_ref: float (default 0.)
    :param auc_ref: previous area under the curve
    :type auc_ref: float (default 0.)
    :param env: Determine if the algorithm will be run locally or via the IBM cloud
    :type: str 
    :param train_only: train only a qsvm and return the model
    :type: bool
    
    :return: List of feature ids, KPIs, qsvc model 
    :rtype: np.array, np.array, qiskit model
    '''
    # -----------------------------------------
    # unitary tests - assert .... 
    assert X.shape[1] >= 3                      # error if X shape is inferior or equal to 3 
    assert X.shape[0] == y.shape[0]             # error if X and y have not the same number of samples
    assert X_test.shape[0] == y_test.shape[0]   # error if X_test and y_test have not the same number of samples
    assert X.shape[1] == X_test.shape[1]        # error if X and X_test have not the same number of features 
    assert nb_features > 0                      # error if the number isn't provided 
    assert backend is not None                  # error if backend isn't provided 
    assert service is not None                  # error if service isn't provided 
    assert options is not None                  # error if options isn't provided 
    
    
    # -----------------------------------------
    # build a session with Primitives for error mitigation 
    with Session(service=service, backend=backend) as session: 

        # definition of the QSVC 
        if env=='local': # run locally
            fidelity = ComputeUncompute(sampler=Sampler_prim())  # This class leverages the sampler primitive to calculate the state fidelity of two quantum circuits following the compute-uncompute method. The fidelity can be defined as the state overlap.
        else: # run on the cloud 
            fidelity = ComputeUncompute(sampler=Sampler(session=session, options = options))
            
        feature_map = ZZFeatureMap(feature_dimension=nb_features, reps=2)                 # feature map 
        new_kernel = FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity)    # quantum kernel 
        adhoc_svc = QSVC(quantum_kernel=new_kernel)                                       # quantum support vector machine with quantum kernel

        # train the QSVC 
        adhoc_svc.fit( X, y)
        
        if train_only:
            return adhoc_svc
        # predict the test in order to be evaluated 
        y_pred = adhoc_svc.predict(X_test)                             # test the model 

        # evaluation metrics 
        bal_acc = metrics.balanced_accuracy_score(y_test, y_pred)      # compute the balanced accuracy
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)   # compute true positive rate, false positive rate, threshold via roc curve
        auc = metrics.auc(fpr, tpr)                                    # compute aera under the curve 

    # -----------------------------------------    
    # if no improvement just return the default values 
    return bal_acc, auc, adhoc_svc

def eval_quantum_fs(_bal_acc_: float, _bal_acc_ref_: float, _auc_:float, _auc_ref_:float, new_combi: tuple, results: pd.DataFrame):
    ''' Function to evaluate the metrics computed for a specific 
    
    :param _bal_acc_: balanced accuracy computed for the permutation
    :type _bal_acc_: float 
    :param _bal_acc_ref_: best balanced accuracy computed for all permutations
    :type _bal_acc_ref_: float
    :param _auc_: area under the curve computed for the permutation
    :type _auc_: float
    :param _auc_ref_: best area under the curve computed for all the permutation
    :type _auc_ref_: float
    :param new_combi: features combination
    :type new_combi: tuple
    :param results: dataframe containing all the performance for all the permutation and combination of features
    :type results: pandas.DataFrame
    
    :return: best balanced accuracy, best area under the curve, associated set of features and results
    :rtype: float, float, set, pandas.DataFrame
    '''
    
    if (_bal_acc_ > _bal_acc_ref_) and (_auc_ > _auc_ref_):              # test between the evaluation and metrics provided
        
        #_model_ = adhoc_svc                                        # save QSVC model
        # save the results if improvement
        print(pd.DataFrame([_bal_acc_, _auc_, new_combi]).T)  # array can be use with np.concatenate
        results = pd.concat([results, pd.DataFrame([_bal_acc_, _auc_, new_combi]).T])
        best = results.sort_values(by=[0, 1], ascending=False)
        best = best.iloc[0, :] # extract best features 
        _bal_acc_ = best[0] # init at best features to only save best improvement
        _auc_ = best[1] 
        _features_ = best[2]
        _bal_acc_ref_ = _bal_acc_                                    # save best balanced accuracy
        _auc_ref_ = _auc_                                            # save best AUC 
                
    return _bal_acc_ref_, _auc_ref_, results.sort_values(by=[0, 1], ascending=False).iloc[0, :][2], results


