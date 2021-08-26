#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  4 14:24:13 2021

@author: mdreveto
"""

import numpy as np 
import networkx as nx
import scipy as sp
import matplotlib.pyplot as plt


from sklearn.metrics import accuracy_score, f1_score
from sklearn.cluster import SpectralClustering

from tqdm import tqdm

import ssl_methods as ssl
import dcsbm_generator as dcsbm
import mnist as mnist

"""
import os
working_directory_path = os.getcwd() # Check current directory's path
os.chdir('/user/mdreveto/home/Documents/Inria/Simulations/') 
import real_networks as real_networks
"""


SIZE_TITLE = 24
SIZE_LABELS = 24
SIZE_TICKS = 18
SIZE_LEGEND = 18


"""

# =============================================================================
# Comparison normalizations
# =============================================================================

n = 500
pin_range = [ 0.035, 0.04, 0.045, 0.05 ]
pout = 0.03
number_labeled_nodes = 50
misclassification_rate = 0.1
nAverage = 50

( cost_mean, cost_ste ) = compare_relaxations( n, pin_range, pout, dcsbm_theta_generation = 'sbm',
                        number_labeled_nodes = number_labeled_nodes, misclassification_rate = misclassification_rate,
                        nAverage = nAverage )




methodsToCompare = list( cost_mean.keys() )
for method in methodsToCompare:
    plt.errorbar( pin_range, cost_mean[method], yerr = cost_ste[method], linestyle = '-.', label= method )
    
legend = plt.legend( loc=0,  fancybox=True, fontsize= SIZE_LEGEND-4)
plt.setp( legend.get_title(),fontsize= SIZE_LEGEND-4 )

plt.xlabel( 'pin', fontsize = SIZE_LABELS)
plt.ylabel( 'Cost', fontsize = SIZE_LABELS)
plt.xticks( pin_range, fontsize = SIZE_TICKS )
plt.yticks( fontsize = SIZE_TICKS )

if(savefig):
    fileName = 'comparison_relaxation_SBM_n_' + str(n) + '_pout_' + str(pout) + '_n_labeled_' + str(number_labeled_nodes) + '_misclassification_' + str(misclassification_rate) + '_nAverage_' + str(nAverage) + '.eps'
    plt.savefig( fileName, format='eps', bbox_inches='tight' )
plt.show( )



# =============================================================================
# Synthetic networks: Figure 2
# =============================================================================


n = 1500
pin = 0.02
pout = 0.01
dcsbm_theta_generation = 'pareto' #Other choices: sbm, normal
number_labeled_nodes = 50
sampling_method = 'uniform'
methodsToCompare = [ 'Algorithm 3.1', 'Centered Kernel', 'Poisson learning' ]
misclassification_rate_range = [ 0, 0.1, 0.2, 0.3, 0.4 ]
nAverage = 50

( accuracies_mean, accuracies_ste ) = synthticNetworks_varyingMisclassificationRate( n, pin, pout, 
                dcsbm_theta_generation = dcsbm_theta_generation,
                misclassification_rate_range = misclassification_rate_range, 
                number_labeled_nodes = number_labeled_nodes,
                methodsToCompare = methodsToCompare, 
                nAverage = nAverage )

fileName = 'accuracy_noisy_oracle_DCSBM_theta_' + str(dcsbm_theta_generation)  + '_n_' + str(n) + '_pin_' + str(pin) + '_pout_' + str(pout) + '_nAverage_' + str(nAverage) + '.eps'
titleFig = 'Accuracy on DC-SBM, theta = ' + str(dcsbm_theta_generation) + ' \n with n = ' + str(n) + 'pin = ' + str(pin) + ', pout = ' + str(pout)
plotVaryingMisclassificationRate(accuracies_mean, accuracies_ste, misclassification_rate_range = misclassification_rate_range, savefig = False, fileName = fileName, titleFig = titleFig )




# =============================================================================
# Synthetic networks: boxplots (Figure not included in the paper)
# =============================================================================

n = 1500
pin = 0.02
pout = 0.01
dcsbm_theta_generation = 'pareto' #Other choices: sbm, normal
number_labeled_nodes = 50
sampling_method = 'uniform'
method = 'Algorithm 3.1'
misclassification_rate = 0.3
nAverage = 5

data = boxplotSyntheticNetworks( n, pin, pout, 
                dcsbm_theta_generation = dcsbm_theta_generation,
                misclassification_rate = misclassification_rate, 
                number_labeled_nodes = number_labeled_nodes,
                method = method, 
                nAverage = nAverage )


# =============================================================================
# MNIST: accuracy for chosen digits pairs and different algorithms (Figure 3)
# =============================================================================

dataset = 'mnist'
digits_selected = [2, 4]
n = 1000
W, labels_true = mnist.build_knn_graph( dataset, number_nearest_neighbors = 8, digits_selected = digits_selected , n = n )
G = nx.from_scipy_sparse_matrix( W )
#G, labels_true = mnist.preprocessing( digits_selected = digits_selected, n = 1000, n_neighbors = 8 )

number_labeled_nodes = 10
nAverage = 10
methodsToCompare = [ 'Algorithm 3.1', 'Centered Kernel', 'Poisson learning' ]
misclassification_rate_range = [0, 0.1, 0.2, 0.3, 0.4]

( accuracies_mean, accuracies_ste ) = realGraph_varyingMisclassificationRate( G, labels_true, misclassification_rate_range = misclassification_rate_range, 
                number_labeled_nodes = number_labeled_nodes,
                methodsToCompare = methodsToCompare, 
                nAverage = nAverage )

fileName = 'mnist_' + str(digits_selected)  + '_n_labeled_nodes_' + str(number_labeled_nodes) + '_nAverage_' + str( nAverage ) + '.eps'
titleFig = 'Different accuracies on MNIST digits = ' + str(digits_selected) + '\n with ' + str(number_labeled_nodes) + 'labeled nodes'
plotVaryingMisclassificationRate(accuracies_mean, accuracies_ste, misclassification_rate_range = misclassification_rate_range, savefig = False, fileName = fileName, titleFig = titleFig )


# =============================================================================
# MNIST: boxplot (Figure 4)
# =============================================================================

dataset = 'mnist'
digits_selected = [ 2, 4 ]
interaction_type = 'KNN'
n_neighbors = 8
W, labels_true = mnist.build_knn_graph( dataset, number_nearest_neighbors = n_neighbors, digits_selected = digits_selected , n = 1000 )
G = nx.from_scipy_sparse_matrix( W )
method = 'Poisson learning'
nAverage = 100
number_labeled_nodes = 40
misclassification_rate = 0.4

fileName = 'mnist_boxplot_different_accuracies_digits_' + str(digits_selected)  + '_n_labeled_nodes_' + str(number_labeled_nodes) + '_noise_' + str(misclassification_rate) + '_nAverage_' + str( nAverage ) + '_method_' + str(method) + '.eps'
titleFig = 'Different accuracies on MNIST digits = ' + str(digits_selected) + '\n with ' + str(number_labeled_nodes) + 'labeled nodes and error rate = ' + str(misclassification_rate) + ' using ' + str(method)

data = boxplotRealGraphs( G, labels_true, digits_selected,
                number_labeled_nodes = number_labeled_nodes, misclassification_rate = misclassification_rate,
                method = method, sampling_method = 'uniform', 
                nAverage = nAverage, savefig = True, fileName = fileName, titleFig = titleFig )

"""


def performance_metric( labels_true, labels_pred, metric = 'accuracy', labelsPermutation = False ):
    if ( labelsPermutation == True ):
        labels_pred_permuted = np.zeros( len( labels_pred ), dtype = int )
        for i in range( len(labels_pred) ):
            if labels_pred[i] == 1:
                labels_pred_permuted[i] = 2
            else:
                labels_pred_permuted[i] = 1
    else:
        labels_pred_permuted = labels_pred
        
    if metric == 'accuracy':
        return max( accuracy_score( labels_true, labels_pred ), accuracy_score( labels_true, labels_pred_permuted ) )
    elif metric == 'f1_score':     
        return max( f1_score( labels_true, labels_pred ), f1_score( labels_true, labels_pred_permuted ) )
    else:
        raise TypeError( "Performance metric not implemented" )


def ssl_methods( G, method, oracle_prediction,  
                alpha_adjacency = 0, tau_adjacency = 0,
                pin = 0.5, pout = 0.2, theta_dcsbm = 0,
                lambdaa = 1,
                alpha_labelSpreading = 0.9,
                sigma1 = 1, sigma2 = 1,
                Acentered = 0 ):
    
    if method == 'Adjacency optimal' or method == 'Algorithm 3.1' or method == 'Adjacency optimal degree-normalized':
        #return ssl.ssl_via_adjacency_optimal( G, oracle_prediction, lambdaa, tau = tau_adjacency )
        return ssl.ssl_via_normalized_laplacian_optimal( G, oracle_prediction, lambdaa, tau = tau_adjacency )
    
    if method == 'Adjacency not optimal':
        return ssl.ssl_via_adjacency_not_optimal( G, oracle_prediction, lambdaa, tau = tau_adjacency , alpha = - alpha_adjacency )
    
    if method == 'Adjacency optimal standard normalization':
        return ssl.ssl_via_adjacency_optimal( G, oracle_prediction, lambdaa, tau = tau_adjacency )
        
    elif method == 'Centered Kernel':
        return ssl.ssl_centered_adjacency_2_clusters( G, oracle_prediction, alpha_adjacency, Acentered = Acentered )
    
    elif method == 'Label Spreading':
        return ssl.labelSpreading( G, oracle_prediction, alpha=alpha_labelSpreading )
    
    elif method == 'Label Propagation':
        return ssl.ssl_gaussian_fields( G, oracle_prediction )
    
    elif method == 'Modularity':
        return ssl.ssl_via_modularity( G, oracle_prediction, lambdaa )
    
    elif method == 'Poisson learning':
        return ssl.poissonLearning( G, oracle_prediction )
    
    else:
        raise TypeError( "Method not implemented" )


def spectralClustering( G , K = 2 ):
    adjacency_matrix = nx.adjacency_matrix( G ).todense( )
    sc = SpectralClustering( n_clusters = K, affinity = 'precomputed', assign_labels = 'discretize' )
    labels_pred = sc.fit_predict( adjacency_matrix ) + np.ones( nx.number_of_nodes( G ) )
    return labels_pred.astype(int)



def compare_relaxations( n, pin_range, pout, dcsbm_theta_generation = 'sbm',
                        number_labeled_nodes = 50, misclassification_rate = 0.1,
                        nAverage = 5 ):
    
    cost_mean = dict( )
    cost_ste = dict( )
    methodsToCompare = [ 'standard', 'normalized' ]
    for method in methodsToCompare:
        cost_mean[method] = [ ]
        cost_ste[method] = [ ]

    for i in tqdm( range( len( pin_range ) ) ):
        pin = pin_range[ i ]
        probs = (pin - pout) * np.eye( 2 ) + pout * np.ones( (2,2) )
        
        cost_normalized = [ ]
        cost_standard = [ ]
        
        for trial in range( nAverage ):
            sizes =  list( np.random.multinomial( n, 1/2 * np.ones( 2 ) ) )
            theta = dcsbm.generateThetaDCSBM( n, law = dcsbm_theta_generation )
            G = dcsbm.generate_DCSBM( sizes, probs, theta )            
            labels_true = dcsbm.getCommunityLabels( G )
                
            vals, vecs = sp.sparse.linalg.eigsh( nx.adjacency_matrix( G ).asfptype(), which='LA', k=2 )
            sigma2 = vals[ 0 ]
            sigma1 = vals[ 1 ]
            tau_adjacency = 4 * ( vals[ 1 ] + vals[ 0 ] ) / n
            #tau_adjacency = 2 * nx.number_of_edges( G ) / n
            
            eta = number_labeled_nodes / n
            eta0 = misclassification_rate * eta
            eta1 = (1-misclassification_rate) * eta
            if eta0 != 0 :
                lambdaa = np.log( eta1 / eta0 )
            else:
                lambdaa = np.log( n * eta1 )
            lambdaa = lambdaa / np.log( (sigma1 + sigma2 ) / (sigma1 - sigma2) )
            
            s = ssl.oracle( G, labels_true, number_labeled_nodes, misclassification_rate, sampling_method = 'same_number_per_class' )
    
            labels_pred_standard = ssl.ssl_via_adjacency_optimal( G, s, lambdaa, tau = tau_adjacency )
            labels_pred_normalized = ssl.ssl_via_normalized_laplacian_optimal( G, s, lambdaa, tau = tau_adjacency )
            
            cost_standard.append( cost( nx.adjacency_matrix( G ).todense(), s, labels_pred_standard, tau = tau_adjacency, lambdaa= lambdaa ) )
            cost_normalized.append( cost( nx.adjacency_matrix( G ).todense(), s, labels_pred_normalized, tau = tau_adjacency, lambdaa= lambdaa ) )
        
        cost_mean['standard'].append( np.mean( cost_standard ) )
        cost_ste['standard'].append( np.std( cost_standard ) / np.sqrt(nAverage) )
        cost_mean['normalized'].append( np.mean( cost_normalized ) )
        cost_ste['normalized'].append( np.std( cost_normalized ) / np.sqrt(nAverage) )

    return ( cost_mean, cost_ste )


def cost( A, oracle_prediction, labels_pred, tau = 0, lambdaa = 0 ):
    x = [ (label ==1 )*(1) + ( label == 2 )*(-1) for label in labels_pred ]
    x = np.asarray( x )
    n = len( labels_pred )
    Iell = np.zeros( ( n, n ) )
    for i in range( n ):
        if oracle_prediction[i] != 0:
            Iell[i,i] = 1
    
    return - x.T @ ( A - tau * np.ones( (n,n) ) ) @ x + lambdaa * ( oracle_prediction - Iell@x ).T @ ( oracle_prediction - Iell@x )




def compare_ssl_methods( G, labels_true, number_labeled_nodes, misclassification_rate, methodsToCompare = ['Adjacency optimal', 'Adjacency Centered' ], 
                        lambdaa = 1, sigma1 = 1, sigma2 = 1, tau_adjacency = 0.1,
                        nAverage = 10,
                        Acentered = 0, alpha_adja = 0):
    
    accuracy = dict( )
    for method in methodsToCompare:
        accuracy[method] = [ ]
        
    for trial in range( nAverage ):
        for method in methodsToCompare:
            if method == 'Spectral Clustering':
                labels_pred = spectralClustering( G )
                accuracy[ method ].append( performance_metric( labels_true, labels_pred, metric = 'accuracy', labelsPermutation = True ) )
            else:
                s = ssl.oracle( G, labels_true, number_labeled_nodes, misclassification_rate, sampling_method = 'same_number_per_class' )
                labels_pred = ssl_methods( G, method, s, lambdaa = lambdaa, tau_adjacency=tau_adjacency, alpha_adjacency = sigma2, Acentered= Acentered )
                unlabeled_nodes = ssl.getUnlabeledNodes( s )
                accuracy[ method ].append( performance_metric( labels_true[ unlabeled_nodes ], labels_pred[ unlabeled_nodes ], metric = 'accuracy', labelsPermutation = False ) )
    
    accuracy_mean, accuracy_ste = dict( ), dict( )
    for method in methodsToCompare:
        accuracy_mean[method] = np.mean( accuracy[method] )
        accuracy_ste[method] = np.std( accuracy[method] ) / np.sqrt( nAverage )
        
    return (accuracy_mean, accuracy_ste)


def plotVaryingMisclassificationRate( accuracies_mean, accuracies_ste, misclassification_rate_range = [0, 0.2, 0.4, 0.5], 
                                         savefig = False, fileName = 'fig.eps', titleFig = 'Fig' ):
    
    methodsToCompare = list( accuracies_mean.keys() )
    for method in methodsToCompare:
        plt.errorbar( misclassification_rate_range, accuracies_mean[method], yerr = accuracies_ste[method], linestyle = '-.', label= method )
    
    legend = plt.legend( loc=0,  fancybox=True, fontsize= SIZE_LEGEND-4)
    plt.setp(legend.get_title(),fontsize= SIZE_LEGEND-4)

    plt.xlabel("Oracle-misclassification ratio", fontsize = SIZE_LABELS)
    plt.ylabel( 'Accuracy', fontsize = SIZE_LABELS)
    
    plt.xticks( misclassification_rate_range, fontsize = SIZE_TICKS )
    plt.yticks( fontsize = SIZE_TICKS )
    
    if(savefig):
        plt.savefig( fileName, format='eps', bbox_inches='tight' )
    else:
        plt.title( titleFig )
    plt.show( )
    
    return 0


def realGraph_varyingMisclassificationRate( G, labels_true, misclassification_rate_range = [0, 0.2, 0.4, 0.5], 
                number_labeled_nodes = 50,
                methodsToCompare = [ 'Label Spreading', 'Label Propagation', 'Adjacency centered', 'Adjacency optimal' ], 
                nAverage = 5 ):
    
    n = nx.number_of_nodes( G )
    
    if 'Adjacency centered' in methodsToCompare:
        P = np.eye( n ) - 1/n * np.ones( (n,n) )
        Acentered = P @ nx.adjacency_matrix( G ) @ P
    else:
        Acentered = 0
    
    if 'Adjacency optimal' in methodsToCompare or 'Algorithm 3.1' in methodsToCompare or 'Modularity' in methodsToCompare:
        vals, vecs = sp.sparse.linalg.eigsh( nx.adjacency_matrix(G).asfptype(), which='LA', k=2)
        sigma2 = vals[0]
        sigma1 = vals[1]
        tau_adjacency = 4 * ( sigma1 + sigma2 ) / n
    
    accuracies_mean = dict( )
    accuracies_ste = dict( )
    for method in methodsToCompare:
        accuracies_mean[method] = []
        accuracies_ste[method] = []

    for i in tqdm( range( len( misclassification_rate_range ) ) ):
        misclassification_rate = misclassification_rate_range[ i ]
        if 'Adjacency optimal' in methodsToCompare or 'Algorithm 3.1' in methodsToCompare:
            eta = number_labeled_nodes / n
            eta0 = misclassification_rate * eta
            eta1 = (1-misclassification_rate) * eta
            if eta0 != 0:
                lambdaa = np.log( eta1 / eta0 )
            else:
                lambdaa = np.log( n * eta1 )
            lambdaa = lambdaa / ( np.log( (sigma1 + sigma2 ) / (sigma1 - sigma2) ) )
            #print( lambdaa )
        accuracy_mean, accuracy_ste = compare_ssl_methods( G, labels_true, number_labeled_nodes, misclassification_rate, 
                                                          methodsToCompare = methodsToCompare, 
                                                          nAverage=nAverage,
                                                          sigma1 = sigma1, sigma2 = sigma2*1.1,
                                                          Acentered = Acentered, tau_adjacency=tau_adjacency)
        for method in methodsToCompare:
            accuracies_mean[method].append( accuracy_mean[method] )
            accuracies_ste[method].append( accuracy_ste[method] )

    return (accuracies_mean, accuracies_ste)


def synthticNetworks_varyingMisclassificationRate( n, pin, pout, dcsbm_theta_generation = 'sbm',
                                                  misclassification_rate_range = [0, 0.2, 0.4, 0.5], 
                number_labeled_nodes = 50,
                methodsToCompare = [ 'Label Spreading', 'Label Propagation', 'Adjacency centered', 'Adjacency optimal' ], 
                nAverage = 5 ):
    
    probs = (pin - pout) * np.eye( 2 ) + pout * np.ones( (2,2) )

    accuracies_mean = dict( )
    accuracies_ste = dict( )
    for method in methodsToCompare:
        accuracies_mean[method] = [ ]
        accuracies_ste[method] = [ ]

    for i in tqdm( range( len( misclassification_rate_range ) ) ):
        sizes =  list( np.random.multinomial( n, 1/2 * np.ones( 2 ) ) )
        theta = dcsbm.generateThetaDCSBM( n, law = dcsbm_theta_generation )
        G = dcsbm.generate_DCSBM( sizes, probs, theta )            
        labels_true = dcsbm.getCommunityLabels( G )
    
        if 'Adjacency centered' in methodsToCompare:
            P = np.eye( n ) - 1/n * np.ones( ( n, n ) )
            Acentered = P @ nx.adjacency_matrix( G ) @ P
        else:
            Acentered = 0
        
        if 'Adjacency optimal' in methodsToCompare or 'Algorithm 3.1' in methodsToCompare or 'Modularity' in methodsToCompare or 'Adjacency optimal standard normalization' in methodsToCompare or 'Adjacency optimal degree-normalized' in methodsToCompare:
            vals, vecs = sp.sparse.linalg.eigsh( nx.adjacency_matrix( G ).asfptype(), which='LA', k=2 )
            sigma2 = vals[0]
            sigma1 = vals[1]
            tau_adjacency = 4 * ( vals[1] + vals[0] ) / n
            #tau_adjacency = 2 * nx.number_of_edges( G ) / n
            
        misclassification_rate = misclassification_rate_range[ i ]
        eta = number_labeled_nodes / n
        eta0 = misclassification_rate * eta
        eta1 = (1-misclassification_rate) * eta
        if eta0 != 0 :
            lambdaa = np.log( eta1 / eta0 )
        else:
            lambdaa = np.log( n * eta1 )
        lambdaa = lambdaa / np.log( (sigma1 + sigma2 ) / (sigma1 - sigma2) )
        print(lambdaa)

        accuracy_mean, accuracy_ste = compare_ssl_methods( G, labels_true, number_labeled_nodes, misclassification_rate, 
                                                          methodsToCompare = methodsToCompare, 
                                                          nAverage=nAverage,
                                                          sigma1 = sigma1, sigma2 = sigma2,
                                                          Acentered = Acentered, tau_adjacency=tau_adjacency)
        for method in methodsToCompare:
            accuracies_mean[method].append( accuracy_mean[method] )
            accuracies_ste[method].append( accuracy_ste[method] )

    return ( accuracies_mean, accuracies_ste )




def boxplotRealGraphs( G, labels_true, digits_selected,
                number_labeled_nodes = 50, misclassification_rate = 0.3,
                method = 'Adjacency optimal' ,
                sampling_method = 'uniform',
                nAverage = 5, savefig = False, fileName = 'fig.eps', titleFig = 'Fig' ):
    
    accuracy_allNodes = [ ]
    accuracy_unlabeledNodes = [ ]
    accuracy_misclassifiedNodes = [ ]
    accuracy_correctlyClassifiedNodes = [ ]
    
    n = nx.number_of_nodes( G )

    if method == 'Adjacency centered':
        A = nx.adjacency_matrix( G )
        P = np.eye( n ) - 1 / n * np.ones( ( n, n ) )
        Acentered = P @ A @ P
        vals, vecs = sp.sparse.linalg.eigsh( A.asfptype( ), which = 'LA', k = 2 )
        sigma2 = vals[ 0 ]
    else:
        Acentered = 0
        sigma2 = 1
    
    if method == 'Adjacency optimal' or method == 'Algorithm 3.1' or method == 'Modularity':
        vals, vecs = sp.sparse.linalg.eigsh( nx.adjacency_matrix( G ).asfptype(), which='LA', k=2 )
        sigma2 = vals[ 0 ]
        sigma1 = vals[ 1 ]
        tau_adjacency = 4 * ( vals[1] + vals[0] ) / n
        #tau_adjacency = 2 * nx.number_of_edges( G ) / n

        eta = number_labeled_nodes / n
        eta0 = misclassification_rate * eta
        eta1 = (1-misclassification_rate) * eta
        if eta0 != 0 :
            lambdaa = np.log( eta1 / eta0 )
        else:
            lambdaa = np.log( n * eta1 )
            lambdaa = lambdaa / np.log( sigma1 / sigma2 )
    else:
        lambdaa = 1
        tau_adjacency = 1

    for i in tqdm( range( nAverage ) ):
        oracle_prediction = ssl.oracle( G, labels_true, number_labeled_nodes, misclassification_rate , sampling_method = sampling_method )
        unlabeledNodes = ssl.getUnlabeledNodes( oracle_prediction )
        misclassifiedNodes = ssl.getMislabeledNodes( oracle_prediction, labels_true )
        correctlyClassifiedNodes = ssl.getCorrectlyLabeledNodes( oracle_prediction, labels_true )
        
        labels_pred = ssl_methods( G, method, oracle_prediction, lambdaa = lambdaa, tau_adjacency=tau_adjacency, alpha_adjacency = sigma2, Acentered= Acentered )
        labels_pred = np.asarray( labels_pred )
        
        accuracy_allNodes.append( max( accuracy_score( labels_true, labels_pred ), 1-accuracy_score(labels_true, labels_pred) ) )
        accuracy_unlabeledNodes.append( max( accuracy_score( labels_true[unlabeledNodes], labels_pred[unlabeledNodes] ), 1 - accuracy_score( labels_true[unlabeledNodes], labels_pred[unlabeledNodes] ) ) )
        accuracy_misclassifiedNodes.append( max ( accuracy_score( labels_true[misclassifiedNodes], labels_pred[misclassifiedNodes] ) , 1 - accuracy_score( labels_true[misclassifiedNodes], labels_pred[misclassifiedNodes] )  ) )
        accuracy_correctlyClassifiedNodes.append( max( accuracy_score( labels_true[correctlyClassifiedNodes], labels_pred[correctlyClassifiedNodes] ) , 1 - accuracy_score( labels_true[correctlyClassifiedNodes], labels_pred[correctlyClassifiedNodes] ) ) )

    #accuracy_allNodes = np.asarray( accuracy_allNodes )
    accuracy_unlabeledNodes = np.asarray( accuracy_unlabeledNodes )
    accuracy_misclassifiedNodes = np.asarray( accuracy_misclassifiedNodes )
    accuracy_correctlyClassifiedNodes = np.asarray( accuracy_correctlyClassifiedNodes )
    
    #data = np.array( [ np.transpose( accuracy_unlabeledNodes ), np.transpose( accuracy_correctlyClassifiedNodes ), np.transpose( accuracy_misclassifiedNodes ) ] )
    #data = np.concatenate( ( accuracy_unlabeledNodes, accuracy_correctlyClassifiedNodes, accuracy_misclassifiedNodes ) )
    
    data = [ accuracy_unlabeledNodes , accuracy_correctlyClassifiedNodes, accuracy_misclassifiedNodes]

    fig, ax = plt.subplots()
    #ax.set_title('Multiple Samples with Different sizes')
    ax.boxplot(data)
    ax.xaxis.set_ticklabels( ['unlabeled', 'correctly \n labeled', 'wrongly \n labeled'] )

    plt.ylabel( "Accuracy", fontsize = SIZE_LABELS )    
    plt.xticks( fontsize = SIZE_TICKS - 2 )
    plt.yticks( np.linspace( 0.5, 1, num=6 ), fontsize = SIZE_TICKS )
    #plt.yticks( fontsize = SIZE_TICKS  )
    
    if(savefig):
        plt.savefig( fileName, format='eps', bbox_inches='tight' )
    else:
        plt.title( titleFig )
    
    return data

