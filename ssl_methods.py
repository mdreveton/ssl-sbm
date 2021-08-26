#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  4 14:22:26 2021

@author: mdreveto
"""



import numpy as np 
import networkx as nx
import scipy as sp

from tqdm import tqdm



def conjgradLinearEquationSolver( K, b, x=None, T=1e4, tol=1e-10, print_ = False):
    """
    K = n-by-n numpy array matrix or scipy sparse matrix
    b = n-by-m RHS numpy ndarray ( not sparse )
    x = n-by-m initial condition
    T = maximum number of iterations
    Outputs solution to Kx = b
    Using conjugate gradient method
    """
    if x is None:
        x = np.zeros_like( b )

    r = b - K@x
    p = r
    rsold = np.sum(r**2,axis=0)
  
    err = 1 
    i = 0
    while (err > tol) and (i < T):
        i += 1
        Kp = K@p
        a = rsold / np.sum( p*Kp, axis=0 )
        x += a * p
        r -= a * Kp
        rsnew = np.sum( r**2, axis=0 )
        err = np.sqrt(np.sum(rsnew)) 
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    if (print_):
        print( i, err )
    return x





def degree_sampling( G, number_labeled_nodes ):
    degrees = [ deg[1] for deg in G.degree() ]
    index_highest_degrees = np.argsort( degrees )
    return index_highest_degrees[ [ -i for i in range( number_labeled_nodes ) ] ]


def uniform_sampling( G, number_labeled_nodes ):
    #nodes = [ int(node) for node in list( G.nodes() ) ]
    #return np.random.choice( range( len(nodes) ), number_labeled_nodes )
    return np.random.choice( range( nx.number_of_nodes( G ) ), number_labeled_nodes, replace = False )


def per_class_sampling( labels_true, number_labeled_nodes_per_class ):
    labeled_nodes = [ ]
    for k in set( labels_true ):
        cluster = [ i for i in range( len(labels_true ) ) if labels_true[i] == k ]
        chosen_nodes = np.random.choice( cluster, number_labeled_nodes_per_class ) 
        for i in chosen_nodes:
            labeled_nodes.append( i )
    
    return labeled_nodes


def oracle( G, labels_true, number_labeled_nodes, misclassification_rate , sampling_method = 'uniform' ):
    """
    labels_true : true community labels (1 or 2) in a list/numpy array format (size N times 1)
    misclassification_rate : oracle rate of mistake  
    
    return oracle_prediction, a n times 1 vector, whose entries (ik) is \pm 1 is node i is labeled by the oracle, and 0 otherwise.
    """
    
    labels_true = list( labels_true )
    n = len( labels_true )
    oracle_prediction = np.zeros( n )
    
    if sampling_method == 'uniform':
        labeled_nodes = uniform_sampling( G, number_labeled_nodes )
    elif sampling_method == 'degree':
        labeled_nodes = degree_sampling( G, number_labeled_nodes )
    elif sampling_method == 'same_number_per_class':
        labeled_nodes = per_class_sampling( labels_true, number_labeled_nodes // len( set( labels_true ) ) )
    else:
        raise TypeError( 'Sampling method not implemented' )
    
    for i in labeled_nodes:
        if labels_true[i] == 1:
            oracle_prediction[i] = 1
        else:
            oracle_prediction[i] = -1
    mislabeled_nodes = np.random.choice( labeled_nodes, int( misclassification_rate * number_labeled_nodes ) , replace = False )
    for i in mislabeled_nodes:
        if labels_true[i] == 1:
            oracle_prediction[i] = -1
        else:
            oracle_prediction[i] = 1
    """
    for i in labeled_nodes:
        random_number = np.random.rand()
        if random_number >= misclassification_rate: #In that case the oracle predicts correctly
            if labels_true[i] == 1:
                oracle_prediction[i] = 1
            else:
                oracle_prediction[i] = -1
        else: #In that case the oracle makes a mistake
            if labels_true[i] == 1:
                oracle_prediction[i] = -1
            else:
                oracle_prediction[i] = 1
    """
    return oracle_prediction


def getLabeledNodes( oracle_prediction ):
    return [ i for i in range(len( oracle_prediction ) ) if oracle_prediction[i] != 0 ]

def getUnlabeledNodes( oracle_prediction ):
    return [i for i in range( len( oracle_prediction ) ) if oracle_prediction[i] == 0 ]


def getCorrectlyLabeledNodes( oracle_prediction, labels_true ):
    return [i for i in getLabeledNodes( oracle_prediction ) if oracle_prediction[i] == labels_true[i] ]

def getMislabeledNodes( oracle_prediction, labels_true ):
    return [i for i in getLabeledNodes( oracle_prediction ) if oracle_prediction[i] != labels_true[i] ]


def onehot_labels( oracle_prediction, K = 2 ):
    labeled_nodes_indexes = getLabeledNodes( oracle_prediction )
    n = len( oracle_prediction )
    S = np.zeros( ( n, K ) )
    for i in labeled_nodes_indexes:
        if oracle_prediction[i] == 1:
            S[ i, 0 ] = 1
        else:
            S[ i, 1 ] = 1
        
    return S


# =============================================================================
# Algorithms used for comparison
# =============================================================================

def makeGeneralizedAdjacencyMatrix( adjacencyMatrix, sigma = 1/2 ):
    """
    Return the generalized  normalized adjacency matrix: so D^(-sigma) * A * D^(sigma-1)
    """
    n = adjacencyMatrix.shape[0]
    D = sp.sparse.csr_matrix.sum(adjacencyMatrix, axis=0)

    
    D1 =sp.sparse.lil_matrix( ( n, n ) ) #Will correspond to D^{-sigma}
    D1_vector = ( ( np.power( D, - float( sigma ) ) ) .A1 )
    for i in range(n):
        D1[i,i] = D1_vector[i]
    D1 = sp.sparse.dia_matrix( D1 )
    
    D2 =sp.sparse.lil_matrix( ( n, n ) ) #will correspond to D^{sigma-1}
    D2_vector = ( ( np.power( D, float( sigma - 1 ) ) ) .A1 )
    for i in range(n):
        D2[i,i] = D2_vector[i]
    D2 = sp.sparse.dia_matrix( D2 )

    return D1 @ adjacencyMatrix @ D2



def labelSpreading( G, oracle_prediction, alpha = 0.9 ):

    n = nx.number_of_nodes( G )
    Anormalized = makeGeneralizedAdjacencyMatrix( nx.adjacency_matrix(G), sigma = 1/2 )
    
    drift = (1-alpha) * oracle_prediction
    K = np.eye(n) - alpha * Anormalized.todense()
    X = np.linalg.solve( K, drift)
    
    labels_pred = [ ( X[i] < 0 )*1 +1 for i in range( n ) ]
    
    return np.asarray( labels_pred )




def ssl_gaussian_fields( G, oracle_prediction ):
    """
    Algorithm from the paper
    Zhu, X., Ghahramani, Z., & Lafferty, J. D. (2003).
    Semi-supervised learning using gaussian fields and harmonic functions.
    In Proceedings of the 20th International conference on Machine learning (ICML-03) (pp. 912-919). 
    """
    n = nx.number_of_nodes( G )
    A = nx.adjacency_matrix( G ).todense( )
    L = nx.laplacian_matrix( G ).todense( )
    labeled_nodes = getLabeledNodes( oracle_prediction )
    unlabeled_nodes = getUnlabeledNodes( oracle_prediction )
    
    X = np.zeros( n )
    X[labeled_nodes] = oracle_prediction[ labeled_nodes ]
    X = np.reshape(X, (n,1))
    
    Luu = L[ unlabeled_nodes, : ]
    Luu = Luu[ :, unlabeled_nodes ]
    
    Aul= A[ :, labeled_nodes ]
    Aul = Aul[ unlabeled_nodes, : ]
    
    drift = Aul @ X[ labeled_nodes ]
    K = Luu
    
    X[ unlabeled_nodes ] = np.linalg.solve( K, drift )
    
    X = np.reshape( X, (n) )
    labels_pred = [ ( X[i] < 0 )*1 +1 for i in range( n ) ]
    
    return np.asarray( labels_pred )


def poissonLearning( G, oracle_prediction ):
    labeled_nodes = getLabeledNodes( oracle_prediction )
    S = onehot_labels( oracle_prediction, K = 2 )
    L = nx.laplacian_matrix( G )
    Sbar = np.zeros( S.shape )
    
    for i in labeled_nodes:
        for k in range( S.shape[ 1 ] ):
            Sbar[ i, k ] = np.mean( S[ labeled_nodes, k ] )
            
    Stilde = S - Sbar
    X = np.zeros( S.shape )
    #X = np.linalg.solve( L.todense( ), Stilde  )
    for k in range( S.shape[ 1 ] ):
        X[:,k] = conjgradLinearEquationSolver( L, Stilde[:,k] )

    #X = conjgradLinearEquationSolver( L, Stilde )
    for k in range( S.shape[ 1 ] ):
        X[:,k] = X[:,k] / np.mean( S[ labeled_nodes, k ] )
    labels_pred = [ np.argmax( X[i,:] ) + 1 for i in range( nx.number_of_nodes(G) ) ]

    return np.asarray( labels_pred )



def ssl_centered_adjacency_2_clusters( G, oracle_prediction, alpha, Acentered = 0 ):
    """
    SSL method proposed in Couillet, Mai 2020
    """
    
    n = nx.number_of_nodes( G )
    if not isinstance( Acentered, np.matrix):
        #print('Acentered not given')
        A = nx.adjacency_matrix( G ).todense()
        P = np.eye(n) - 1/n * np.ones( (n,n) )    
        Acentered = P @ A @ P
    
    labeled_nodes = getLabeledNodes( oracle_prediction )
    unlabeled_nodes = getUnlabeledNodes( oracle_prediction )
    
    X = np.zeros( n )
    X[labeled_nodes] = oracle_prediction[ labeled_nodes ]
    X = np.reshape( X, (n,1) )
    
    Wuu = Acentered[ unlabeled_nodes, : ]
    Wuu = Wuu[ :, unlabeled_nodes ]
    
    Wul= Acentered[ :, labeled_nodes ]
    Wul = Wul[ unlabeled_nodes, : ]
    
    drift = Wul @ X[ labeled_nodes ]
    
    K = alpha * np.eye( np.shape(Wuu)[0] ) - Wuu
    
    X[ unlabeled_nodes ] = np.linalg.solve( K, drift )
    
    X = np.reshape( X, (n) )
    labels_pred = [ ( X[i] < 0 )*1 +1 for i in range( n ) ]
    
    return np.asarray( labels_pred )




# =============================================================================
# Algo of the paper
# =============================================================================


def secular_equation( alpha, d, delta ):
    liste = [ ( d[i] / (alpha - delta[i]) ) **2 for i in range( len( d ) ) ]
    return sum( liste )

def secular_equation_iterative_step( alpha , d, delta, s ):
    return alpha - 2 * ( f(alpha, d, delta, s) + s**2) / fprime( alpha, d, delta, s ) * ( np.sqrt( f(alpha, d, delta, s) + s**2 ) / s - 1 )


def f(alpha, d, delta, s):
    non_zeros_d = np.argwhere( d != 0 )
    return sum( [  ( d[i] / (delta[i] - alpha) )**2 for i in non_zeros_d ] ) [0] -s**2
    
def fprime(alpha, d, delta, s):
    non_zeros_d = np.argwhere( d != 0 )
    return sum( [  2 * d[i] **2 / ( delta[i] - alpha )**3 for i in non_zeros_d ] ) [ 0 ]


def secular_equation_solver( d, delta , s ):
    """
    Solve the secular equation.
    Algo taken from W. Gander, G. H. Golub, and U. Von Matt, A constrained eigenvalue problem.
    d refers to b.
    """
    
    k = np.argwhere( d != 0 ) [0] [0]
    alpha_current = delta[k] - np.abs( d[k] ) / s
    alpha_next = secular_equation_iterative_step( alpha_current, d, delta, s )
    count = 0
    while ( alpha_next < alpha_current and count < 1000 ) :
        count += 1
        alpha_current = alpha_next
        alpha_next = secular_equation_iterative_step( alpha_current , d, delta, s ) 
    
    #print('Number iterations : ', count)
    return alpha_next


def ssl_via_adjacency_optimal( G, oracle_prediction, lambdaa, tau = 0 ):
    """
    Algo 3.1 of the paper (with the standard constraint)
    """
    n = nx.number_of_nodes( G )
    A =  nx.adjacency_matrix(G).todense( )
    A = np.array( A )
    index_labeled_nodes = getLabeledNodes( oracle_prediction )
    
    Pl = np.zeros( (n,n) )
    for labeled_node in index_labeled_nodes:
        Pl[labeled_node, labeled_node] = 1
    
    b = lambdaa * oracle_prediction    
    C = - A + tau * np.ones( (n,n) ) + lambdaa * Pl
    
    vals_C, vecs_C = np.linalg.eigh( C )
    delta = vals_C
    d = np.transpose(vecs_C) @ b
    
    if ( len( np.argwhere( d != 0 ) ) != 0 ):
        alpha_optimal = secular_equation_solver( d, delta, np.sqrt( n ) )
        M = C - alpha_optimal * np.eye( n )
        X = np.linalg.solve( M, b )
    else:
        X = vecs_C[::,0]
    
    labels_pred = [ ( X[i] < 0 )*1 + 1 for i in range( n ) ]
    return np.asarray( labels_pred )



def ssl_via_adjacency_not_optimal( G, oracle_prediction, lambdaa, tau = 0 , alpha = 0 ):
    n = nx.number_of_nodes( G )
    A =  nx.adjacency_matrix(G).todense( )
    A = np.array( A )
    index_labeled_nodes = getLabeledNodes( oracle_prediction )
    
    Pl = np.zeros( (n,n) )
    for labeled_node in index_labeled_nodes:
        Pl[labeled_node, labeled_node] = 1
    
    b = lambdaa * oracle_prediction
    C = - A + tau * np.ones( (n,n) ) + lambdaa * Pl
    
    if ( len( np.argwhere( b != 0 ) ) != 0 ):
        if alpha == 0:
            vals, vecs = sp.sparse.linalg.eigsh( nx.adjacency_matrix(G).asfptype(), which='LA', k=2)
            alpha = - vals[-2] * 1.1
        M = C - alpha * np.eye( n )
        X = np.linalg.solve( M, b )
    else:
        vals_C, vecs_C = sp.sparse.linalg.eigsh( C, which='SA', k=1 ) #Verify if SA correct
        X = vecs_C[::,0]
    
    labels_pred = [ ( X[i] < 0 )*1 + 1 for i in range( n ) ]
    return np.asarray( labels_pred )





def ssl_via_normalized_laplacian_optimal( G, oracle_prediction, lambdaa, tau = 0 ):
    """
    Algo 3.1 of the paper, with the degree-normalized constraint.
    """
    n = nx.number_of_nodes( G )
    A =  nx.adjacency_matrix(G)
    index_labeled_nodes = getLabeledNodes( oracle_prediction )
    
    Pl = np.zeros( (n,n) )
    for labeled_node in index_labeled_nodes:
        Pl[labeled_node, labeled_node] = 1
    
    b = lambdaa * oracle_prediction    
    L = np.eye( n ) - makeGeneralizedAdjacencyMatrix( A )
    
    D = sp.sparse.csr_matrix.sum( A, axis=0 )
    D1 =sp.sparse.lil_matrix( ( n, n ) ) #Will correspond to D^{-1/2}
    D1_vector = ( ( np.power( D, - float( 1/2 ) ) ) .A1 )
    for i in range(n):
        D1[i,i] = D1_vector[i]
    D1 = sp.sparse.dia_matrix( D1 )

    C = L + tau * D1 @ np.ones( (n,n) ) @ D1 + lambdaa * Pl
    C = L + tau * np.ones( (n,n) ) + lambdaa * Pl
    C = np.array( C )

    vals_C, vecs_C = np.linalg.eigh( C )
    delta = vals_C
    d = np.transpose(vecs_C) @ b
    
    if ( len( np.argwhere( d != 0 ) ) != 0 ):
        alpha_optimal = secular_equation_solver( d, delta, np.sqrt( n ) )
        M = C - alpha_optimal * np.eye( n )
        X = np.linalg.solve( M, b )
    else:
        X = vecs_C[::,0]
    
    labels_pred = [ ( X[i] < 0 )*1 + 1 for i in range( n ) ]
    return np.asarray( labels_pred )
