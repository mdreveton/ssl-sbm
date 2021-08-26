#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 17:32:49 2021

@author: mdreveto
"""

import numpy as np
import scipy as sp
from tqdm import tqdm
from annoy import AnnoyIndex
import scipy.sparse as sparse

"""
dataset = 'mnist'
M = np.load( dataset + '.npz', allow_pickle = True )
data = M['data']
labels = M['labels']
( W, labels_true ) = build_knn_graph( dataset, number_nearest_neighbors = 8 )


-------


file = '/user/mdreveto/home/Documents/Inria/Simulations/Other_people_code/Jeff_Calder_graph_learning/kNNData/'
M = np.load( file + 'MNIST_vae.npz', allow_pickle = True )
I= M['I']
J = M['J']
D = M['D']
#W = weight_matrix_homogenized( I, J, D, 10, f = negexp )
file = '/user/mdreveto/home/Documents/Inria/Simulations/Other_people_code/Jeff_Calder_graph_learning/Data/'
M = np.load( file + 'MNIST_labels.npz', allow_pickle = True )
labels = M['labels']

np.savez_compressed( 'MNIST_vae_perso', I = I, J = J, D = D, labels = labels )

np.savez_compressed( 'MNIST_vae_perso', weights = W, labels = labels )

"""


def indexSelectedLabels( labels, labels_to_select ):    
    if labels_to_select == 'all':
        return [ i for i in range( len( labels ) ) ]
    else:
        return [ i for i in range( len( labels ) ) if labels[ i ] in labels_to_select ]





def build_knn_graph( dataset, number_nearest_neighbors = 8, digits_selected = 'all' , n = 1000 ):
    
    if dataset.lower() not in ['mnist', 'fashion-mnist', 'mnist_vae']:
        raise TypeError( 'This dataset is not supported' )
    
    elif dataset.lower() == 'mnist_vae':
        M = np.load( dataset + '.npz', allow_pickle = True )
        I= M['I']
        J = M['J']
        D = M['D']
        digits = M[ 'labels' ]        
        digits = digits.reshape( ( 70000, ) )
        digits = digits.astype( int )

        index_selected = indexSelectedLabels( digits, digits_selected )
        I = I[ index_selected, : ]
        J = J[ index_selected, : ]
        D = D[ index_selected, : ]
        digits = digits[ index_selected ]
        labels_true = digits + np.ones( len(digits) )
        return weight_matrix( I, J, D, number_nearest_neighbors, f = negexp ), np.asarray( labels_true, dtype = int )

    else:
        dataFile = dataset + '.npz'
        M = np.load( dataFile, allow_pickle = True )
        data = M[ 'data' ]
        digits = M[ 'labels' ]        
        digits = digits.reshape( ( len( digits ), ) )
        digits = digits.astype( int )
        
        index_selected = indexSelectedLabels( digits, digits_selected )        
        if n < len( index_selected ):
            index_selected = index_selected[ :n ]
        data = data[ index_selected, : ]
        digits = digits[ index_selected ]
        
        I, J, D = knn_with_annoy( data, k = number_nearest_neighbors )
    
    labels_true = list( )
    for i in range( len( index_selected ) ):
        labels_true.append( digits_selected.index( digits[ i ] ) + 1 )
        
        
    return weight_matrix( I, J, D, number_nearest_neighbors, f = negexp ), np.asarray( labels_true, dtype = int )
    



def negexp(x):
    return np.exp(-x)

def knn_with_annoy( data, k = 30, similarity = 'euclidean', fileName = None ):
    """
    Perform approximate nearest neighbor search, returning indices I,J of neighbors, and distance D
    Metric can be "angular", "euclidean", "manhattan", "hamming", or "dot".
    """
    n = data.shape[0]   #Number of points
    dim = data.shape[1] #Dimension

    print('kNN search with Annoy approximate nearest neighbor package...')

    u = AnnoyIndex( dim, similarity )  # Length of item vector that will be indexed
    for i in range( n ):
        u.add_item( i, data[i,:] )

    u.build( 10 ) #10 trees
    
    D = []
    I = []
    J = []
    for i in tqdm( range( n ) ):
        A = u.get_nns_by_item( i, k, include_distances=True, search_k=-1 )
        I.append( [ i ]*k )
        J.append( A[ 0 ] )
        D.append( A[ 1 ] )

    I = np.array( I )
    J = np.array( J )
    D = np.array( D )
    
    if fileName != None:
        np.savez_compressed( fileName , I=I, J=J, D=D )
    
    return I, J, D


def weight_matrix( I, J, D, k, f=negexp, symmetrize = True ):

    #Restrict I,J,D to k neighbors
    k = np.minimum(I.shape[1],k)
    I = I[:,:k]
    J = J[:,:k]
    D = D[:,:k]

    n = I.shape[0]
    k = I.shape[1]

    D = D*D
    eps = D[:,k-1]/4
    D = f(D/eps[:,None])

    #Flatten
    I = I.flatten()
    J = J.flatten()
    D = D.flatten()

    #Construct sparse matrix and convert to Compressed Sparse Row (CSR) format
    W = sparse.coo_matrix((D, (I,J)),shape=(n,n)).tocsr()

    if symmetrize:
        W = (W + W.transpose())/2
    for i in range( W.shape[0]):
        W[i,i] = 0
        
    return W


