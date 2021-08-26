#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  4 14:30:03 2021

@author: mdreveto
"""

import networkx as nx
import numpy as np


def generate_DCSBM( sizes, probs, theta, c = False ):
    n = sum( sizes )
    community_labels = []
    
    for k in range(len(sizes)):
        community_labels += [ k+1] * sizes[ k ] 
    
    if community_labels:
        np.random.shuffle( community_labels )
    
    nodes = [i for i in range( n ) ]
    G = nx.Graph( )
    G.add_nodes_from( nodes )
    
    for i in nodes:
        G.nodes[i]['community'] = community_labels[ i ]
    
    edges = [ ]
    for i in range( n ):
        for j in range( i ):
            if np.random.rand( ) < theta[i] * theta[j] * probs[ community_labels[i] - 1] [ community_labels[j] - 1]:
                edges.append( (i,j) )
    
    G.add_edges_from( edges )
    
    return G


def getCommunityLabels( G ):
    labels = [ ]
    for node in G.nodes:
        labels.append( G.nodes[node]['community'] )
    return np.asarray( labels, dtype = int )



def generateThetaDCSBM( N, law = 'sbm' ):
    
    if law == 'sbm':
        return np.ones( N )
    elif law == 'normal':
        theta = np.zeros( N )
        sigma = 0.25 #Here sigma for normal law can be changed
        for i in range( N ):
            #theta[ i ] = np.abs( 1 + np.random.normal( loc = 0, scale = 0.25 ) )
            theta[ i ] = np.abs( np.random.normal( loc = 0, scale = sigma ) ) + 1 - sigma * np.sqrt( 2 / np.pi )
            #TODO: seems the mean is not equal to 1 ?
        return theta

    elif law == 'pareto':
        a = 3 #Here can change the Pareto parameter
        return (np.random.pareto( a, N ) + 1) * (a-1) / a
    else:
        raise TypeError("The method is not implemented")