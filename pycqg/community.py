from __future__ import print_function, division
import numpy as np
from functools import reduce
import networkx as nx
from .crystgraph import remove_selfloops, cycle_sums, graph_dim



def find_communities_old1(QG):
    """
    (Old version)
    Find communitis of crystal quotient graph QG using Girva_Newman algorithm.
    QG: networkx.MultiGraph
    Return: a list of networkx.MultiGraph
    """
    tmpG = remove_selfloops(QG)
    comp=nx.algorithms.community.girvan_newman(tmpG)
    for c in comp:
        SGs = [tmpG.subgraph(indices) for indices in c]
        dims = [np.linalg.matrix_rank(cycle_sums(SG)) for SG in SGs]
        sumDim = sum(dims)
        if sumDim == 0:
            break

    partition = [list(p) for p in c]
    return partition

def find_communities_old2(QG, maxStep=1000):
    """
    Find communitis of crystal quotient graph QG using Girva_Newman algorithm, slightly different from find_communities.
    QG: networkx.MultiGraph
    Return: a list of networkx.MultiGraph
    """
    tmpG = remove_selfloops(QG)
    partition = []
    for _ in range(maxStep):
        comp=nx.algorithms.community.girvan_newman(tmpG)
        for c in comp:
            extendList = []
            sumDim = 0
            dims = []
            # print("c: {}".format(sorted(c)))
            for indices in c:
                SG = tmpG.subgraph(indices)
                dim = np.linalg.matrix_rank(cycle_sums(SG))
                sumDim += dim
                dims.append(dim)
                if dim == 0:
                    partition.append(list(indices))
                else:
                    extendList.append(tmpG.subgraph(indices))
            if sumDim == 0:
                return partition
            # print("dims:{}".format(dims))

            if len(extendList) > 0:
                tmpG = reduce(nx.union, extendList)
                break

def find_communities(QG):
    """
    Find communitis of crystal quotient graph QG using Girva_Newman algorithm.
    QG: networkx.MultiGraph
    Return: a list of networkx.MultiGraph
    """
    tmpG = remove_selfloops(QG)
    partition = []
    # inputArr = list(nx.connected_component_subgraphs(tmpG))
    inputArr = [tmpG.subgraph(comp).copy() for comp in nx.connected_components(tmpG)]
    step = 0
    while len(inputArr) > 0:
        c = inputArr.pop()
        compDim = graph_dim(c)
        if compDim == 0:
            partition.append(list(c.nodes()))
            print("community 0D {}".format(c.nodes()))
        else:
            comp=nx.algorithms.community.girvan_newman(c)
            step += 1
            print("GN step {}".format(step))
            for indices in next(comp):
                print("{}D".format(compDim))
                print(indices)
                inputArr.append(tmpG.subgraph(indices))

    return partition

def find_communities_3D(QG):
    """
    Find communitis with dimensionality lower than 3 of crystal quotient graph QG using Girva_Newman algorithm.
    QG: networkx.MultiGraph
    Return: a list of networkx.MultiGraph
    """
    tmpG = remove_selfloops(QG)
    partition = []
    # inputArr = list(nx.connected_component_subgraphs(tmpG))
    inputArr = [tmpG.subgraph(comp).copy() for comp in nx.connected_components(tmpG)]
    step = 0
    while len(inputArr) > 0:
        c = inputArr.pop()
        compDim = graph_dim(c)
        if compDim < 3:
            partition.append(list(c.nodes()))
            # print("community {}D {}".format(compDim, c.nodes()))
        else:
            comp=nx.algorithms.community.girvan_newman(c)
            step += 1
            # print("{}D".format(compDim))
            # print("GN step {}".format(step))
            for indices in next(comp):
                # print(indices)
                inputArr.append(tmpG.subgraph(indices))

    return partition