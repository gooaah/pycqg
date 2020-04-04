#!/usr/bin/env python
## Crystal Quotient Graph
from __future__ import print_function, division
import ase.io
from pycqg.crystgraph import quotient_graph, graph_dim, cycle_sums, getMult_3D, getMult_2D, getMult_1D, number_of_parallel_edges, max_graph_dim
import networkx as nx
import numpy as np
import sys


if __name__ == "__main__":

    filename = sys.argv[1]
    atoms = ase.io.read(filename)

    if len(sys.argv) == 3:
        coef = float(sys.argv[2])
    else:
        coef = 1.1

    assert (atoms.get_pbc() == [1,1,1]).all(), "The input structure should be a crystal."


    QG = quotient_graph(atoms, coef)
    edges = QG.edges()
    degree = QG.degree()
    print("number of bonds: %s"% (len(edges)))
    print("Number of self loops: {}".format(QG.number_of_selfloops()))
    print("Number of parallel edges: {}".format(number_of_parallel_edges(QG)))
    print("Number of components: {}".format(nx.number_connected_components(QG)))
    print("Coordination Numbers: {}".format(list(degree.values())))


    print("Component\tDimension")
    for i,subG in enumerate(nx.connected_component_subgraphs(QG)):
        dim = graph_dim(subG)
        print("{}\t\t{}".format(i+1, dim))
    print("Max Dimension: {}".format(max_graph_dim(QG)))

    print("\nCalculating Multplicities...")
    print("Component\tMultiplicity")
    for i,subG in enumerate(nx.connected_component_subgraphs(QG)):
        subCS = cycle_sums(subG)
        dim = np.linalg.matrix_rank(subCS)
        if dim == 3:
            print("{}\t\t{}".format(i+1,getMult_3D(subCS)))
        elif dim == 2:
            print("{}\t\t{}".format(i+1,getMult_2D(subCS)))
        elif dim == 1:
            print("{}\t\t{}".format(i+1,getMult_1D(subCS)))
        else:
            print("{}\t\t1".format(i+1))









