#!/usr/bin/env python
## Crystal Quotient Graph
from __future__ import print_function, division
import ase.io
from pycqg.crystgraph import quotient_graph, graph_dim, cycle_sums, getMult_3D, getMult_2D, getMult_1D, number_of_parallel_edges, max_graph_dim, nodes_and_offsets
import networkx as nx
import numpy as np
import sys
from ase.build import sort
from ase import Atoms


if __name__ == "__main__":

    filename = sys.argv[1]
    atoms = ase.io.read(filename)

    if len(sys.argv) == 3:
        coef = float(sys.argv[2])
    else:
        coef = 1.1

    assert (atoms.get_pbc() == [1,1,1]).all(), "The input structure should be a crystal."
    
    print('Get components in the crystal')


    QG = quotient_graph(atoms, coef)

    print("Component\tDimension")
    # for i,subG in enumerate(nx.connected_component_subgraphs(QG)):
    for i,subG in enumerate(nx.connected_components(QG)):
        dim = graph_dim(QG.subgraph(subG))
        print("{}\t\t{}".format(i+1, dim))
        
        # save all components
        subAts = atoms[list(subG)]
        ase.io.write('comp{}.vasp'.format(i), sort(subAts))
        # save isolated components
        if dim == 0:
            nodes, offsets = nodes_and_offsets(QG.subgraph(subG))
            subAts = atoms[nodes]
            molPos = subAts.get_positions()
            molPos += np.dot(offsets, subAts.get_cell())
            mol = Atoms(numbers=subAts.numbers, positions=molPos, pbc=False)
            ase.io.write('mol{}.xyz'.format(i), mol)

        
    print("Max Dimension: {}".format(max_graph_dim(QG)))


