from __future__ import print_function, division
import numpy as np
import itertools
from ase.optimize import BFGS, LBFGS, FIRE
from ase.optimize.sciopt import SciPyFminBFGS, SciPyFminCG
from ase.constraints import UnitCellFilter, ExpCellFilter, StrainFilter
from ase.data import covalent_radii
import networkx as nx
from .crystgraph import graph_dim, graph_embedding, GraphCalculator

class GraphGenerator:
    def __init__(self):
        pass

    def build_graph(self, atoms, coords, dim=None, maxtry=50, randGen=False):
        """
        Build QG from atoms.
        atoms: (ASE.Atoms) the input crystal structure.
        coords: a list containing coordination numbers for all atoms.
        dim: int or None, the target dimension. If None, do not restrict dimension.
        maxtry: max try times
        randGen: If True, randomly generate quotient graph when direct generation fails
        """
        self.originAtoms = atoms
        assert sum(coords)%2 == 0, "Sum of coordination numbers should be even number!"
        Nat = len(atoms)
        assert Nat == len(coords), "coords and atoms should have same length!"
        cell = np.array(atoms.get_cell())
        pos = atoms.get_positions()
        numbers = atoms.get_atomic_numbers()
        ## Calculate relative vectors between different atoms in the same cell, and save the indices and cell offset, and sum of covalent radius
        disp = []
        pair = []
        rsum = []
        for m,n in itertools.combinations(range(Nat), 2):
            disp.append(pos[n] - pos[m])
            pair.append([m,n,0,0,0])
            rsum.append(covalent_radii[numbers[m]] + covalent_radii[numbers[n]])
        inDisp = np.array(disp)
        inPair = np.array(pair)
        inRsum = rsum[:]
        ## Calculate relative vectors between atoms in the origin cell and atoms in the surrounding 3x3x3 supercell
        negative_vec = lambda vec: tuple([-1*el for el in vec])
        duplicate = []
        for offset in itertools.product(range(-1, 2),range(-1, 2),range(-1, 2)):
            if offset != (0,0,0):
                cellDisp = np.dot(offset, cell)
                if Nat > 1:
                    newDisp = inDisp + cellDisp
                    newPair = inPair.copy()
                    newPair[:,2:5] = offset
                    disp.extend(list(newDisp))
                    pair.extend(newPair.tolist())
                    rsum.extend(inRsum)
                ## Consider equivalent atoms in two different cells
                if negative_vec(offset) not in duplicate:
                    duplicate.append(offset)
                    for ii in range(Nat):
                        disp.append(cellDisp)
                        pair.append([ii, ii, offset[0], offset[1], offset[2]])
                        rsum.append(2*covalent_radii[numbers[ii]])

        disp = np.array(disp)
        D = np.linalg.norm(disp, axis=1)
        ratio = D/rsum
        self.disp = disp
        self.pair = pair
        self.D = D
        self.ratio = ratio

        sortInd = np.argsort(ratio)

        ## Search for optimal connectivity
        edges, edgeInds = search_edges(sortInd, coords, pair)

        G = nx.MultiGraph()
        for i in range(Nat):
            G.add_node(i)

        if edges is not None:
            for edge, ind in zip(edges, edgeInds):
                m,n,i,j,k = edge
                G.add_edge(m,n, vector=[i,j,k], direction=(m,n), ratio=ratio[ind])

            self.randG = G
            if nx.number_connected_components(G) == 1:
                if dim is not None:
                    if graph_dim(G) == dim:
                        return G
                else:
                    return G

        ## If we cannot build the graph, we choose the edges randomly
        # raise NotImplementedError("Random generation has never been implemented!")
        print("Cannot build quotient graph directly.")
        if not randGen:
            return None

        print("Try to build graph randomly.")
        for _ in range(maxtry):
            randInd = sortInd[:]
            np.random.shuffle(randInd)
            edges, edgeInds = search_edges(randInd, coords, pair)
            if edges is not None:
                G.remove_edges_from(list(G.edges()))
                for edge, ind in zip(edges, edgeInds):
                    m,n,i,j,k = edge
                    G.add_edge(m,n, vector=[i,j,k], direction=(m,n), ratio=ratio[ind])

                    self.randG = G
                    if nx.number_connected_components(G) == 1:
                        if dim is not None:
                            if graph_dim(G) == dim:
                                return G
                        else:
                            return G

        print("Fail in generating quotient graph.")
        return None

    def optimize(self, calcParm, driver, optParm, standardize=False):
        """
        Optimize structure according to the quotient graph.
        calcParm: dict, parameters of GraphCalculator
        driver: optimization driver, such as ase.optimize.BFGS
        optParm: dict, parameters of optimization driver
        standardize: If True, change the scaled positions of atoms to standard placement before optimization.
        """
        graph = self.randG
        atoms = self.originAtoms.copy()

        if standardize:
            standPos, graph = graph_embedding(graph)
            atoms.set_scaled_positions(standPos)

        calc = GraphCalculator(**calcParm)
        atoms.set_calculator(calc)
        atoms.info['graph'] = graph.copy()


def search_edges(indices, coords, pairs):
    """
    search for edges so that coordination numbers are fulfilled.
    """
    assert len(indices) == len(pairs)
    edges = []
    edgeInds = []
    curCoords = np.array(coords)
    for ind in indices:
        m,n,_,_,_ = pairs[ind]
        if (curCoords == 0).all():
            break
        if curCoords[m] > 0 and curCoords[n] > 0:
            edges.append(pairs[ind])
            edgeInds.append(ind)
            curCoords[m] -= 1
            curCoords[n] -= 1
    if (curCoords == 0).all():
        return edges, edgeInds
    else:
        return None, None