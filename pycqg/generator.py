from __future__ import print_function, division
import numpy as np
import itertools
from ase.optimize import BFGS, LBFGS, FIRE
from ase.optimize.sciopt import SciPyFminBFGS, SciPyFminCG
from ase.constraints import UnitCellFilter, ExpCellFilter, StrainFilter
from ase.data import covalent_radii
import networkx as nx
from .crystgraph import graph_dim, graph_embedding
from .calculator import GraphCalculator
try:
    from pymatgen.analysis.local_env import MinimumDistanceNN, VoronoiNN
except:
    pass

class GraphGenerator:
    def __init__(self):
        pass

    def build_graph(self, atoms, coords, dim=None, maxtry=50, randGen=False, is2D=False):
        """
        Build QG from atoms.
        atoms: (ASE.Atoms) the input crystal structure.
        coords: a list containing coordination numbers for all atoms.
        dim: int or None, the target dimension. If None, do not restrict dimension.
        maxtry: max try times
        randGen: If True, randomly generate quotient graph when direct generation fails
        is2D: If False, use 3x3x3 supercell. If True, use 3x3x1 supercell.
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
        if is2D == True:
            offsetIter = itertools.product(range(-1, 2),range(-1, 2),[0])
        else:
            offsetIter = itertools.product(range(-1, 2),range(-1, 2),range(-1, 2))

        for offset in offsetIter:
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
        Unfinished
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


def get_neighbors_of_site_with_index(struct, n, approach, delta, cutoff=10.0):

    if approach == "min_dist":
        return MinimumDistanceNN(tol=delta, cutoff=cutoff).get_nn_info(struct, n)

    if approach == "voronoi":
        return VoronoiNN(tol=delta, cutoff=cutoff).get_nn_info(struct, n)

def quot_gen(atoms, struct, approach, delta, add_ratio=False):
    radius = [covalent_radii[number] for number in atoms.get_atomic_numbers()]
    cutoff = atoms.cell.cellpar()[:3].min()
    G = nx.MultiGraph()
    for i in range(len(struct)):
        G.add_node(i)

    for i in range(len(struct)):
        site = struct[i]
        
        neighs_list = get_neighbors_of_site_with_index(struct,i,approach,delta,cutoff)
        for nn in neighs_list:
            j = nn['site_index']
            if i <= j:
                rsum = radius[i] + radius[j]
                ratio = site.distance(nn['site']) / rsum
                if add_ratio:
                    G.add_edge(i,j, vector=np.array(nn['image']), direction=(i,j), ratio=ratio)
                else:
                    G.add_edge(i,j, vector=np.array(nn['image']), direction=(i,j))

    return G

def get_coor(struct):

    coor_list = []
    for i in range(len(struct)):
        site = struct[i]
        neighs_list = get_neighbors_of_site_with_index(struct,i)
        coor_list.append(len(neighs_list))

    return coor_list