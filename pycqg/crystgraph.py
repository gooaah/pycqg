## Crystal Quotient Graph
from __future__ import print_function, division
from functools import reduce
from ase.neighborlist import neighbor_list
from ase.calculators.calculator import Calculator, all_changes
from ase.optimize import BFGS, LBFGS, FIRE
from ase.optimize.sciopt import SciPyFminBFGS, SciPyFminCG
from ase.constraints import UnitCellFilter, ExpCellFilter, StrainFilter 
from ase.data import covalent_radii
import ase.io
import networkx as nx
import numpy as np
import sys, itertools
from math import gcd

def quotient_graph(atoms, coef=1.1,):
    """
    Return crystal quotient graph of the atoms.
    atoms: (ASE.Atoms) the input crystal structure 
    coef: (float) the criterion for connecting two atoms. If d_{AB} < coef*(r_A + r_B), atoms A and B are regarded as connected. r_A and r_B are covalent radius of A,B.
    Return: networkx.MultiGraph
    """

    cutoffs = [covalent_radii[number]*coef for number in atoms.get_atomic_numbers()]
    # print("cutoffs: %s" %(cutoffs))
    G = nx.MultiGraph()
    for i in range(len(atoms)):
        G.add_node(i)

    for i, j, S in zip(*neighbor_list('ijS', atoms, cutoffs)):
        if i <= j:
            G.add_edge(i,j, vector=S, direction=(i,j))

    return G

def cycle_sums(G):
    """
    Return the cycle sums of the crystal quotient graph G.
    G: networkx.MultiGraph
    Return: a (Nx3) matrix
    """
    SG = nx.Graph(G) # Simple graph, maybe with loop.
    cycBasis = nx.cycle_basis(SG)
    # print(cycBasis)
    cycleSums = []

    for cyc in cycBasis:
        cycSum = np.zeros([3])
        for i in range(len(cyc)):
            vector = SG[cyc[i-1]][cyc[i]]['vector']
            direction = SG[cyc[i-1]][cyc[i]]['direction']
            cycDi = (cyc[i-1], cyc[i])
            if cycDi == direction:
                cycSum += vector
                # cycSum += SG[cyc[i-1]][cyc[i]]['vector']
                # print("path->: %s, vector: %s" %((cyc[i-1], cyc[i]), SG[cyc[i-1]][cyc[i]]['vector']))

            elif cycDi[::-1] == direction:
                cycSum -= vector
                # cycSum -= SG[cyc[i-1]][cyc[i]]['vector']
                # print("path<-: %s, vector: %s" %((cyc[i-1], cyc[i]), SG[cyc[i-1]][cyc[i]]['vector']))

            else:
                raise RuntimeError("Error in direction!")
                # print("Error in direction!")
        cycleSums.append(cycSum)

    for edge in SG.edges():
        numEdges = list(G.edges()).count(edge)
        if numEdges > 1:
            direction0 = G[edge[0]][edge[1]][0]['direction']
            vector0 = G[edge[0]][edge[1]][0]['vector']
            for j in range(1, numEdges):
                directionJ = G[edge[0]][edge[1]][j]['direction']
                vectorJ = G[edge[0]][edge[1]][j]['vector']
                # cycSum = G[edge[0]][edge[1]][0]['vector'] - G[edge[0]][edge[1]][j]['vector']
                if direction0 == directionJ:
                    cycSum = vector0 - vectorJ
                elif direction0[::-1] == directionJ:
                    cycSum = vector0 + vectorJ
                else:
                    raise RuntimeError("Error in direction!")
                cycleSums.append(cycSum)

    return np.array(cycleSums, dtype=np.int)

def graph_dim(G):
    """
    Return the dimensionality of the crystal quotient graph G.
    G: networkx.MultiGraph
    Return: int
    """
    return np.linalg.matrix_rank(cycle_sums(G))

def max_graph_dim(G):
    """
    Return the max dimensionality among all of the components in crystal quotient graph G.
    G: networkx.MultiGraph
    Return: int
    """
    maxDim = 0
    for i,subG in enumerate(nx.connected_component_subgraphs(G)):
        dim = graph_dim(subG)
        if dim > maxDim:
            maxDim = dim
    return maxDim

def reduce_cycSums(cycSums):

    csArr = cycSums.tolist()
    noZeroSum = []
    for cs in csArr:
        cst = tuple(cs)
        if cst != (0,0,0):
            # if cst[0] < 0:
            if num_geq_zero(cst) < 2:
                cst = tuple([-1*i for i in cst])
            noZeroSum.append(cst)
    noZeroSum = list(set(noZeroSum))
    noZeroMat = np.array(noZeroSum)

    return noZeroMat

def num_geq_zero(vec):
    """
    Count number of elements greater than or equal to zero.
    """
    n = 0
    for i in vec:
        if i >= 0:
            n += 1

    return n

def getMult_3D(cycSums):
    """
    Return the self-penetration multiplicities of the 3D crystal quotient graph G.
    G: networkx.MultiGraph
    Return: int
    """
    assert np.linalg.matrix_rank(cycSums) == 3
    noZeroMat = reduce_cycSums(cycSums)

    # determinants
    allDets = []
    for comb in itertools.combinations(range(len(noZeroMat)), 3):
        mat = noZeroMat[list(comb)]
        det = abs(np.linalg.det(mat))
        if abs(det) > 1e-3:
            allDets.append(det)
    minDet = int(min(allDets))
    return minDet

def getMult_2D(cycSums):
    """
    Return the self-penetration multiplicities of the 2D crystal quotient graph G.
    G: networkx.MultiGraph
    Return: int
    """
    assert np.linalg.matrix_rank(cycSums) == 2
    noZeroMat = reduce_cycSums(cycSums)

    # determinants
    allDets = []
    for comb in itertools.combinations(range(len(noZeroMat)), 2):
        mat = noZeroMat[list(comb)]
        for indices in itertools.combinations(range(3), 2):
            det = abs(np.linalg.det(mat[:,indices]))
            if abs(det) > 1e-3:
                allDets.append(det)
    minDet = int(min(allDets))
    return minDet

def getMult_1D(cycSums):
    """
    Return the self-penetration multiplicities of the 1D crystal quotient graph G.
    G: networkx.MultiGraph
    Return: int
    """
    assert np.linalg.matrix_rank(cycSums) == 1
    noZeroMat = reduce_cycSums(cycSums)

    allDets = []
    for row in noZeroMat.tolist():
        # no zero row
        nZRow = [int(i) for i in row if i != 0]
        if len(nZRow) == 3:
            allDets.append(gcd(gcd(nZRow[0], nZRow[1]), nZRow[2]))
        elif len(nZRow) == 2:
            allDets.append(gcd(nZRow[0], nZRow[1]))
        elif len(nZRow) == 1:
            allDets.append(abs(nZRow[0]))
    minDet = int(min(allDets))

    return minDet

def number_of_parallel_edges(QG):
    numPE = 0
    for i, j in set(QG.edges()):
        numEdges = len(QG[i][j])
        if numEdges > 1:
            numPE += 1
    return numPE

def find_communities(QG):
    """
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

def find_communities2(QG, maxStep=1000):
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

def remove_selfloops(G):
    newG = G.copy()
    loops = list(newG.selfloop_edges())
    newG.remove_edges_from(loops)
    return newG

def nodes_and_offsets(G):
    offSets = []
    nodes = list(G.nodes())
    paths = nx.single_source_shortest_path(G, nodes[0])
    for index, i in enumerate(nodes):
        if index is 0:
            offSets.append([0,0,0])
        else:
            path = paths[i]
            offSet = np.zeros((3,))
            for j in range(len(path)-1):
                # print(j)
                vector = G[path[j]][path[j+1]][0]['vector']
                direction = G[path[j]][path[j+1]][0]['direction']
                pathDi = (path[j], path[j+1])
                if pathDi == direction:
                    offSet += vector
                elif pathDi[::-1] == direction:
                    offSet -= vector
                else:
                    raise RuntimeError("Error in direction!")
            offSets.append(offSet.tolist())
    return nodes, offSets

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
    
def edge_ratios(graph):
    """
    Analyse distance ratio for every edge
    """
    ratios = []
    for edge in graph.edges(data=True):
        _, _, data = edge
        ratios.append(data['ratio'])
    
    return np.array(ratios)


class CrystalGraph:
    def __init__(self, atoms, coef=1.1, buildQG=True):
        """
        atoms: (ASE.Atoms) the input crystal structure.
        coef: (float) the criterion for connecting two atoms. If d_{AB} < coef*（r_A + r_B), atoms A and B are regarded as connected. r_A and r_B are covalent radius of A,B.
        buildQG: If it is true, build quotient graph using atoms and coef.
        """
        self.atoms = atoms
        if buildQG:
            self.update_graph(coef)

    def update_graph(self, coef):
        """
        Set new coef and rebuild the quotient graph
        """
        self.coef = coef
        self.graph = quotient_graph(self.atoms, self.coef)

    def get_max_dim(self):
        """
        Same to max_graph_dim
        """
        return max_graph_dim(self.graph)

    def analyse(self):
        """
        Analyse dimensionality and multiplicity of each connected component.
        """
        dims = []
        mults = []
        subGs = []
        for i, subG in enumerate(nx.connected_component_subgraphs(self.graph)):
            subGs.append(subG)
            cycSum = cycle_sums(subG)
            dim = np.linalg.matrix_rank(cycSum)
            dims.append(dim)
            if dim == 3:
                mult = getMult_3D(cycSum)
            elif dim == 2:
                mult = getMult_2D(cycSum)
            elif dim == 1:
                mult = getMult_1D(cycSum)
            elif dim == 0:
                mult = 1
            mults.append(mult)
            # print("{}\t\t{}\t{}".format(i+1, dim, mut))
        
        self.dims = dims
        self.mults = mults
        self.subGs = subGs


class GraphGenerator:
    def __init__(self):
        pass

    def build_graph(self, atoms, coords, dim=None, maxtry=10, randGen=False):
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

        ## If cannot build the graph, choose the edges randomly
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