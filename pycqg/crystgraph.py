## Crystal Quotient Graph
from __future__ import print_function, division
from functools import reduce
from ase.neighborlist import neighbor_list
from ase.optimize import BFGS, LBFGS, FIRE
from ase.optimize.sciopt import SciPyFminBFGS, SciPyFminCG
from ase.constraints import UnitCellFilter, ExpCellFilter, StrainFilter
from ase.data import covalent_radii
import ase.io
import networkx as nx
import numpy as np
import sys, itertools, functools
from math import gcd

def quotient_graph(atoms, coef=1.1,):
    """
    Return crystal quotient graph of the atoms.
    atoms: (ASE.Atoms) the input crystal structure
    coef: (float) the criterion for connecting two atoms. If d_{AB} < coef*(r_A + r_B), atoms A and B are regarded as connected. r_A and r_B are covalent radius of A,B.
    Return: networkx.MultiGraph
    """

    cutoffs = [covalent_radii[number]*coef for number in atoms.get_atomic_numbers()]
    radius = [covalent_radii[number] for number in atoms.get_atomic_numbers()]
    # print("cutoffs: %s" %(cutoffs))
    G = nx.MultiGraph()
    for i in range(len(atoms)):
        G.add_node(i)

    for i, j, S, d in zip(*neighbor_list('ijSd', atoms, cutoffs)):
        if i <= j:
            rsum = radius[i] + radius[j]
            ratio = d/rsum # bond parameter
            G.add_edge(i,j, vector=S, direction=(i,j), ratio=ratio)

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
    # for i,subG in enumerate(nx.connected_component_subgraphs(G)):
    for c in nx.connected_components(G):
        dim = graph_dim(G.subgraph(c))
        if dim > maxDim:
            maxDim = dim
    return maxDim

def reduce_cycSums(cycSums):
    """
    Delete zero and duplicate vectors in the input cycle sum matrix.
    (Transform the type of elements to int)
    """

    csArr = cycSums.tolist()
    noZeroSum = []
    for cs in csArr:
        cst = tuple([int(n) for n in cs])
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

def get_basis(cycSums):
    """
    Get the basis of the cycle sum matrix and self-penetation multiplicity.
    
    Parameters:

    cycSums: a matrix (Nx3)
        cycle sum matrix

    Returns:

    basis: a matrix (Dx3)
        basic vectors. D is the dimensionality or rank.
    mult: int
        self-penetation multiplicity
    """

    dim = np.linalg.matrix_rank(cycSums)
    noZeroMat = reduce_cycSums(cycSums)


    # multiplicity
    mult = None
    basis = None
    if dim == 3: 
        for comb in itertools.combinations(range(len(noZeroMat)), 3):
            mat = noZeroMat[list(comb)]
            m = abs(np.linalg.det(mat))
            if mult and m > 1e-3 and m < mult: # avoid error led by float type
                mult = int(m)
                basis = mat
            elif not mult:
                mult = int(m)
                basis = mat

    elif dim == 2:
        for comb in itertools.combinations(range(len(noZeroMat)), 2):
            mat = noZeroMat[list(comb)]
            # cross product of two vector
            # convert to int for gcd
            detRow = [int(i) for i in np.cross(mat[0], mat[1])]
            m = functools.reduce(gcd, detRow)
            if mult and m > 0 and m < mult: 
                mult = m
                basis = mat
            elif not mult:
                mult = m
                basis = mat

    elif dim == 1:
        for row in noZeroMat.tolist():
            mat = [int(i) for i in row]
            m = functools.reduce(gcd, mat)
            if mult and m > 0 and m < mult: 
                mult = m
                basis = mat
            elif not mult:
                mult = m
                basis = mat

    elif dim == 0:
        basis = []
        mult = 1

    else:
        raise RuntimeError("Please check the input matrix!")
    
    return basis, mult




def getMult_3D(cycSums):
    """
    Return the self-penetration multiplicities of the 3D crystal quotient graph G.
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


def remove_selfloops(G):
    newG = G.copy()
    # loops = list(newG.selfloop_edges())
    loops = list(nx.selfloop_edges(G))
    newG.remove_edges_from(loops)
    return newG

def nodes_and_offsets(G):
    """
    Look for the cell offsets connected to the atom a0(0,0,0).
    a0(0,0,0) is the first atom in cell (0,0,0)
    Usually it only makes sense for 0D sturctures.
    """
    assert nx.number_connected_components(G) == 1, "The graph should be connected!"
    offSets = []
    nodes = list(G.nodes())
    paths = nx.single_source_shortest_path(G, nodes[0])
    for index, i in enumerate(nodes):
        if index == 0:
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


def graph_embedding(graph):
    """
    standard embedding of quotient graph.
    return scaled positions and modified graph
    """
    assert nx.number_connected_components(graph) == 1, "The input graph should be connected!"

    if len(graph) == 1:
        return np.array([[0,0,0]]), graph

    ## Laplacian Matrix
    lapMat = nx.laplacian_matrix(graph).todense()
    invLap = np.linalg.inv(lapMat[1:,1:])
    sMat = np.zeros((len(graph), 3))
    for edge in graph.edges(data=True):
        _,_,data = edge
        i,j = data['direction']
        sMat[i] += data['vector']
        sMat[j] -= data['vector']
    pos = np.dot(invLap,sMat[1:])
    pos = np.concatenate(([[0,0,0]], pos))
    pos = np.array(pos)
    # remove little difference
    pos[np.abs(pos)<1e-4] = 0
    # solve offsets
    offsets = np.floor(pos).astype(np.int)
    pos -= offsets
    newG = graph.copy()
    for edge in newG.edges(data=True, keys=True):
        m,n, key, data = edge
        i,j = data['direction']
        ## modify the quotient graph according to offsets
        # newG.edge[m][n][key]['vector'] = offsets[j] - offsets[i] + data['vector']
        newG[m][n][key]['vector'] = offsets[j] - offsets[i] + data['vector']

    return pos, newG

def barycenter_disp(atoms, graph):
    """
    Compute the displacement of every atom from the barycenter of neighboring atoms.
    atoms: ASE Atoms object, the structure
    graph: the quotient graph related to atoms
    """

    assert len(atoms) == graph.number_of_nodes(), "The number of atoms should equal to number of nodes in graph!"

    pos = atoms.get_scaled_positions()
    # Sum of neighboring positions
    sumPos = np.zeros_like(pos)
    degrees = np.zeros(len(atoms))
    for edge in graph.edges(data=True):
        _,_,data = edge
        i,j = data['direction']
        sumPos[i] += pos[j] + data['vector']
        sumPos[j] += pos[i] - data['vector']
        degrees[i] += 1
        degrees[j] += 1
    
    barycenters = sumPos / np.expand_dims(degrees,1)
    disp = pos - barycenters

    return disp

def sym_graph_embedding(graph):
    """
    Almost same to `graph_embedding` but in a symbolic way.
    Return positions are composed by excat integer and fractional numbers
    """
    assert nx.number_connected_components(graph) == 1, "The input graph should be connected!"

    if len(graph) == 1:
        return np.array([[0,0,0]]), graph


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
        bases = []
        for comp in enumerate(nx.connected_components(self.graph)):
            subG = self.graph.subgraph(comp).copy()
            subGs.append(subG)
            cycSum = cycle_sums(subG)
            dim = np.linalg.matrix_rank(cycSum)
            dims.append(dim)
            # if dim == 3:
            #     mult = getMult_3D(cycSum)
            # elif dim == 2:
            #     mult = getMult_2D(cycSum)
            # elif dim == 1:
            #     mult = getMult_1D(cycSum)
            # elif dim == 0:
            #     mult = 1
            basis, mult = get_basis(cycSum)
            mults.append(mult)
            bases.append(basis)
            # print("{}\t\t{}\t{}".format(i+1, dim, mut))

        self.dims = dims
        self.mults = mults
        self.subGs = subGs
        self.bases = bases
