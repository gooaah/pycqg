## Crystal Quotient Graph
from __future__ import print_function, division
from functools import reduce
from ase.neighborlist import neighbor_list, primitive_neighbor_list
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
    while len(inputArr) > 0:
        c = inputArr.pop()
        if graph_dim(c) == 0:
            partition.append(list(c.nodes()))
            print("0D {}".format(c.nodes()))
        else:
            comp=nx.algorithms.community.girvan_newman(c)
            print("GN step")
            for indices in next(comp):
                print(indices)
                inputArr.append(tmpG.subgraph(indices))

    return partition


def remove_selfloops(G):
    newG = G.copy()
    # loops = list(newG.selfloop_edges())
    loops = list(nx.selfloop_edges(G))
    newG.remove_edges_from(loops)
    return newG

def nodes_and_offsets(G):
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

def graph_embedding(graph):
    """
    standard embedding of quotient graph.
    return scaled positions and modified graph
    """
    assert nx.number_connected_components(graph) == 1

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


def edge_ratios(graph):
    """
    Analyse distance ratio for every edge
    """
    ratios = []
    for edge in graph.edges(data=True):
        _, _, data = edge
        ratios.append(data['ratio'])

    return np.array(ratios)

def min_unbond_ratio(atoms, bondCoef):
    """
    Compute min distance ration for unbonded atom pairs
    atoms: ASE's Atoms object, the input structure
    bondCoef: same to coef in quotient_graph(), the cutoff parameter
    """
    # assert bondCoef < maxCoef, "Bonded ratios should be less than unbonded ratios."
    maxCoef = bondCoef + 1
    curCoef = maxCoef
    cutoffs = [covalent_radii[number]*maxCoef for number in atoms.get_atomic_numbers()]
    radius = [covalent_radii[number] for number in atoms.get_atomic_numbers()]

    for i, j, d in zip(*neighbor_list('ijd', atoms, cutoffs)):
        if i <= j:
            rsum = radius[i] + radius[j]
            ratio = d/rsum # bond parameter
            if bondCoef < ratio < curCoef:
                curCoef = ratio

    return curCoef


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
        for comp in enumerate(nx.connected_components(self.graph)):
            subG = self.graph.subgraph(comp).copy()
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

class GraphCalculator(Calculator):
    """
    Calculator written for optimizing crystal structure based on an initial quotient graph.
    I design the potential so that the following criteria are fulfilled when the potential energy equals zero:
    Suppose there are two atoms A, B. R_{AB} = r_{A} + r_{B} is the sum of covalent radius.
    If A,B are bonded, cmin <= d_{AB}/R_{AB} <= cmax.
    If A,B are not bonded, cmax+cadd <= d_{AB}/R_{AB}.
    """
    implemented_properties = ['energy', 'forces', 'stress']
    default_parameters = {
        'cmin': 0.5,
        'cmax': 1,
        'cadd': 0.,
        'k1': 1e-1,
        'k2': 1e-1,
        # 'mode': 0,
        'useGraphRatio': False,
        'ratioErr': 0.1,
    }
    def __init__(self, **kwargs):
        Calculator.__init__(self, **kwargs)

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        # Calculator.calculate(self, atoms, properties, system_changes)
        graph = atoms.info['graph']
        Nat = len(atoms)
        assert Nat == len(graph), "Number of atoms should equal number of nodes in the initial graph!"

        numbers = atoms.get_atomic_numbers()

        cmax = self.parameters.cmax
        cmin = self.parameters.cmin
        cadd = self.parameters.cadd
        k1 = self.parameters.k1
        k2 = self.parameters.k2
        useGraphRatio = self.parameters.useGraphRatio
        ratioErr = self.parameters.ratioErr
        # mode = self.parameters.mode
        cunbond = cmax + cadd

        # print(cadd)

        energy = 0.
        forces = np.zeros((Nat, 3))
        stress = np.zeros((3, 3))

        # print("Calling GraphCalculator.calculate")

        ## offsets for all atoms
        spos = atoms.get_scaled_positions(wrap=False)
        # remove little difference
        spos[np.abs(spos)<1e-4] = 0
        offsets = np.floor(spos).astype(np.int)
        # print(offsets[-1])
        # print(atoms.positions[-1,1])
        # if mode == 1:
        #     atoms.wrap()
        #     pos = atoms.get_positions(wrap=True)
        # elif mode == 0:
        pos = atoms.get_positions()
        cell = atoms.get_cell()
        pairs = []
        ## Consider bonded atom pairs
        for edge in graph.edges(data=True, keys=True):
            m,n, key, data = edge
            i,j = data['direction']
            # ## modify the quotient graph according to offsets
            # if mode == 1:
            #     edgeVec = offsets[j] - offsets[i] + data['vector']
            #     n1, n2, n3 = edgeVec
            # elif mode == 0:
            edgeVec = data['vector']
            n1, n2, n3 = offsets[j] - offsets[i] + data['vector']

            # graph.edge[m][n][key]['vector'] = edgeVec
            graph[m][n][key]['vector'] = edgeVec
            pairs.append((i,j,n1,n2,n3))
            rsum = covalent_radii[numbers[i]] + covalent_radii[numbers[j]]
            # Use ratio attached in graph or not
            if useGraphRatio:
                ratio = data['ratio']
                # cmax = ratio + ratioErr
                cmax = ratio
                cmin = ratio - ratioErr
            Dmin = cmin * rsum
            Dmax = cmax * rsum
            # print("bond")
            # print(Dmax)
            cells = np.dot(edgeVec, cell)
            dvec = pos[j] + cells - pos[i]
            D = np.linalg.norm(dvec)
            # when the distance is too small
            if D > 1e-3:
                uvec = dvec/D
            else:
                dvec = np.random.rand(3)
                uvec = dvec/np.linalg.norm(dvec)
            if Dmin <= D <= Dmax:
                continue
            else:
                if D < Dmin:
                    # scalD = (D-Dmin)/rsum
                    # energy += 0.5*k1*scalD**2
                    # f = k1*scalD*uvec/rsum
                    energy += 0.5*k1*(D-Dmin)**2
                    f = k1*(D-Dmin)*uvec
                elif D > Dmax:
                    # scalD = (D-Dmax)/rsum
                    # energy += 0.5*k1*scalD**2
                    # f = k1*scalD*uvec/rsum
                    energy += 0.5*k1*(D-Dmax)**2
                    f = k1*(D-Dmax)*uvec
                # elif D <= 1e-3: # when the distance is too small
                #     energy += 0.5*k1*(D-Dmin)**2
                #     dvec = np.random.rand(3)
                #     uvec = dvec/np.linalg.norm(dvec)
                #     f = k1*(D-Dmin)*uvec
                forces[i] += f
                forces[j] -= f
                stress += np.dot(f[np.newaxis].T, dvec[np.newaxis])

        # Wraping atoms might lead to changing vector labels.
        # Thererfore, to avoid wraping positions, 
        # I copy a new atoms and get wrapped positions
        copyAts = atoms.copy()
        copyAts.wrap()
        pos = copyAts.get_positions()
        ## Consider unbonded atom pairs
        cutoffs = [covalent_radii[n]*(cunbond+0.05) for n in numbers]
        # for i, j, S in zip(*neighbor_list('ijS', atoms, cutoffs)):
        for i, j, S in zip(*neighbor_list('ijS', copyAts, cutoffs)):
            if i <= j:
                # if i,j is bonded, skip this process
                n1,n2,n3 = S
                pair1 = (i,j,n1,n2,n3)
                pair2 = (j,i,-n1,-n2,-n3)
                if pair1 in pairs or pair2 in pairs:
                    # print("Skip")
                    continue
                rsum = covalent_radii[numbers[i]] + covalent_radii[numbers[j]]
                Dmax = (cunbond) * rsum
                # print("Unbond")
                # print(Dmax)
                cells = np.dot(S, cell)
                dvec = pos[j] + cells - pos[i]
                D = np.linalg.norm(dvec)
                if D > 1e-3:
                    uvec = dvec/D
                else:
                    dvec = np.random.rand(3)
                    D = np.linalg.norm(dvec)
                    uvec = dvec/D
                    # D = 1e-3
                if D < Dmax:
                    energy += 0.5*k2*(D-Dmax)**2
                    f = k2*(D-Dmax)*uvec
                    forces[i] += f
                    forces[j] -= f
                    stress += np.dot(f[np.newaxis].T, dvec[np.newaxis])


        # if mode == 1:
        #     ## save the quotient graph
        #     atoms.info['graph'] = graph
        #     atoms.wrap()

        # self.atoms = atoms

        stress = 1*stress/atoms.get_volume()

        self.results['energy'] = energy
        self.results['free_energy'] = energy
        self.results['forces'] = forces
        self.results['stress'] = stress.flat[[0, 4, 8, 5, 2, 1]]