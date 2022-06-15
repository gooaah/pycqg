from __future__ import print_function, division
import numpy as np
from ase.data import covalent_radii
import networkx as nx
from ase.neighborlist import neighbor_list
from .crystgraph import nodes_and_offsets, get_basis, cycle_sums

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

def analyze_vacuum(lat, spos, QG):
    """
    Only for 2D layers. Compute the layer thickness and vacuum beteewn layers.

    Parameters:
    lat: (3x3) matrix
        lattice matrix
    spos: (Nx3) matrix
        scaled positions
    QG: graph with N nodes
        quotient graph of the crystal defined by lat and spos

    Returns:

    """

    # 2 basic vectors of the layer
    comps = [QG.subgraph(comp).copy() for comp in nx.connected_components(QG)]
    allBasis = []
    for G in comps:
        basis, _ = get_basis(cycle_sums(G))
        allBasis.extend(basis)
    assert np.linalg.matrix_rank(allBasis) == 2, "Must be a 2D structure and all the layers should be parallel."

    # use the last basis, because all the layers are parallel.
    a, b = basis
    c = [int(i) for i in np.cross(a,b)]
    # find the axis not perpendicular to c
    nonZeroInd = np.nonzero(c)[0][0]

    # stack direction perpendicular to the layer
    stackVec = np.cross(np.dot(a,lat), np.dot(b,lat))
    stackVec = stackVec/np.linalg.norm(stackVec)
    stackLen = np.dot(lat[nonZeroInd], stackVec)
    stackLen = abs(stackLen)

    bottoms_tops = []
    # get offset for every atom
    for G in comps:
        indices, offsets = nodes_and_offsets(G)
        offsets = np.array(offsets)
        pos = np.dot(offsets + spos[indices], lat)
        # project the positions to the stack direction
        proj = np.dot(pos, stackVec)
        bottoms_tops.append((min(proj), max(proj)))
        # thick = max(proj) - min(proj)
        # vac = stackLen - thick
    
    bottoms_tops = sorted(bottoms_tops, key=lambda x:x[0])
    btArr = np.array(bottoms_tops)
    bottoms = btArr[:,0]
    tops = btArr[:,1]
    tmp = bottoms[0]
    bottoms[:-1] = bottoms[1:]
    bottoms[-1] = tmp
    gaps = bottoms - tops
    gaps[-1] += stackLen


    return stackLen, bottoms_tops, gaps