from __future__ import print_function, division
import numpy as np
import networkx as nx
from ase import Atoms
from .crystgraph import quotient_graph, cycle_sums, graph_dim, remove_selfloops, nodes_and_offsets
try:
    from functools import reduce
except ImportError:
    pass


class MolCryst:
    def __init__(self, numbers, cell, sclPos, partition, offSets=None, sclCenters=None, rltSclPos=None, info=dict()):
        """
        numbers: atomic numbers for all atoms in the structure, same to ase.Atoms.numbers
        cell: cell of the structure, a 3x3 matrix
        sclPos: scaled positions for all atoms, same to ase.Atoms.scaled_positions
        partition: atom indices for every molecule, like [[0,1], [2,3,4]], which means atoms 0,1 belong to one molecule and atoms 3,4,5 belong to another molecule.
        offSets: cell offsets for every atom, a Nx3 integer matrix
        sclCenters: centers of molecules, in fractional coordination
        rltSclPos: positions of atoms relative to molecule centers, in fractional coodination
        To define a molecule crystal correctly, you must set offSets OR sclCenters and rltSclPos.
        """
        self.info = info
        self.partition = tuple([list(p) for p in partition])
        if offSets is not None:
            self.numbers, self.cell, self.sclPos, self.offSets = list(map(np.array, [numbers, cell, sclPos, offSets]))
            # self.partition = tuple(partition)
            assert len(numbers) == len(sclPos) == len(offSets)
            numAts = len(numbers)
            indSet = set(list(reduce(lambda x, y: x+y, partition)))
            assert indSet == set(range(numAts))
            self.dispPos = self.sclPos + self.offSets

            sclCenters, rltSclPos = zip(*[self.get_center_and_rltSclPos(indices) for indices in partition])
            sclCenters = np.array(sclCenters)
            self.sclCenters = sclCenters - np.floor(sclCenters)
            self.centers = np.dot(self.sclCenters, self.cell)
            self.rltSclPos = rltSclPos
            self.rltPos = [np.dot(pos, self.cell) for pos in rltSclPos]

        elif sclCenters is not None and rltSclPos is not None:
            # self.numbers, self.cell, self.sclPos = list(map(np.array, [numbers, cell, sclPos]))
            # self.partition = tuple(partition)
            # assert len(numbers) == len(sclPos)
            self.numbers, self.cell = list(map(np.array, [numbers, cell]))
            # sclPos should be calculated from sclCenters and rltSclPos
            self.sclPos = np.zeros((len(numbers), 3))
            numAts = len(numbers)
            indSet = set(list(reduce(lambda x, y: x+y, partition)))
            assert indSet == set(range(numAts))

            self.sclCenters = sclCenters - np.floor(sclCenters)
            self.centers = np.dot(self.sclCenters, self.cell)
            self.rltSclPos = rltSclPos
            self.rltPos = [np.dot(pos, self.cell) for pos in rltSclPos]
            self.update()


        else:
            raise RuntimeError("Need to set offsets or (sclCenters and rltSclPos)!")

        self.numAts = len(numbers)
        self.numMols = len(partition)
        self.molNums = [self.numbers[p].tolist() for p in self.partition]

    def get_center_and_rltSclPos(self, indices):
        """
        Return centers and scaled relative positions for each molecule
        """
        molPos = self.dispPos[indices]
        center = molPos.mean(0)
        rltPos = molPos - center
        return center, rltPos

    def get_sclCenters(self):
        """
        Return scaled relative positions for each molecule
        """
        return self.sclCenters[:]

    def get_centers(self):
        """
        Return centers for each molecule
        """
        return self.centers[:]

    def get_rltPos(self):
        """
        Return relative positions for each molecule
        """
        return self.rltPos[:]

    def get_radius(self):
        """
        Return radius for each molecule
        """
        radius = []
        for pos in self.rltPos:
            rmax = max(np.linalg.norm(pos, axis=1))
            radius.append(rmax)
        return radius

    def copy(self):
        """
        Return a copy of current object
        """
        return MolCryst(self.numbers, self.cell, self.sclPos, self.partition, sclCenters=self.sclCenters, rltSclPos=self.rltSclPos, info=self.info.copy())

    def to_dict(self, infoCopy=True):
        """
        Return a dictionary containing all properties of the current molecular crystal
        """
        molDict = {
            'numbers': self.numbers,
            'cell': self.cell,
            'sclPos': self.sclPos,
            'partition': self.partition,
            'sclCenters': self.sclCenters,
            'rltSclPos': self.rltSclPos,
        }
        if infoCopy:
            molDict['info'] = self.info.copy()
        return molDict

    def get_cell(self):
        """
        Return the cell
        """
        return self.cell[:]

    def get_numbers(self):
        """
        Return atomic numbers
        """
        return self.numbers[:]

    def get_scaled_positions(self):
        """
        Return scaled positions of all atoms
        """
        return self.sclPos[:]

    def get_volume(self):
        """
        Return the cell volume
        """
        return np.linalg.det(self.cell)

    def get_mols(self):
        """
        Return all the molecules as a list of ASE's Atoms objects
        """
        mols = []
        for n, indices in enumerate(self.partition):
            mol = Atoms(numbers=self.numbers[indices], positions=self.rltPos[n])
            mols.append(mol)
        return mols

    def set_cell(self, cell, scale_atoms=False, scale_centers=True):
        """
        Set the cell
        scale_atoms: scale all atoms (may change relative positions) or not
        scale_center: scale molecule centers (do not change relative positions) or not
        """
        self.cell = np.array(cell)
        if scale_atoms:
            self.centers = np.dot(self.sclCenters, self.cell)
            self.rltPos = [np.dot(pos, self.cell) for pos in self.rltSclPos]
        else:
            if scale_centers:
                self.centers = np.dot(self.sclCenters, self.cell)
            self.update_centers_and_rltPos(self.centers, self.rltPos)

    def to_atoms(self):
        """
        Return the molecular crystal as an ASE's Atoms object
        """
        return Atoms(numbers=self.numbers, scaled_positions=self.sclPos, cell=self.cell, pbc=1, info=self.info.copy())

    def update_centers_and_rltPos(self, centers=None, rltPos=None):
        """
        Set centers and relative positions in Cartesian coodination
        """
        invCell = np.linalg.inv(self.cell)
        if centers is not None:
            self.centers = np.array(centers)
            self.sclCenters = np.dot(self.centers, invCell)
        if rltPos is not None:
            self.rltPos = [np.array(pos) for pos in rltPos]
            self.rltSclPos = [np.dot(pos, invCell) for pos in self.rltPos]
        self.update()

    def update_sclCenters_and_rltSclPos(self, sclCenters=None, rltSclPos=None):
        """
        Set centers and relative positions in fractional coodination
        """

        if sclCenters is not None:
            self.sclCenters = np.array(sclCenters)
            self.centers = np.dot(self.sclCenters, self.cell)
        if rltSclPos is not None:
            self.rltSclPos = [np.array(pos) for pos in rltSclPos]
            self.rltPos = [np.dot(pos, self.cell) for pos in self.rltSclPos]
        self.update()

    def update(self):
        """
        Update centers and relative positions
        """
        molNum = len(self.partition)
        posList = [self.sclCenters[i]+self.rltSclPos[i] for i in range(molNum)]
        tmpSclPos = reduce(lambda x,y: np.concatenate((x,y), axis=0), posList)
        indices = list(reduce(lambda x, y: x+y, self.partition))
        self.sclPos[indices] = tmpSclPos


def atoms2molcryst(atoms, coef=1.1):
    """
    Convert crystal to molecular crystal
    atoms: (ASE.Atoms) the input crystal structure
    coef: (float) the criterion for connecting two atoms
    Return: MolCryst
    """
    QG = quotient_graph(atoms, coef)
    graphs = [QG.subgraph(comp).copy() for comp in nx.connected_components(QG)]
    partition = []
    offSets = np.zeros([len(atoms), 3])
    for G in graphs:
        if graph_dim(G) == 0 and G.number_of_nodes() > 1:
            nodes, offs = nodes_and_offsets(G)
            partition.append(nodes)
            for i, offSet in zip(nodes, offs):
                offSets[i] = offSet

        else:
            for i in G.nodes():
                partition.append([i])
                offSets[i] = [0,0,0]

    molC = MolCryst(numbers=atoms.get_atomic_numbers(), cell=atoms.get_cell(),
    sclPos=atoms.get_scaled_positions(), partition=partition, offSets=offSets, info=atoms.info.copy())

    return molC

def primitive_atoms2molcryst(atoms, coef=1.1):
    """
    Convert crystal to molecular crystal
    atoms: (ASE.Atoms) the input crystal structure
    coef: (float) the criterion for connecting two atoms
    Return: tags and offsets
    """
    QG = quotient_graph(atoms, coef)
    graphs = [QG.subgraph(comp).copy() for comp in nx.connected_components(QG)]
    partition = []
    offSets = np.zeros([len(atoms), 3])
    tags = np.zeros(len(atoms))
    for G in graphs:
        if graph_dim(G) == 0 and G.number_of_nodes() > 1:
            nodes, offs = nodes_and_offsets(G)
            partition.append(nodes)
            for i, offSet in zip(nodes, offs):
                offSets[i] = offSet
        else:
            for i in G.nodes():
                partition.append([i])
                offSets[i] = [0,0,0]

    for tag, p in enumerate(partition):
        for j in p:
            tags[j] = tag

    return tags, offSets