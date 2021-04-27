#!/usr/bin/env python
## Generate random structures
from ase.ga.startgenerator import StartGenerator
from ase import Atoms
from ase.ga.utilities import closest_distances_generator, CellBounds
import ase.io
import sys

if __name__ == "__main__":

    numConfig = int(sys.argv[1]) # number of random configurations

    if len(sys.argv) == 3:
        numAtoms = int(sys.argv[2]) # number of atoms in per configuration
    else:
        numAtoms = 20

    slab = Atoms('', pbc=True)
    blocks = [(6, numAtoms)]
    # blocks = ['C']*numAtoms
    blmin = {(6,6):2}
    volume = 15 * numAtoms
    cellbounds = CellBounds(bounds={'phi': [35, 145], 'chi': [35, 145],
                                'psi': [35, 145], 'a': [3, 50],
                                'b': [3, 50], 'c': [3, 50]})


    sg = StartGenerator(slab, blocks, blmin, box_volume=volume,
        number_of_variable_cell_vectors=3,
        test_too_far=False,cellbounds=cellbounds)

    for i in range(numConfig):
        ats = sg.get_new_candidate()
        ase.io.write('rand_{}.vasp'.format(i),ats,vasp5=1,direct=1)
