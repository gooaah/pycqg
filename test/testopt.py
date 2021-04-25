 #!/usr/bin/env python
## Crystal Quotient Graph
from __future__ import print_function, division
import ase.io
from ase import Atoms
from ase.optimize import BFGS, FIRE
from ase.optimize.sciopt import SciPyFminBFGS, SciPyFminCG
from ase.constraints import UnitCellFilter, ExpCellFilter, StrainFilter
from pycqg.crystgraph import CrystalGraph, GraphGenerator, GraphCalculator, graph_embedding, edge_ratios
import networkx as nx
import numpy as np
import sys, time
# from scipy.optimize import minimize


if __name__ == "__main__":

    filename = sys.argv[1]
    atoms = ase.io.read(filename)
    oriAts = atoms[:]

    if len(sys.argv) == 3:
        embed = sys.argv[2]
    else:
        embed = 0



    gen = GraphGenerator()
    # randG = gen.build_graph(atoms, [2,2,2,2,4,4])
    randG = gen.build_graph(atoms, [4]*len(atoms)) # set all the coordination numbers as 4
    # randG = gen.build_graph(atoms, [6]*len(atoms))

    ratios = edge_ratios(randG)
    print("edge ratios: min: {}, max: {}, mean: {}".format(ratios.min(), ratios.max(), ratios.mean()))

    # Embeding or not
    if embed:
        print("embeding graph")
        stardPos, randG = graph_embedding(randG)
        atoms.set_scaled_positions(stardPos)
    ase.io.write('start.vasp', atoms, vasp5=1, direct=1)

    calc = GraphCalculator(k1=2, k2=2, cmax= 1.2, cmin=.7, cadd=0.1)
    atoms.set_calculator(calc)
    atoms.info['graph'] = randG.copy()

    class graphOpt(BFGS):
        def converged(self, forces=None):
            if forces is None:
                forces = self.atoms.get_forces()
            return (forces**2).sum(axis=1).max() <= self.fmax**2

    print("Optimize the structure using methods in ASE:")
    # opt = BFGS(StrainFilter(atoms))
    # opt = BFGS(ExpCellFilter(atoms))
    # opt = SciPyFminCG(ExpCellFilter(atoms))
    # opt = graphOpt(StrainFilter(atoms))
    opt = graphOpt(ExpCellFilter(atoms),maxstep=0.2)
    # opt.emax = 1e-3
    # opt = graphOpt(ExpCellFilter(atoms))
    opt.run(fmax=0, steps=500)
    # # # opt = BFGS(atoms, maxstep=0.5)
    # opt.emax = 1e-3
    # opt.run(fmax=0.05, steps=50)

    print("Last loss function: {}".format(atoms.get_potential_energy()))
    ase.io.write('end.vasp', atoms, vasp5=1, direct=1)


