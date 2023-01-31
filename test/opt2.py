#!/usr/bin/env python
## Crystal Quotient Graph
from __future__ import print_function, division
import ase.io
from ase import Atoms
from ase.optimize import BFGS, FIRE
from ase.optimize.sciopt import SciPyFminBFGS, SciPyFminCG
from ase.constraints import UnitCellFilter, ExpCellFilter, StrainFilter
from pycqg.crystgraph import graph_embedding, edge_ratios
from pycqg.calculator import GraphCalculator
from pycqg.generator import GraphGenerator
from pycqg.generator import quot_gen
from pymatgen.io.cif import CifParser
import networkx as nx
import numpy as np
import sys, time
# from scipy.optimize import minimize

approach = input('1:MinimumDistanceNN; 2:VoronoiNN; 3:CN == 4: ')

if __name__ == "__main__":

    filename = sys.argv[1]
    atoms = ase.io.read(filename)
    parser = CifParser(filename)
    struct = parser.get_structures()[0]
    oriAts = atoms[:]

    if len(sys.argv) == 3:
        embed = sys.argv[2]
    else:
        embed = 0


    # randG = gen.build_graph(atoms, [2,2,2,2,4,4])
    if approach == '1':
        randG = quot_gen(atoms, struct, approach='min_dist', delta=0.1, add_ratio=True)
    if approach == '2':
        randG = quot_gen(atoms, struct, approach='voronoi', delta=0.6, add_ratio=True)
    if approach == '3':
        gen = GraphGenerator()
        randG = gen.build_graph(atoms, [4]*len(atoms))

    if randG is not None:
        ratios = edge_ratios(randG)
        print("edge ratios: min: {}, max: {}, mean: {}".format(ratios.min(), ratios.max(), ratios.mean()))
    else:
        print("Fail in generating graph!")
        sys.exit()

    # Embeding or not
    if embed:
        print("embeding graph")
        stardPos, randG = graph_embedding(randG)
        atoms.set_scaled_positions(stardPos)
    ase.io.write('start.vasp', atoms, vasp5=1, direct=1)

    calc = GraphCalculator(k1=2, k2=2, cmax= 1.2, cmin=.7, cunbond=1.3)
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
    opt = graphOpt(ExpCellFilter(atoms),maxstep=1)
    # opt.emax = 1e-3
    # opt = graphOpt(ExpCellFilter(atoms))
    opt.run(fmax=0, steps=200)
    # # # opt = BFGS(atoms, maxstep=0.5)
    # opt.emax = 1e-3
    # opt.run(fmax=0.05, steps=50)

    print("Last loss function: {}".format(atoms.get_potential_energy()))
    ase.io.write('end.vasp', atoms, vasp5=1, direct=1)


