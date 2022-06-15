from __future__ import print_function, division
from ase.neighborlist import neighbor_list
from ase.data import covalent_radii
import networkx as nx
import itertools
import numpy as np
from ase.calculators.calculator import Calculator, all_changes
# from .crystgraph import


class GraphCalculator(Calculator):
    """
    Calculator written for optimizing crystal structure based on an initial quotient graph.
    I design the potential so that the following criteria are fulfilled when the potential energy equals zero:
    Suppose there are two atoms A, B. R_{AB} = r_{A} + r_{B} is the sum of covalent radius.
    If A,B are bonded, cmin <= d_{AB}/R_{AB} <= cmax.
    If A,B are not bonded, cunbond <= d_{AB}/R_{AB}.
    """
    implemented_properties = ['energy', 'forces', 'stress']
    default_parameters = {
        'cmin': 0.5,
        'cmax': 1,
        'cunbond': 1.05,
        # 'cadd': 0.,
        'k1': 1e-1,
        'k2': 1e-1,
        # 'mode': 0,
        'useGraphRatio': False,
        'useGraphK': False,
        # 'ratioErr': 0.1,
        'ignorePairs': [], # pairs of ignored pairs of atomic numbers
    }
    def __init__(self, **kwargs):
        Calculator.__init__(self, **kwargs)

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        # Calculator.calculate(self, atoms, properties, system_changes)
        graph = atoms.info['graph'].copy()
        Nat = len(atoms)
        assert Nat == len(graph), "Number of atoms should equal number of nodes in the initial graph!"

        numbers = atoms.get_atomic_numbers()
        unqNumber = set(numbers.tolist()) # unique atomic numbers

        cmax = self.parameters.cmax
        cmin = self.parameters.cmin
        cunbond = self.parameters.cunbond
        # cadd = self.parameters.cadd
        k1 = self.parameters.k1
        k2 = self.parameters.k2
        useGraphRatio = self.parameters.useGraphRatio
        useGraphK = self.parameters.useGraphK
        # ratioErr = self.parameters.ratioErr
        ignorePairs = [tuple(sorted(p)) for p in self.parameters.ignorePairs]
        # mode = self.parameters.mode
        # cunbond = cmax + cadd

        # sums of radius
        radArr = covalent_radii[atoms.get_atomic_numbers()]
        radArr = np.expand_dims(radArr,1)
        rsumMat = radArr + radArr.T

        # set string constant
        if useGraphK:
            pass
        else:
            for edge in graph.edges(data=True, keys=True):
                m,n, key, data = edge
                graph[m][n][key]['k'] = k1


        # print(cadd)

        energy = 0.
        forces = np.zeros((Nat, 3))
        stress = np.zeros((3, 3))

        # print("Calling GraphCalculator.calculate")

        ## offsets for all atoms
        spos = atoms.get_scaled_positions(wrap=False)
        # remove little difference
        # TODO: The operator (set zero) hinders optimization of CoAs2_mp-2715. But it is important for AgF2_mp-7715. It needs more test.
        spos[np.abs(spos)<1e-8] = 0
        offsets = np.floor(spos).astype(np.int)
        offsets = np.zeros_like(spos)
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
            ## TODO: Not sure whether offset is crucial
            n1, n2, n3 = offsets[j] - offsets[i] + data['vector']
            # n1, n2, n3 = offsets[j] - offsets[i] + data['vector']

            # graph.edge[m][n][key]['vector'] = edgeVec
            graph[m][n][key]['vector'] = edgeVec
            pairs.append((i,j,n1,n2,n3))
            # rsum = covalent_radii[numbers[i]] + covalent_radii[numbers[j]]
            rsum = rsumMat[i,j]
            # Use ratio attached in graph or not
            if useGraphRatio:
                # ratio = data['ratio']
                # # cmax = ratio + ratioErr
                # cmax = ratio
                # cmin = ratio - ratioErr
                cmin, cmax = data['ratios']
                assert cmin < cmax, "cmin should be less than cmax!"
            Dmin = cmin * rsum
            Dmax = cmax * rsum
            # print("bond")
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
                    energy += 0.5*data['k']*(D-Dmin)**2
                    f = data['k']*(D-Dmin)*uvec
                elif D > Dmax:
                    # scalD = (D-Dmax)/rsum
                    # energy += 0.5*k1*scalD**2
                    # f = k1*scalD*uvec/rsum
                    energy += 0.5*data['k']*(D-Dmax)**2
                    f = data['k']*(D-Dmax)*uvec
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
        # copyAts = atoms.copy()
        # copyAts.wrap()
        # pos = copyAts.get_positions()
        pos = atoms.get_positions()
        ## Consider unbonded atom pairs
        # cutoffs = [covalent_radii[n]*cunbond for n in numbers]
        cutoffs = dict()
        for num1, num2 in itertools.combinations_with_replacement(unqNumber,2):
            if tuple(sorted([num1,num2])) not in ignorePairs:
                cutoffs[(num1,num2)] = cunbond*(covalent_radii[num1]+covalent_radii[num2])
        # unbondEn = 0
        ## TODO: I am still unsure whether atoms or copyAts should be used here. 
        for i, j, S in zip(*neighbor_list('ijS', atoms, cutoffs)):
        # for i, j, S in zip(*neighbor_list('ijS', copyAts, cutoffs)):
            if i <= j:
                # if i,j is bonded, skip this process
                n1,n2,n3 = S
                pair1 = (i,j,n1,n2,n3)
                pair2 = (j,i,-n1,-n2,-n3)
                if pair1 in pairs or pair2 in pairs:
                    # print("Skip")
                    continue
                # rsum = covalent_radii[numbers[i]] + covalent_radii[numbers[j]]
                rsum = rsumMat[i,j]
                Dmax = cunbond * rsum
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
                    # just for test
                    # unbondEn += 0.5*k2*(D-Dmax)**2
                    f = k2*(D-Dmax)*uvec
                    forces[i] += f
                    forces[j] -= f
                    stress += np.dot(f[np.newaxis].T, dvec[np.newaxis])
        
        # print("Unbond Energy: {}".format(unbondEn))


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