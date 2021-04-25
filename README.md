# Pycqg

A Python package for construction and analysis of crystal quotient graphs.

## Dependencies
- NumPy
- ASE
- NetworkX 2.5

## Install

``` shell
$ python setup.py install
```

## Usage of the scripts

Currently, there are two scripts in `test/`: `analyze.py` and `testopt.py`.

`analyze.py` can compute dimensions and multiplicities for all the components in the given structure.

``` shell
$ python analyze.py <StructureFile> <BondRatio>
```

`StructureFile` should be supported by `ASE`. `BondRatio` is the criterion to connect atoms. The default value is 1.1.

`testopt.py` firstly finds the optimal quotient graph for the configuration with given coordination numbers (using the method proposed in https://link.aps.org/doi/10.1103/PhysRevB.97.014104). Then the script optimizes the atomic positions and lattice to make the structure fitted to the quotient graph.

``` shell
$ python testopt.py <StructureFile> <Embed>
```
`Embed` decides whether the quotient graph is embeded into real space to generate initial structures. The default value is 0 (do not embed).

## Examples
The structure files are in `test/`.

### Mix-dimensional
The structure `COD_7027514.cif` contains 2D, 1D, and 0D components.
``` shell
$ python analyze.py COD_7027514.cif
number of bonds: 80
Number of parallel edges: 0
Number of compenents: 7
Component       Dimension
1               2
2               1
3               1
4               0
5               0
6               0
7               0
Max Dimension: 2

Calculating Multplicities...
Component       Multiplicity
1               1
2               1
3               1
4               1
5               1
6               1
7               1
```

### Self-penetration
The structure `Cu2O.cif` contains two translationally equivalent but disconnected subnets, so its multiplicity is 2.

``` shell
$ python analyze.py Cu2O.cif
number of bonds: 8
Number of parallel edges: 0
Number of compenents: 1
Component       Dimension
1               3
Max Dimension: 3

Calculating Multplicities...
Component       Multiplicity
1               2
```

### Optimize structure based on quotient graph
``` shell
$ python testopt.py rand_1.cif
edge ratios: min: 1.3016713493158165, max: 1.8759146848507082, mean: 1.5974768198995881
Optimize the structure using methods in ASE:
          Step     Time          Energy         fmax
graphOpt:    0 20:19:22       17.854404       48.6020
graphOpt:    1 20:19:22        2.999464       16.4721
graphOpt:    2 20:19:23        0.087089        1.2450
graphOpt:    3 20:19:23        0.041000        0.8320
graphOpt:    4 20:19:23        0.006525        0.3127
graphOpt:    5 20:19:23        0.000692        0.1032
graphOpt:    6 20:19:23        0.000000        0.0000
Last loss function: 0.0
```

`rand_1.cif` is a random carbon structure. The shortest C-C distance is 1.97 A, which larger than the typical C-C bond length. In `testopt.py`, coordination numbers of all the atoms are set as 4. The newly generated file `end.vasp` is the final structure, in which all the coordination numbers become 4 after optimization.

## Citations
If you are referencing Pycqg in a publication, please cite the following paper:
- Gao, H., Wang, J., Guo, Z. et al. Determining dimensionalities and multiplicities of crystal nets. npj Comput Mater 6, 143 (2020). https://doi.org/10.1038/s41524-020-00409-0


## License
Pycqg is distributed under the terms of the [GNU Lesser General Public License 3](LICENSE).
