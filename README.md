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

`testopt.py` firstly finds the optimal quotient graph for the configuration with given coordination numbers (using the method proposed in https://link.aps.org/doi/10.1103/PhysRevB.97.014104). Then the script adjusts the atomic positions and lattice to make the structure fitted to the quotient graph.

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

### Optimize structure based quotient graph
``` shell
$ python testopt.py rand_1.cif
edge ratios: min: 1.3016713493158165, max: 1.8759146848507082, mean: 1.5974768198995881
Optimize the structure using methods in ASE:
          Step     Time          Energy         fmax
graphOpt:    0 19:54:11       17.854404       48.6020
graphOpt:    1 19:54:12        3.600196       16.1202
graphOpt:    2 19:54:12        0.915217        6.5040
graphOpt:    3 19:54:12        0.218044        2.4064
graphOpt:    4 19:54:12        0.059072        1.1275
graphOpt:    5 19:54:12        0.002781        0.2345
graphOpt:    6 19:54:12        0.000014        0.0166
graphOpt:    7 19:54:12        0.000000        0.0003
graphOpt:    8 19:54:12        0.000000        0.0000
graphOpt:    9 19:54:12        0.000000        0.0000
graphOpt:   10 19:54:13        0.000000        0.0000
graphOpt:   11 19:54:13        0.000000        0.0000
graphOpt:   12 19:54:13        0.000000        0.0000
graphOpt:   13 19:54:13        0.000000        0.0000
graphOpt:   14 19:54:13        0.000000        0.0000
graphOpt:   15 19:54:13        0.000000        0.0000
graphOpt:   16 19:54:13        0.000000        0.0000
graphOpt:   17 19:54:13        0.000000        0.0000
graphOpt:   18 19:54:14        0.000000        0.0000
graphOpt:   19 19:54:14        0.000000        0.0000
graphOpt:   20 19:54:14        0.000000        0.0000
graphOpt:   21 19:54:14        0.000000        0.0000
graphOpt:   22 19:54:14        0.000000        0.0000
graphOpt:   23 19:54:14        0.000000        0.0000
graphOpt:   24 19:54:14        0.000000        0.0000
graphOpt:   25 19:54:15        0.000000        0.0000
graphOpt:   26 19:54:15        0.000000        0.0000
graphOpt:   27 19:54:15        0.000000        0.0000
graphOpt:   28 19:54:15        0.000000        0.0000
graphOpt:   29 19:54:15        0.000000        0.0000
graphOpt:   30 19:54:15        0.000000        0.0000
graphOpt:   31 19:54:15        0.000000        0.0000
graphOpt:   32 19:54:15        0.000000        0.0000
graphOpt:   33 19:54:16        0.000000        0.0000
graphOpt:   34 19:54:16        0.000000        0.0000
graphOpt:   35 19:54:16        0.000000        0.0000
graphOpt:   36 19:54:16        0.000000        0.0000
graphOpt:   37 19:54:16        0.000000        0.0000
graphOpt:   38 19:54:16        0.000000        0.0000
graphOpt:   39 19:54:16        0.000000        0.0000
graphOpt:   40 19:54:17        0.000000        0.0000
graphOpt:   41 19:54:17        0.000000        0.0000
graphOpt:   42 19:54:17        0.000000        0.0000
graphOpt:   43 19:54:17        0.000000        0.0000
graphOpt:   44 19:54:17        0.000000        0.0000
Last loss function: 0.0
```

`rand_1.cif` is a random carbon structure. The shortest C-C distance is 1.97 A, which larger than the typical C-C bond length. In `testopt.py`, coordination numbers of all the atoms are set as 4. The newly generated file `end.vasp` is the final structure, in which all the coordination numbers become 4 after optimization.

## Citations
If you are referencing Pycqg in a publication, please cite the following paper:
- Gao, H., Wang, J., Guo, Z. et al. Determining dimensionalities and multiplicities of crystal nets. npj Comput Mater 6, 143 (2020). https://doi.org/10.1038/s41524-020-00409-0


## License
Pycqg is distributed under the terms of the [GNU Lesser General Public License 3](LICENSE).
