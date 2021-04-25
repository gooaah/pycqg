# Pycqg

A Python package for construction and analysis of crystal quotient graphs.

## Dependencies
- NumPy
- ASE
- NetworkX 2.5

## Usage of the script
First add the path of this project into `PYTHONPATH`.
``` shell
$ python analyze.py <StructureFile> <BondRatio>
```

`StructureFile` should be supported by `ASE`. `BondRatio` is the criterion to connect atoms. The default value is 1.1.

## Examples
Under `test/`
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


## Citations
If you are referencing Pycqg in a publication, please cite the following paper:
- Gao, H., Wang, J., Guo, Z. et al. Determining dimensionalities and multiplicities of crystal nets. npj Comput Mater 6, 143 (2020). https://doi.org/10.1038/s41524-020-00409-0


## License
Pycqg is distributed under the terms of the [GNU Lesser General Public License 3](LICENSE).
