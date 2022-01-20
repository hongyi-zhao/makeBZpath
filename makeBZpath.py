#!/usr/bin/env python

# makeBZpath.py
# This python script automatically produces a list of explicit q points along a Brillouin zone (BZ) path.
# Such a list can be directly used in the input file of the matdyn.x binary; PHonon package of Quantum-Espresso (QE).
# The script parses the crystal structure found in the input file used to run the Quantum Espresso (QE) pw.x binary.
# The crystal structure is automatically detected and the suggested BZ path is produced with the approximate number of points requested by the user.
#
# You can use or distribut this script as you wish, without any garantee and under your entire responsability.
# Since it uses codes of others you should cite their work properly, as you can see in the comments bellow.
#
# If you improve this script, please, send me your improved version.
#
# Marcelo Falcão de Oliveira - University of São Paulo (marcelo.falcao@usp.br)

# How to use:
# $ python makeBZpath.py <input_filename> <arg1> <arg2>
# 
# - input_filename is the input file used with pw.x (QE)
# - arg1 is a boolean, true or false, if you want or not extra BZ points for the non time-reversal cases.
#        obs.: If the crystal does have reversal symetry your choice doesn't matter and no extra points are produced.
#              If the crystal does not have reversal symetry, by choosing false it will produce extra points.
# - arg2 is a natural number expressing the number of explicit points you want in your BZ path. The output will be an approximation.

# Output:
#              
# Imput file: >input_filename<
# Crystal with inversion symetry: >boolean<
# Bravais lattice: >lattice designation code<
# Conventional lattice vectors:
# >matrix of lattice vectors if present<
# Primitive lattice vectors:
# >matrix of primitive lattice vectors<
#
# ### Number of q points and explicit BZ path, just copy and paste in your matdyn.x input file ###
#
# >number of q points<
# >list of q points with a comment (!) when applicable: k-point label at the points of high symmetry (k-point path)<

import sys
import numpy
import re

# SeeK-path is a python module to obtain band paths in the Brillouin zone of crystal structures, developed by
# - Y. Hinuma, G. Pizzi, Y. Kumagai, F. Oba, I. Tanaka, Band structure diagram paths based on crystallography, Comp. Mat. Sci. 128, 140 (2017)
# Its essential library is spglib, developed by
# - A. Togo, I. Tanaka, Spglib: a software library for crystal symmetry search, arXiv:1808.01590 (2018)
# PyPI: https://pypi.org/project/seekpath/
# GitHub: https://github.com/giovannipizzi/seekpath
# To install: pip install seekpath
import seekpath

from collections import OrderedDict

symbols = {"H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16,
              "Cl": 17, "Ar": 18, "K": 19, "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30,
              "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36, "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42, "Tc": 43, "Ru": 44,
              "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50, "Sb": 51, "Te": 52, "I": 53, "Xe": 54, "Cs": 55, "Ba": 56, "La": 57, "Ce": 58,
              "Pr": 59, "Nd": 60, "Pm": 61, "Sm": 62, "Eu": 63, "Gd": 64, "Tb": 65, "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70, "Lu": 71, "Hf": 72,
              "Ta": 73, "W": 74, "Re": 75, "Os": 76, "Ir": 77, "Pt": 78, "Au": 79, "Hg": 80, "Tl": 81, "Pb": 82, "Bi": 83, "Po": 84, "At": 85, "Rn": 86,
              "Fr": 87, "Ra": 88, "Ac": 89, "Th": 90, "Pa": 91, "U": 92, "Np": 93, "Pu": 94, "Am": 95, "Cm": 96, "Bk": 97, "Cf": 98, "Es": 99, "Fm": 100,
              "Md": 101, "No": 102, "Lr": 103, "Rf": 104, "Db": 105, "Sg": 106, "Bh": 107, "Hs": 108, "Mt": 109, "Ds": 110, "Rg": 111, "Cn": 112, "Uut": 113,
              "Uuq": 114, "Uup": 115, "Uuh": 116, "Uus": 117, "Uuo": 118
            }


# This function is used to read and parse the crystal structure from the input pw.x file.
# It was modified (mainly shortned) from its parent function found in the Phonopy code, developed by
# Atsushi Togo and Isao Tanaka, First principles phonon calculations in materials science, Scr. Mater., 108, 1-5 (2015).
# Website: https://phonopy.github.io/phonopy/
def read_pwscf(filename):
    """Read crystal structure."""
    
    with open(filename) as f:
        pwscf_in = PwscfIn(f.readlines())
    tags = pwscf_in.get_tags()
    lattice = tags["cell_parameters"]
    positions = [pos[1] for pos in tags["atomic_positions"]]
    species = [pos[0] for pos in tags["atomic_positions"]]
    unique_species = []

    for x in species:
        if x not in unique_species:
            unique_species.append(x)

    numbers = []
    is_unusual = False
    for x in species:
        if x in symbols:
            numbers.append(symbols[x])
        else:
            numbers.append(-unique_species.index(x))
            is_unusual = True

    if is_unusual:
        positive_numbers = []
        for n in numbers:
            if n > 0:
                if n not in positive_numbers:
                    positive_numbers.append(n)

        available_numbers = list(range(1, 119))
        for pn in positive_numbers:
            available_numbers.remove(pn)

        for i, n in enumerate(numbers):
            if n < 1:
                numbers[i] = available_numbers[-n]

    return lattice, positions, numbers


# This class is used to parse the crystal structure from the input pw.x file.
# It was modified (mainly shortned) from its parent class found in the Phonopy code, developed by
# Atsushi Togo and Isao Tanaka, First principles phonon calculations in materials science, Scr. Mater., 108, 1-5 (2015).
# Website: https://phonopy.github.io/phonopy/
class PwscfIn: 
    """Class to create QE input file."""

    _set_methods = OrderedDict(
        [
            ("ibrav", "_set_ibrav"),
            ("celldm(1)", "_set_celldm1"),
            ("nat", "_set_nat"),
            ("ntyp", "_set_ntyp"),
            ("atomic_species", "_set_atom_types"),
            ("atomic_positions", "_set_positions"),
            ("cell_parameters", "_set_lattice")
        ]
    )

    def __init__(self, lines):
        """Init method."""
        self._tags = {}
        self._current_tag_name = None
        self._values = None
        self._collect(lines)

    def get_tags(self):
        """Return tags."""
        return self._tags

    def _collect(self, lines):
        elements = {}
        tag_name = None

        for line in lines:
            _line = line.split("!")[0]
            if (
                "atomic_positions" in _line.lower()
                or "cell_parameters" in _line.lower()
            ):
                if len(_line.split()) == 1:
                    raise RuntimeError(
                        "A unit has to be specified for %s." % _line.strip()
                    )
                else:
                    words = _line.split()[:2]
            elif "atomic_species" in _line.lower():
                words = _line.split()
            else:  # other tag names and values
                line_replaced = _line.replace("=", " ").replace(",", " ")
                words = line_replaced.split()

            for val in words:
                if val.lower() in self._set_methods:  # tag name
                    tag_name = val.lower()
                    elements[tag_name] = [
                        val,
                    ]
                elif tag_name is not None:  # Ensure some tag name is set.
                    elements[tag_name].append(val)

        # Check if some necessary tag_names exist in elements keys.
        for tag_name in ["ibrav", "nat", "ntyp"]:
            if tag_name not in elements:
                raise RuntimeError("%s is not found in the input file." % tag_name)

        # Set values in self._tags[tag_name]
        for tag_name in self._set_methods:
            if tag_name in elements:
                self._current_tag_name = elements[tag_name][0]
                self._values = elements[tag_name][1:]
                if tag_name in self._set_methods.keys():
                    getattr(self, self._set_methods[tag_name])()

    def _set_ibrav(self):
        ibrav = int(self._values[0])
        if ibrav != 0:
            raise RuntimeError("Only %s = 0 is supported." % self._current_tag_name)
        self._tags["ibrav"] = ibrav

    def _set_celldm1(self):
        self._tags["celldm(1)"] = float(self._values[0])

    def _set_nat(self):
        self._tags["nat"] = int(self._values[0])

    def _set_ntyp(self):
        self._tags["ntyp"] = int(self._values[0])

    def _set_lattice(self):
        """
        Calculate and set lattice parameters.
        Invoked by CELL_PARAMETERS tag_name.
        """
        if len(self._values[1:]) < 9:
            raise RuntimeError("%s is wrongly set." % self._current_tag_name)

        lattice = numpy.reshape([float(x) for x in self._values[1:10]], (3, 3))
        self._tags["cell_parameters"] = lattice

    def _set_positions(self):
        unit = self._values[0].lower()
        if "crystal" not in unit:
            raise RuntimeError(
                "Only ATOMIC_POSITIONS format with " "crystal coordinates is supported."
            )

        natom = self._tags["nat"]
        pos_vals = self._values[1:]
        if len(pos_vals) < natom * 4:
            raise RuntimeError("ATOMIC_POSITIONS is wrongly set.")

        positions = []
        for i in range(natom):
            positions.append(
                [pos_vals[i * 4], [float(x) for x in pos_vals[i * 4 + 1 : i * 4 + 4]]]
            )

        self._tags["atomic_positions"] = positions

    def _set_atom_types(self):
        num_types = self._tags["ntyp"]
        if len(self._values) < num_types * 3:
            raise RuntimeError("%s is wrongly set." % self._current_tag_name)

        species = []

        for i in range(num_types):
            species.append(
                [
                    self._values[i * 3],
                    float(self._values[i * 3 + 1]),
                    self._values[i * 3 + 2],
                ]
            )

        self._tags["atomic_species"] = species


#######################################################
#           Main Script
#######################################################

# reads the cell parameters, atomic positions and atomic species
cell, positions, numbers = read_pwscf(sys.argv[1])
# imput for the seekpath functions
structure=[cell,positions,numbers]

# Check whether the user wants extra points or not
if sys.argv[2] == 'true':
    choice = True 
else:
    choice = False 

# used for crytal identification
result = seekpath.get_path(structure,choice)

# primary output
print ("Imput file: " + sys.argv[1])
print ("Crystal with inversion symetry: " + str(result.get('has_inversion_symmetry')))
print ("Bravais lattice: " + result.get('bravais_lattice_extended'))
print ("Conventional lattice vectors:")
print (result.get('cont_lattice'))
print ("Primitive lattice vectors:")
print (result.get('primitive_lattice'))

# large step just to run get_explicit_k_path in order to get the minimum possible number of points (k-points path)
step = 100 
result = seekpath.get_explicit_k_path(structure,choice,step)
# number of k-points (minus 1 in order to get the path length)
last = len(result.get('explicit_kpoints_rel')) - 1
length = result.get('explicit_kpoints_linearcoord')[last]
# total points requested by the user
totalpoints = int(sys.argv[3])

# if the user is requesting a number equal or bellow zero sets to the minimum
if totalpoints <= 0:
    totalpoints = last + 1

# path step size in order to generate an approximate number of requested q points  
step = length / totalpoints
result = seekpath.get_explicit_k_path(structure,choice,step)

# refresh the total of points
totalpoints = (len(result.get('explicit_kpoints_rel')))

# main output
print (" ")
print ("### Number of q points and explicit BZ path, just copy and paste in your matdyn.x input file ###")
print (" ")
print (totalpoints)

for i in range(0,totalpoints):
    # check if the point is a special k-point and prints its label as a comment
    label = re.sub(r'(\S+)', r' ! \1', result.get('explicit_kpoints_labels')[i])
    
    for k in result.get('explicit_kpoints_rel')[i]:
        print("{:1.10f}".format(k), end=" ")
    
    print (label)
