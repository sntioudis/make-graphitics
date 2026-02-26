from math import sqrt
import numpy as np
from .base import Molecule


class Hexagon_Graphene_Sheet(Molecule):

    def __init__(self, nx, ny, forcefield="OPLS"):

        config = self.crystal_params()

        self.CC = config[forcefield]["CC"]
        self.layer_gap = config[forcefield]["layer_gap"]

        self.nx = nx
        self.ny = ny

        self.a = sqrt(3.0) * self.CC

        self.natoms = 2 * self.nx * self.ny

    # ----------------------------------------------------------
    # Cell
    # ----------------------------------------------------------

    def cell_shape(self):
        return [
            self.a * self.nx,
            self.a * sqrt(3.0)/2.0 * self.ny,
            self.layer_gap
        ]

    # ----------------------------------------------------------
    # Coordinates
    # ----------------------------------------------------------

    def cell_coords(self):

        coords = []

        CC = self.CC

        a1 = np.array([sqrt(3.0)*CC, 0.0, 0.0])
        a2 = np.array([sqrt(3.0)/2.0*CC, 3.0/2.0*CC, 0.0])

        basis = [
            np.array([0.0, 0.0, 0.0]),
            np.array([sqrt(3.0)/2.0*CC, 0.5*CC, 0.0])
        ]

        for i in range(self.nx):
            for j in range(self.ny):
                R = i*a1 + j*a2
                for b in basis:
                    coords.append(R + b)

        return np.array(coords)

    # ----------------------------------------------------------
    # Molecule labels
    # ----------------------------------------------------------

    def assign_molecules(self, lattice_dimensions):
        molecule_labels = []
        for z in range(lattice_dimensions[2]):
            labels = np.ones(self.natoms, dtype=int)
            molecule_labels.extend(list(labels + z))
        return molecule_labels

    # ----------------------------------------------------------
    # Atom types
    # ----------------------------------------------------------

    def assign_atom_labels(self, lattice_dimensions):
        atom_labels = []
        for z in range(lattice_dimensions[2]):
            atom_labels += [1] * self.natoms
        return atom_labels

    # ----------------------------------------------------------
    # Charges
    # ----------------------------------------------------------

    def assign_atom_charges(self, lattice_dimensions, q):
        atom_charges = []
        for z in range(lattice_dimensions[2]):
            atom_charges += [0.0] * self.natoms
        return atom_charges

    # ----------------------------------------------------------
    # Bonds (analytical graphene connectivity)
    # ----------------------------------------------------------

    def assign_bonds(self, lattice_dimensions):

        coords = self.cell_coords()
        box = self.cell_shape()

        bonds = []
        cutoff = 1.1 * self.CC

        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):

                dx = coords[i][0] - coords[j][0]
                dy = coords[i][1] - coords[j][1]

                # Periodic boundary conditions
                dx -= box[0] * round(dx / box[0])
                dy -= box[1] * round(dy / box[1])

                dist = np.sqrt(dx*dx + dy*dy)

                if dist < cutoff:
                    bonds.append([i + 1, j + 1])

        return np.array(bonds, dtype=int)

    # ----------------------------------------------------------
    # Angles (each carbon has 3 neighbours)
    # ----------------------------------------------------------

    def assign_angles(self, lattice_dimensions):

        angles = []
        bonds = self.assign_bonds(lattice_dimensions)

        # Build adjacency list
        adjacency = {i+1: [] for i in range(self.natoms)}

        for b in bonds:
            adjacency[b[0]].append(b[1])
            adjacency[b[1]].append(b[0])

        for atom in adjacency:
            neighbours = adjacency[atom]
            for i in range(len(neighbours)):
                for j in range(i+1, len(neighbours)):
                    angles.append([neighbours[i], atom, neighbours[j]])

        return np.array(angles, dtype=int)

    # ----------------------------------------------------------
    # No torsions needed
    # ----------------------------------------------------------

    def assign_dihedrals(self, lattice_dimensions):
        return np.empty((0, 4), dtype=int)

    def assign_impropers(self, lattice_dimensions):
        return np.empty((0, 4), dtype=int)

    # ----------------------------------------------------------
    # Types
    # ----------------------------------------------------------

    def connection_types(self):
        bond_types = [[1, 1]]
        angle_types = [[1, 1, 1]]
        dihedral_types = []
        improper_types = []

        return bond_types, angle_types, dihedral_types, improper_types
