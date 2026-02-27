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
    # Cell (orthogonal box)
    # ----------------------------------------------------------

    def cell_shape(self):
        return [
            self.a * self.nx,
            1.5 * self.CC * self.ny,
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
            np.array([0.0, 0.0, 0.0]),                         # A
            np.array([sqrt(3.0)/2.0*CC, 0.5*CC, 0.0])          # B
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
    # Bonds (analytical periodic graphene topology)
    # ----------------------------------------------------------

    def assign_bonds(self, lattice_dimensions):

        bonds = []

        for i in range(self.nx):
            for j in range(self.ny):

                # Periodic wrapping
                im = (i - 1) % self.nx
                jm = (j - 1) % self.ny

                # Current cell atom indices (1-based indexing)
                base = 2 * (i * self.ny + j)
                A = base + 1
                B = base + 2

                # 1) Bond inside unit cell
                bonds.append([A, B])

                # 2) A → B in (i-1, j)
                base_im = 2 * (im * self.ny + j)
                bonds.append([A, base_im + 2])

                # 3) A → B in (i, j-1)
                base_jm = 2 * (i * self.ny + jm)
                bonds.append([A, base_jm + 2])

        return np.array(bonds, dtype=int)

    # ----------------------------------------------------------
    # Angles
    # ----------------------------------------------------------

    def assign_angles(self, lattice_dimensions):

        bonds = self.assign_bonds(lattice_dimensions)

        adjacency = {i+1: [] for i in range(self.natoms)}

        for b in bonds:
            adjacency[b[0]].append(b[1])
            adjacency[b[1]].append(b[0])

        angles = []

        for atom, neighbours in adjacency.items():
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
