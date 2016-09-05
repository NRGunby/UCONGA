import linalg
import UCONGA_analyse
import unittest
import numpy
import molecule
from random import random


class TestRMSD(unittest.TestCase):
    def setUp(self):
        self.mol = molecule.from_cml('test_molecules/RR(orSS).cml')

    def test_self_rmsd(self):
        self_rmsd = UCONGA_analyse.calc_rmsd(self.mol.coord_matrix(), self.mol.coord_matrix())
        self.assertAlmostEqual(self_rmsd, 0)

    def test_self_min_rmsd(self):
        self_rmsd = UCONGA_analyse.calc_min_rmsd(self.mol.coord_matrix(), self.mol.coord_matrix())
        self.assertAlmostEqual(self_rmsd, 0)

    def test_self_mod_min_rmsd(self):
        b = self.mol.copy().coord_matrix()
        x, y, z = random(), random(), random()
        angle = random()
        rot_mat = linalg.rotation_axis_angle((x, y, z), angle)
        b = b.dot(rot_mat)
        self_rmsd = UCONGA_analyse.calc_min_rmsd(self.mol.coord_matrix(), b)
        self.assertAlmostEqual(self_rmsd, 0)


if __name__ == '__main__':
    unittest.main()
