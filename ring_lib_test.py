import ring_lib
import molecule
import unittest
from random import shuffle
import warnings
from UCONGA_analyse import calc_min_rmsd
from numpy import dot

idx_conformer_testing = 0

class TestFindSystems(unittest.TestCase):
    def test_spiro(self):
        spiro_mol = molecule.from_cml('test_molecules/spiro.cml')
        ring_systems = ring_lib.find_ring_systems(spiro_mol)
        self.assertEqual(len(ring_systems), 2)

    def test_fused(self):
        fused_mol = molecule.from_cml('test_molecules/fused.cml')
        ring_systems = ring_lib.find_ring_systems(fused_mol)
        self.assertEqual(len(ring_systems), 1)

    def test_bridged(self):
        bridged_mol = molecule.from_cml('test_molecules/bridged.cml')
        ring_systems = ring_lib.find_ring_systems(bridged_mol)
        self.assertEqual(len(ring_systems), 1)


class TestFindConformers(unittest.TestCase):

    def test_not_present(self):
        b_mol = molecule.from_cml('test_molecules/benzene.cml')
        rs = ring_lib.find_ring_systems(b_mol)[0]
        confs = ring_lib.find_ring_conformers(b_mol, rs)
        self.assertEqual(len(confs), 1)

    def test_ethylcyclohexane(self):
        each_mol = molecule.from_cml('test_molecules/ethylcyclohexane_isospectral.cml')
        rs = ring_lib.find_ring_systems(each_mol)[0]
        confs = ring_lib.find_ring_conformers(each_mol, rs)
        self.assertEqual(len(confs), 2)

class TestFindPlane(unittest.TestCase):

    def test_planar_ring(self):
        m = molecule.from_cml('test_molecules/benzene.cml')
        m_ring = m.copy_without_H()[0]
        m_ring.center()
        m_norm = ring_lib.find_plane_of_ring(m_ring.coord_matrix())
        res = sum([dot(i.coords, m_norm) for i in m_ring.atoms])
        self.assertAlmostEqual(res, 0)

    def test_nonplanar_ring(self):
        m = molecule.from_cml('test_molecules/ethylcyclohexane_isospectral.cml')
        m_ring = m.copy_subset([0, 1, 2, 5, 8, 13])
        m_ring.center()
        m_norm = ring_lib.find_plane_of_ring(m_ring.coord_matrix())
        res = sum([dot(i.coords, m_norm) for i in m_ring.atoms])
        self.assertAlmostEqual(res, 0)

class TestIsFlippable(unittest.TestCase):

    def test_3membered_ring(self):
        m = molecule.from_cml('test_molecules/methylcyclopropane.cml')
        res = ring_lib.is_flippable(m, [1, 5, 8])
        self.assertFalse(res)

    def test_aromatic(self):
        m = molecule.from_cml('test_molecules/benzene.cml')
        res = ring_lib.is_flippable(m, [0, 1, 2, 3, 4, 5])
        self.assertFalse(res)

    def test_flippable(self):
        m = molecule.from_cml('test_molecules/ethylcyclohexane_isospectral.cml')
        res = ring_lib.is_flippable(m, [0, 1, 2, 5, 8, 13])
        self.assertTrue(res)

class TestFlipRing(unittest.TestCase):

    def test_is_flipped(self):
        m = molecule.from_cml('test_molecules/chiral_ring.cml')
        old_torsion = m.get_torsion(3, 1, 0, 17)
        flipped_m = ring_lib.flip_ring(m, [0, 2, 5, 8, 11, 1])
        flipped_torsion = flipped_m.get_torsion(3, 1, 0, 17)
        self.assertAlmostEqual(old_torsion, -1*flipped_torsion, 1)

    def test_idempotent(self):
        m = molecule.from_cml('test_molecules/chiral_ring.cml')
        flipped_m = ring_lib.flip_ring(m, [0, 2, 5, 8, 11, 1])
        m_again = ring_lib.flip_ring(flipped_m, [0, 2, 5, 8, 11, 1])
        m.center()
        m_again.center()
        res = calc_min_rmsd(m.coord_matrix(), m_again.coord_matrix())
        self.assertAlmostEqual(res, 0)

    def test_preserve_stereochem(self):
        m = molecule.from_cml('test_molecules/chiral_ring.cml')
        old_stereo = m.get_morgan_equivalencies()
        flipped_m = ring_lib.flip_ring(m, [0, 2, 5, 8, 11, 1])
        flipped_stereo = flipped_m.get_morgan_equivalencies()
        self.assertEqual(old_stereo, flipped_stereo)

if __name__ == '__main__':
    unittest.main()
