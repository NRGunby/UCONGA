import ring_lib
import molecule
import unittest
from random import shuffle
import warnings
from UCONGA_analyse import calc_min_rmsd

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
        ech_mol = molecule.from_cml('test_molecules/ethylcyclohexane_isospectral.cml')
        rs = ring_lib.find_ring_systems(ech_mol)[0]
        confs = ring_lib.find_ring_conformers(ech_mol, rs)
        self.assertEqual(len(confs), 2)

if __name__ == '__main__':
    unittest.main()