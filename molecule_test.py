import molecule
import unittest
from math import pi


class TestRingFinding(unittest.TestCase):
    def test_at0_not_in_ring(self):
        mol = molecule.from_cml('test_molecules/methylcyclopropane.cml')
        self.assertEqual(len(mol.rings), 1)
        self.assertEqual(len(mol.rings[0]), 4)

    def test_spiro(self):
        mol = molecule.from_cml('test_molecules/spiro.cml')
        self.assertEqual(len(mol.rings), 2)
        self.assertEqual(len(mol.rings[0]), 4)
        self.assertEqual(len(mol.rings[1]), 4)

    def test_unconnected(self):
        mol = molecule.from_cml('test_molecules/unconnected_rings.cml')
        self.assertEqual(len(mol.rings), 3)
        self.assertEqual(len(mol.rings[0]), 4)
        self.assertEqual(len(mol.rings[1]), 4)
        self.assertEqual(len(mol.rings[2]), 4)

    def test_bridged(self):
        mol = molecule.from_cml('test_molecules/bridged.cml')
        self.assertEqual(len(mol.rings), 3)
        self.assertEqual(len(mol.rings[0]), 5)
        self.assertEqual(len(mol.rings[1]), 5)
        self.assertEqual(len(mol.rings[2]), 5)

    def test_fused(self):
        mol = molecule.from_cml('test_molecules/fused.cml')
        self.assertEqual(len(mol.rings), 3)
        c = [len(i) for i in mol.rings]
        self.assertEqual(c, [4, 4, 5])




class TestMolecule(unittest.TestCase):
    def setUp(self):
        self.mol = molecule.from_cml('test_molecules/benzene.cml')

    def test_heavy_valences(self):
        c_normal = self.mol.atoms[1]
        self.assertEqual(c_normal.get_heavy_valence(), 2)

    def test_get_distance(self):
        c_2 = self.mol.atoms[1]
        c_6 = self.mol.atoms[5]
        self.assertAlmostEqual(c_2.get_distance(c_6), 2.4159090205314)

    def test_valid_ids(self):
        self.assertTrue(self.mol.is_valid_id(11))
        self.assertTrue(self.mol.is_valid_id(0))
        self.assertTrue(self.mol.is_valid_id(7))

    def test_invalid_ids(self):
        self.assertFalse(self.mol.is_valid_id(12))
        self.assertFalse(self.mol.is_valid_id(-1))
        self.assertFalse(self.mol.is_valid_id(2.5))

    def test_all_bonds(self):
        a = [i for i in self.mol.all_bonds()]
        self.assertItemsEqual(a,
                              [(0, 1), (1, 4), (4, 5),
                               (3, 5), (2, 3), (0, 2),
                               (0, 8), (1, 9), (4, 7),
                               (5, 6), (3, 11), (2, 10)])

    def test_bond_order(self):
        self.assertEqual(self.mol.get_bond_order(0, 1), 1.5)
        self.assertEqual(self.mol.get_bond_order(6, 5), 1)
        self.assertEqual(self.mol.get_bond_order(2, 1), 0)

    def test_in_ring(self):
        self.assertTrue(self.mol.is_in_ring(0, 1))

    def test_not_in_ring(self):
        self.assertFalse(self.mol.is_in_ring(0, 8))

    def test_ring_non_bonded(self):
        self.assertTrue(self.mol.is_in_ring(0, 5))


class TestTorsions(unittest.TestCase):
    def setUp(self):
        self.mol = molecule.from_cml('test_molecules/unoptimised_perchloroethane.cml')

    def test_iterate(self):
        torsions = [i for i in self.mol.all_torsions()]
        self.assertEqual(len(torsions), 1)
        self.assertEqual(list(torsions[0][1:3]), [0, 1])

    def test_get_torsion(self):
        opposites = self.mol.get_torsion(2, 0, 1, 7)
        self.assertAlmostEqual(abs(opposites), pi, places=5)

    def test_torsion_signs(self):
        left = self.mol.get_torsion(2, 0, 1, 6)
        right = self.mol.get_torsion(2, 0, 1, 5)
        self.assertAlmostEqual(left, -1*right, places=5)

    def test_set(self):
        self.mol.set_torsion(2, 0, 1, 6, pi)
        with open('tmp.cml', 'w') as cml_file:
            cml_file.write(self.mol.to_cml())
        self.assertAlmostEqual(abs(self.mol.get_torsion(2, 0, 1, 6)),
                               pi, places=5)
        self.assertAlmostEqual(abs(self.mol.get_torsion(2, 0, 1, 5)),
                               pi/3.0, places=5)
        self.assertAlmostEqual(abs(self.mol.get_torsion(2, 0, 1, 7)),
                               pi/3.0, places=5)
        self.assertAlmostEqual(abs(self.mol.get_torsion(3, 0, 1, 6)),
                               pi/3.0, places=5)
        self.assertAlmostEqual(abs(self.mol.get_torsion(4, 0, 1, 6)),
                               pi/3.0, places=5)

class TestParastereochemistry(unittest.TestCase):
    def test_eq_dimer_ring_parastereochemistry(self):
        mol = molecule.from_cml('test_molecules/cis_cis_[[methylcyclobutyl]methyl,methyl]cyclobutane.cml').get_morgan_equivalencies()
        self.assertEqual(mol[0], mol[17])
        self.assertEqual(mol[5], mol[22])

    def test_uniq_dimer_ring_parastereochemistry(self):
        mol = molecule.from_cml('test_molecules/cis_trans_[[methylcyclobutyl]methyl,methyl]cyclobutane.cml').get_morgan_equivalencies()
        self.assertNotEqual(mol[0], mol[17])
        self.assertNotEqual(mol[5], mol[22])

    def test_uuuu_tetramethylcyclooctane(self):
        mol = molecule.from_cml('test_molecules/uuuu_tetramethylcyclobutane.cml').get_morgan_equivalencies()
        self.assertEqual(len(set([mol[0], mol[1], mol[2], mol[5]])), 1)

    def test_udud_tetramethylcyclooctane(self):
        mol = molecule.from_cml('test_molecules/udud_tetramethylcyclobutane.cml').get_morgan_equivalencies()
        self.assertEqual(len(set([mol[0], mol[1], mol[2], mol[5]])), 2)
        self.assertEqual(len(set([mol[0], mol[2]])), 1)
        self.assertEqual(len(set([mol[1], mol[5]])), 1)

class TestSymmetryUtilities(unittest.TestCase):
    def basic_morgan(self, mol):
        '''
        Get the symmetry classes without the stereochemistry
        '''
        all_ids_neighbours = [each.get_bond_ids() for each in mol.atoms]
        connectivities = [each.get_heavy_valence() for each in mol.atoms]
        initial_ranks = molecule.rank(connectivities)
        new_ranks = molecule.rank_until_converged(initial_ranks, all_ids_neighbours)
        pi_functionalities = [len(filter(lambda x: x > 1, i))
                              for i in mol.bonds]
        atomic_numbers = [each.num for each in mol.atoms]
        for atom_property in (pi_functionalities, atomic_numbers):
            init_ranks = [(r << 8) + a for r, a in
                          zip(new_ranks, atom_property)]
            new_ranks = molecule.rank_until_converged(init_ranks, all_ids_neighbours)
        return new_ranks
    
    def test_find_dbl(self):
        mol = molecule.from_cml('test_molecules/dbl_finding_test.cml')
        new_ranks = self.basic_morgan(mol)
        dbl_systems = [i for i in mol.find_dbl_systems(new_ranks)]
        self.assertEqual(len(dbl_systems), 3)
        self.assertItemsEqual([i[2:] for i in dbl_systems], [(4, 14),(5, 7),(12, 20)])

    def test_cumulene_rearrangement(self):
        base_mol = molecule.from_cml('test_molecules/dbl_finding_test.cml')
        sixless_range = range(len(base_mol.atoms))
        sixless_range.remove(6)
        mid_high = base_mol.copy_subset(sixless_range + [6])
        mid_low = base_mol.copy_subset([6] + sixless_range)
        for mol in (base_mol, mid_high, mid_low):
            # Fake the sym classes
            new_ranks = self.basic_morgan(mol)
            dbl_systems = [i for i in mol.find_dbl_systems(new_ranks)]
            self.assertEqual(len(dbl_systems), 3)
            self.assertItemsEqual([i[0] for i in dbl_systems], [1, 1, 2])

    def test_tetrahedral_assignment(self):
        mol = molecule.from_cml('test_molecules/tetrahedral_stereo_test.cml')
        new_ranks =self.basic_morgan(mol)
        tetrahedral_stereochem = mol.assign_tet_stereochem(new_ranks)
        self.assertEqual(tetrahedral_stereochem[0], 0)
        self.assertNotEqual(tetrahedral_stereochem[1], 0)
        self.assertNotEqual(tetrahedral_stereochem[3], 0)
        self.assertNotEqual(tetrahedral_stereochem[4], 0)
        self.assertNotEqual(tetrahedral_stereochem[1], tetrahedral_stereochem[4])
        self.assertEqual(tetrahedral_stereochem[3], tetrahedral_stereochem[4])

    def test_dbl_assignment(self):
        mol = molecule.from_cml('test_molecules/dbl_assignment_test.cml')
        new_ranks = self.basic_morgan(mol)
        dbl_stereochem = mol.assign_dbl_stereochem(new_ranks)
        self.assertEqual(dbl_stereochem[2], 0)
        self.assertEqual(dbl_stereochem[17], 0)
        self.assertNotEqual(dbl_stereochem[1], 0)
        self.assertNotEqual(dbl_stereochem[5], 0)
        self.assertNotEqual(dbl_stereochem[3], 0)
        self.assertNotEqual(dbl_stereochem[6], 0)
        self.assertNotEqual(dbl_stereochem[4], 0)
        self.assertNotEqual(dbl_stereochem[7], 0)
        self.assertEqual(dbl_stereochem[1], dbl_stereochem[3])
        self.assertEqual(dbl_stereochem[1], dbl_stereochem[5])
        self.assertNotEqual(dbl_stereochem[1], dbl_stereochem[4])

    def test_ring_para(self):
        mol = molecule.from_cml('test_molecules/ring_para_test.cml')
        new_ranks = self.basic_morgan(mol)
        para_stereochem = mol.assign_para_stereochem(new_ranks)
        self.assertEqual(para_stereochem[3], 0)
        self.assertEqual(para_stereochem[13], 0)
        self.assertNotEqual(para_stereochem[5], 0)
        self.assertNotEqual(para_stereochem[7], 0)
        self.assertNotEqual(para_stereochem[14], 0)
        self.assertNotEqual(para_stereochem[7], para_stereochem[14])
        self.assertNotEqual(para_stereochem[5], para_stereochem[7])

    def test_ring_connection_para(self):
        mol = molecule.from_cml('test_molecules/connected_ring_para_test.cml')
        new_ranks = self.basic_morgan(mol)
        para_stereochem = mol.assign_para_stereochem(new_ranks)
        self.assertNotEqual(para_stereochem[6], 0)
        self.assertNotEqual(para_stereochem[28], 0)
        self.assertEqual(para_stereochem[6], para_stereochem[28])

    def test_mixed_types_double_join_para(self):
        mol = molecule.from_cml('test_molecules/mixed_parastereocentres.cml')
        new_ranks = self.basic_morgan(mol)
        para_stereochem = mol.assign_para_stereochem(new_ranks)
        self.assertNotEqual(para_stereochem[0], 0)
        self.assertNotEqual(para_stereochem[33], 0)

    def test_bridged_ring(self):
        mol = molecule.from_cml('test_molecules/barrelene_para_test.cml')
        new_ranks = self.basic_morgan(mol)
        para_stereochem = mol.assign_para_stereochem(new_ranks)
        self.assertNotEqual(para_stereochem[1], 0)
        self.assertNotEqual(para_stereochem[5], 0)
        self.assertNotEqual(para_stereochem[7], 0)
        self.assertEqual(para_stereochem[1], para_stereochem[5])
        self.assertEqual(para_stereochem[1], para_stereochem[7])

    def test_multiple_rings(self):
        mol = molecule.from_cml('test_molecules/multiple_rings_para_test.cml')
        new_ranks = self.basic_morgan(mol)
        para_stereochem = mol.assign_para_stereochem(new_ranks)
        self.assertEqual(para_stereochem[0], 0)
        self.assertNotEqual(para_stereochem[2], 0)
        self.assertNotEqual(para_stereochem[12], 0)
        self.assertNotEqual(para_stereochem[19], 0)

    def test_multiple_systems(self):
        mol = molecule.from_cml('test_molecules/test_para_multiple_systems.cml')
        new_ranks = self.basic_morgan(mol)
        para_stereochem = mol.find_para_groups(new_ranks)
        self.assertItemsEqual([set([j.get_id() for j in i]) for i in para_stereochem], [set([1, 3]), set([8, 10])])

    def test_allene(self):
        mol = molecule.from_cml('test_molecules/para_allene_test.cml')
        new_ranks = self.basic_morgan(mol)
        para_stereochem = mol.find_para_groups(new_ranks)
        self.assertEqual(len(para_stereochem), 1)
        self.assertItemsEqual([i.get_id() for i in para_stereochem[0]], (6, 20))



if __name__ == '__main__':
    unittest.main()
