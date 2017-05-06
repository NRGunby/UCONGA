import unittest
import UCONGA_generate
import molecule
from math import pi, degrees

scaling = 0.7
tolerance = 0.2617993877991494  # 15 degrees in radians
pce = 'test_molecules/unoptimised_perchloroethane.cml'
unoptimised_perchloroethane = molecule.from_cml(pce)

cl1 = unoptimised_perchloroethane.atoms[2]
cl_cis = unoptimised_perchloroethane.atoms[6]
cl_trans = unoptimised_perchloroethane.atoms[7]
good_pair = (cl1, cl_trans)
bad_pair = (cl1, cl_cis)

toluene = molecule.from_cml('test_molecules/tol.cml')
at1 = toluene.atoms[0]
at2 = toluene.atoms[1]
at7 = toluene.atoms[6]
at8 = toluene.atoms[7]
dbl_ring_bond = (at1, at2)
methyl_bond = (at2, at7)
h_bond = (at7, at8)

rotatable_mol = molecule.from_cml('test_molecules/rotatable_test.cml')
cis_trans = molecule.from_cml('test_molecules/cis_trans.cml')
ttbds = molecule.from_cml('test_molecules/tri-t-butyl-disilane.cml')
digauche = molecule.from_cml('test_molecules/methylbutane_digauche.cml')
monogauche_1 = molecule.from_cml('test_molecules/methylbutane_monogauche_1.cml')
monogauche_2 = molecule.from_cml('test_molecules/methylbutane_monogauche_2.cml')
linear_mol = molecule.from_cml('test_molecules/1_butyne.cml')

class TestChooseScaling(unittest.TestCase):
    def test_neopentane(self):
        m = molecule.from_cml('test_molecules/neopentane.cml')
        res = UCONGA_generate.choose_scaling(m)
        self.assertEqual(res, 0.7)

    def test_propane(self):
        m = molecule.from_cml('test_molecules/propane.cml')
        res = UCONGA_generate.choose_scaling(m)
        self.assertEqual(res, 0.9)

    def test_isobutane(self):
        m = molecule.from_cml('test_molecules/isobutane.cml')
        res = UCONGA_generate.choose_scaling(m)
        self.assertEqual(res, 0.8)

class TestGroupRotatable(unittest.TestCase):
    def test_very_big(self):
        big_mol = molecule.from_cml('test_molecules/bigmol.cml')
        grps = UCONGA_generate.group_rotatable_bonds(big_mol)
        res = sorted([sorted(i) for i in grps])
        self.assertEqual(res, [[(1, 5), (5, 8), (8, 11)],[(14, 17), (17, 20), (20, 23)]])

    def test_ring(self):
        ring_mol = molecule.from_cml('test_molecules/112triethylcyclopropane.cml')
        grps = UCONGA_generate.group_rotatable_bonds(ring_mol)
        res = sorted([sorted(i) for i in grps])
        self.assertEqual(res, [[(0, 5), (0, 6)],[(2, 13)]])

    def test_alkene(self):
        alkene_mol = molecule.from_cml('test_molecules/transdiethylethylene.cml')
        grps = UCONGA_generate.group_rotatable_bonds(alkene_mol)
        res = sorted([sorted(i) for i in grps])
        self.assertEqual(res, [[(0, 2)], [(1, 5)]])

class TestAttachRigid(unittest.TestCase):
    def test_ring(self):
        ring_mol = molecule.from_cml('test_molecules/112triethylcyclopropane.cml')
        grps = UCONGA_generate.group_rotatable_bonds(ring_mol)
        fragments = UCONGA_generate.attach_rigid_linkers(grps, ring_mol)
        res = sorted([sorted(i) for i in fragments])
        ref = [[23, 26, 27, 25, 7, 22, 24, 1, 6, 9, 10, 8, 11, 12, 13, 2, 4, 5, 3, 15],
               [1,  2, 4, 5, 3, 15, 14, 17, 18, 16, 19, 20, 21]]
        self.assertEqual(res, sorted([sorted([j - 1 for j in i]) for i in ref]))

def test_alkene(self):
    alkene_mol = molecule.from_cml('test_molecules/transdiethylethylene.cml')
    grps = UCONGA_generate.group_rotatable_bonds(alkene_mol)
    fragments = UCONGA_generate.attach_rigid_linkers(grps, alkene_mol)
    res = sorted([sorted(i) for i in fragments])
    ref = [[1, 2, 4, 5, 6, 7, 9, 8, 13, 14, 15],
           [1, 2, 4, 5, 3, 10, 11, 12, 16, 17, 18]]
    self.assertEqual(res, sorted([sorted([j - 1 for j in i]) for i in ref]))

class TestRecombineFragments(unittest.TestCase):
    def test_alkene_negative(self):
        alkene_mol = molecule.from_cml('test_molecules/transdiethylethylene.cml')
        grps = UCONGA_generate.group_rotatable_bonds(alkene_mol)
        fragments = UCONGA_generate.attach_rigid_linkers(grps, alkene_mol)
        group_sizes = [len(i) for i in grps]
        recombined_frags = UCONGA_generate.recombine_fragments(fragments, alkene_mol, group_sizes)
        res = sorted([sorted(i) for i in recombined_frags])
        ref = [[1, 2, 4, 5, 6, 7, 9, 8, 13, 14, 15],
               [1, 2, 4, 5, 3, 10, 11, 12, 16, 17, 18]]
        self.assertEqual(res, sorted([sorted([j - 1 for j in i]) for i in ref]))

    def test_alkene_positive(self):
        alkene_mol = molecule.from_cml('test_molecules/cisdiethylethylene.cml')
        grps = UCONGA_generate.group_rotatable_bonds(alkene_mol)
        fragments = UCONGA_generate.attach_rigid_linkers(grps, alkene_mol)
        group_sizes = [len(i) for i in grps]
        recombined_frags = UCONGA_generate.recombine_fragments(fragments, alkene_mol, group_sizes)
        res = sorted([sorted(i) for i in recombined_frags])
        ref = [[1, 2, 4, 5, 6, 7, 9, 8, 13, 14, 15, 3, 10, 11, 12, 16, 17, 18]]
        self.assertEqual(res, sorted([sorted([j - 1 for j in i]) for i in ref]))

    def test_cyclic_positive(self):
        ring_mol = molecule.from_cml('test_molecules/112triethylcyclopropane.cml')
        grps = UCONGA_generate.group_rotatable_bonds(ring_mol)
        fragments = UCONGA_generate.attach_rigid_linkers(grps, ring_mol)
        group_sizes = [len(i) for i in grps]
        recombined_frags = UCONGA_generate.recombine_fragments(fragments, ring_mol, group_sizes)
        res = sorted([sorted(i) for i in recombined_frags])
        ref = [[1, 2, 4, 5, 6, 7, 9, 8, 13, 14, 15, 3, 10, 11, 12, 16, 17, 18,
                19, 20, 21, 22, 23, 24, 25, 26, 27]]
        self.assertEqual(res, sorted([sorted([j - 1 for j in i]) for i in ref]))

    def test_cyclic_negative(self):
        ring_mol = molecule.from_cml('test_molecules/unmergable_ring.cml')
        grps = UCONGA_generate.group_rotatable_bonds(ring_mol)
        fragments = UCONGA_generate.attach_rigid_linkers(grps, ring_mol)
        group_sizes = [len(i) for i in grps]
        recombined_frags = UCONGA_generate.recombine_fragments(fragments, ring_mol, group_sizes)
        res = sorted([sorted(i) for i in recombined_frags])
        ref = [[1, 2, 3, 6, 7, 12, 9, 10, 4, 5, 8, 16, 17, 18],
                [1, 2, 3, 6, 7, 12, 9, 10, 4, 5, 11, 13, 14, 15]]
        self.assertEqual(res, sorted([sorted([j - 1 for j in i]) for i in ref]))

    def test_oversized_negative(self):
        ring_mol = molecule.from_cml('test_molecules/too_big_to_merge.cml')
        grps = UCONGA_generate.group_rotatable_bonds(ring_mol)
        fragments = UCONGA_generate.attach_rigid_linkers(grps, ring_mol)
        group_sizes = [len(i) for i in grps]
        recombined_frags = UCONGA_generate.recombine_fragments(fragments, ring_mol, group_sizes)
        self.assertEqual(len(recombined_frags), 2)

class TestFindOlderSiblings(unittest.TestCase):
    def test_older_sibling(self):
        classes = ttbds.get_morgan_equivalencies()
        elder_sibling_ids = {}
        for each_id, each_bond in enumerate(UCONGA_generate.find_rotatable_bonds(ttbds)):
            elder_sibling_ids[each_bond] = UCONGA_generate.find_older_sibling_ids(each_bond, each_id, ttbds, classes)
        for k, v in elder_sibling_ids.items():
            if k == (1, 0, 31, 32):
                self.assertEqual(v, ['1'])
            else:
                self.assertEqual(v, [])
class TestFindMaxAngles(unittest.TestCase):
    def angle_test(self, mol_name, expected_result):
        mol = molecule.from_cml(mol_name)
        backbone = mol.copy_without_H()[0]
        classes = backbone.get_morgan_equivalencies()
        centralness = [sum(i) for i in backbone.distances]
        rot_bonds = UCONGA_generate.find_rotatable_bonds(backbone)
        result = UCONGA_generate.find_max_angles(rot_bonds, centralness, backbone, classes, False)
        self.assertEqual(result, expected_result)

    def test_assym_2(self):
        self.angle_test('test_molecules/assymmetric_2_test.cml', [[360]])

    def test_assym_3(self):
        self.angle_test('test_molecules/assymmetric_3_test.cml', [[360]])

    def test_sym_3(self):
        self.angle_test('test_molecules/symmetric_3_test.cml', [[120]])

    def test_sym_2(self):
        self.angle_test('test_molecules/symmetric_2_test.cml', [[180]])

    def test_term_ring(self):
        self.angle_test('test_molecules/terminal_ring.cml', [[180]])

    def test_para_ring(self):
        self.angle_test('test_molecules/para_ring.cml', [[180]])

    def test_sat_para_ring(self):
        self.angle_test('test_molecules/saturated_para_ring.cml', [[360]])

    def test_nonrotor(self):
        self.angle_test('test_molecules/non_rotor.cml', [[360],['0'],[360]])

class TestTestPair(unittest.TestCase):
    def test_good_pair(self):
        result = UCONGA_generate.test_pair(good_pair, scaling)
        self.assertTrue(result)

    def test_bad_pair(self):
        result = UCONGA_generate.test_pair(bad_pair, scaling)
        self.assertFalse(result)


class TestTestMol(unittest.TestCase):

    def test_good_mol(self):
        result = UCONGA_generate.test_mol(toluene, scaling)
        self.assertTrue(result)

    def test_bad_mol(self):
        result = UCONGA_generate.test_mol(unoptimised_perchloroethane, scaling)
        self.assertFalse(result)


class TestTestRing(unittest.TestCase):

    def test_in_ring(self):
        result = UCONGA_generate.test_ring(*dbl_ring_bond)
        self.assertFalse(result)

    def test_not_in_ring(self):
        result = UCONGA_generate.test_ring(*methyl_bond)
        self.assertTrue(result)


class TestTestOrder(unittest.TestCase):

    def test_multiple(self):
        result = UCONGA_generate.test_order(*dbl_ring_bond)
        self.assertFalse(result)

    def test_single(self):
        result = UCONGA_generate.test_order(*methyl_bond)
        self.assertTrue(result)


class TestTestInteresting(unittest.TestCase):

    def test_h_bond(self):
        result = UCONGA_generate.test_interesting(*h_bond)
        self.assertFalse(result)

    def test_Me_bond(self):
        result = UCONGA_generate.test_interesting(*methyl_bond)
        self.assertFalse(result)

    def test_linear_bond(self):
        result = UCONGA_generate.test_interesting(linear_mol.atoms[6], linear_mol.atoms[8])
        self.assertFalse(result)

    def test_interesting_bond(self):
        result = UCONGA_generate.test_interesting(*dbl_ring_bond)
        self.assertTrue(result)


class TestFindRotatables(unittest.TestCase):
    def test_find_rotatables(self):
        result = UCONGA_generate.find_rotatable_bonds(rotatable_mol)
        self.assertEqual(len(result), 1)
        bond = result[0][1:3]
        self.assertItemsEqual(bond, (1, 6))


class TestMakeRotamer(unittest.TestCase):
    def test_simple(self):
        deltas = (pi/3.0,)
        rot = ((7, 1, 6, 15),)
        curr = rotatable_mol.get_torsion(*rot[0])
        new_rotamer = UCONGA_generate.make_rotamer(rotatable_mol, rot, deltas)
        self.assertEqual(curr, rotatable_mol.get_torsion(*rot[0]))
        self.assertAlmostEqual(pi/3.0,
                               new_rotamer.get_torsion(*rot[0]))

class TestGenerateAll(unittest.TestCase):
    def test_no_bonds(self):
        result = [i for i in UCONGA_generate.make_all_rotations(toluene, 60, 0.7, 15)]
        self.assertEqual(len(result), 1)

if __name__ == '__main__':
    unittest.main()