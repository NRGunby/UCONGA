import atom
import unittest
import numpy
import molecule


class TestUnattachedAtom(unittest.TestCase):
    
    def setUp(self):
        self.atom = atom.atom(1, 0, 0, 0)
    
    def test_atomic_number(self):
        self.assertEqual(self.atom.num, 1)
    
    def test_coords(self):
        for i, j in zip(self.atom.coords, [0.0, 0.0, 0.0]):
            self.assertEqual(i, j)

    def test_unattached_id(self):
        with self.assertRaises(ValueError):
            self.atom.get_id()

    def test_vdw(self):
        self.assertAlmostEqual(self.atom.get_vdw(), 1.1)
    
    def test_translate(self):
        vec = numpy.array([1.0, -1.0, 0.762])
        self.atom.translate(vec)
        for i, j in zip(self.atom.coords, vec):
            self.assertAlmostEqual(i, j)

    def test_rotate(self):
        new_at = atom.atom(1, 1, 0, 0)
        r_z = numpy.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        new_at.rotate(r_z)
        for i, j in zip(new_at.coords, [0, 1, 0]):
            self.assertAlmostEqual(i, j)

    def test_copy(self):
        self.new_atom = self.atom.copy()
        for i, j in zip(self.new_atom.coords, [0.0, 0.0, 0.0]):
            self.assertEqual(i, j)
        self.assertEqual(self.new_atom.num, 1)
        self.new_atom.num = 4
        self.assertEqual(self.new_atom.num, 4)
        self.assertEqual(self.atom.num, 1)

    def test_xyz(self):
        xyz_fragments = self.atom.to_xyz().split()
        self.assertEqual(xyz_fragments, ['H', '0', '0', '0'])

class TestAttachedAtom(unittest.TestCase):
    def setUp(self):
        mol = molecule.from_cml('test_molecules/tol.cml')
        mol.center()
        self.tol_atoms = mol.atoms

    def test_attached_id(self):
        self.assertEqual(range(len(self.tol_atoms)), [i.get_id() for i in self.tol_atoms])

    def test_bond_ids(self):
        #print 'About to test bond ids'
        self.assertEqual(self.tol_atoms[0].get_bond_ids(), [1, 2, 10])

    def test_hvy_valence(self):
        #print 'About to test heavy valence'
        self.assertEqual(self.tol_atoms[0].get_heavy_valence(), 2)
        self.assertEqual(self.tol_atoms[1].get_heavy_valence(), 3)

    def test_away_from(self):
        #print 'About to test away_from'
        self.assertEqual(self.tol_atoms[0].search_away_from(1), [2, 10])

    def test_all_away_from(self):
        #print 'About to test all_away_from'
        self.assertEqual(self.tol_atoms[1].all_neighbours_away_from(6),
                         [0, 2, 3, 4, 5, 10, 11, 12, 13, 14])

    def test_vector(self):
        ref = numpy.array([ 1.352544,  0.338065,  0.002097])
        for i, j in zip(ref, self.tol_atoms[0].get_vector(self.tol_atoms[1])):
            self.assertAlmostEqual(i, j)

    def test_distance(self):
        ref = numpy.linalg.norm(numpy.array([ 1.352544,  0.338065,  0.002097]))
        self.assertAlmostEqual(ref, self.tol_atoms[0].get_distance(self.tol_atoms[1]))

    def test_hybridisation(self):
        self.assertEqual(self.tol_atoms[1].get_hybridisation(), 2)
        self.assertEqual(self.tol_atoms[6].get_hybridisation(), 3)




if __name__ == '__main__':
    unittest.main()