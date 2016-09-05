import linalg
import unittest
from math import pi, sqrt
import numpy
import random

class TestNormalise(unittest.TestCase):
    def test_works(self):
        result = linalg.normalise(numpy.array([1, 2, 1, 3, 1]))
        self.assertAlmostEqual(numpy.linalg.norm(result),1)


class TestRotationMatrix(unittest.TestCase):

    def test_simple(self):
        result = linalg.rotation_axis_angle(numpy.array([0, 0, 2]), pi/2.0)
        should_be = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        for i, j in zip(result, should_be):
            for k, l in zip(i, j):
                self.assertAlmostEqual(k, l)

    def test_angle(self):
        result = linalg.rotation_axis_angle(numpy.array([0, 1, 0]), pi/6.0)
        should_be = [[sqrt(3)/2.0, 0, 0.5], [0, 1, 0], [-0.5, 0, sqrt(3)/2]]
        for i, j in zip(result, should_be):
            for k, l in zip(i, j):
                self.assertAlmostEqual(k, l)


class TestAxesRotationMatrix(unittest.TestCase):
    def test_identity(self):
        result = linalg.rotation_from_axes(numpy.array([2, -5, 0.9]), numpy.array([2, -5, 0.9]))
        should_be = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        for i, j in zip(result, should_be):
            for k, l in zip(i, j):
                self.assertAlmostEqual(k, l)

    def test_opposite(self):
        a = numpy.array([2, -5, 0.9])
        b = numpy.array([-2, 5, -0.9])
        result = linalg.rotation_from_axes(a, b)
        c = result.dot(a.transpose()).transpose()
        for i, j in zip(b, c):
            self.assertAlmostEqual(i, j)

    def test_working(self):
        v1 = numpy.array([0.5, -0.5, 3])
        v2 = numpy.array([-3, 0.5, 0.5])
        m = linalg.rotation_from_axes(v1, v2)
        v3 = m.dot(v1.transpose()).transpose()
        for i, j in zip(v2, v3):
            self.assertAlmostEqual(i, j)

class TestReflectionPlane(unittest.TestCase):
    def random_vector(self):
        return numpy.array([random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)])
    def test_identity(self):
        for i in range(20):
            vecs = (self.random_vector(), self.random_vector())
            rp = linalg.reflection_plane(*vecs)
            reflected_vecs = [rp.dot(i) for i in vecs]
            diffs = [i -j for i, j in zip(vecs, reflected_vecs)]
            for i in diffs:
                self.assertTrue((numpy.fabs(i) < 1E-4).all())
    def test_reflects(self):
        coords = (numpy.array([1, 0, 0]), numpy.array([0, 1, 0]), numpy.array([0, 0, 1]))
        for i in coords:
            plane = filter(lambda x: not x is i, coords)
            reference = -1 * i
            active = linalg.reflection_plane(*plane).dot(i)
            self.assertTrue((numpy.fabs(reference - active) < 1E-4).all())



if __name__ == '__main__':
    unittest.main()
