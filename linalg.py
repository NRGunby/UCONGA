import math
import numpy


def normalise(a):
    '''
    Normalises a vector
    Accepts: a numpy vector
    Returns: a numpy vector pointing in the same direction with magnitude 1
    '''
    a_norm = numpy.linalg.norm(a)
    return numpy.array([float(each)/a_norm for each in a])


def rotation_axis_angle(axis, angle):
    '''
    Returns the 3x3 matrix for rotation by an angle around an axis
    Accepts: an axis as a numpy array, and an angle in radians
    Returns: a rotation matrix as a numpy array
    '''
    sin = math.sin(angle)
    cos = math.cos(angle)
    comp = 1 - cos
    x, y, z = normalise(axis)
    mat = numpy.array([[(cos + x*x*comp), (x*y*comp - z*sin), (x*z*comp + y*sin)],
                       [(y*x*comp + z*sin), (cos + y*y*comp), (y*z*comp - x*sin)],
                       [(z*x*comp - y*sin), (z*y*comp + x*sin), (cos + z*z*comp)]])
    should_be_I = mat.dot(mat.transpose())
    I = numpy.ma.identity(3)
    numpy.testing.assert_array_almost_equal(I, should_be_I, 3)
    return mat


def rotation_from_axes(ax1, ax2):  # To test
    '''
    Calculate the matrix to rotate one vector to another
    Accepts: two 3-vectors as numpy arrays
    Returns: a rotation matrix as a numpy array
    '''
    # Probably a more numpy-ish way of doing this
    if max(numpy.absolute(ax1 - ax2)) < 1E-7:
        return numpy.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    elif max(numpy.absolute(ax1 + ax2)) < 1E-7:
        ang = angle_between(ax1, ax2)
        z = math.sqrt(1/(1 + (ax1[2]/ax1[1])**2))
        y = math.sqrt(1 - z**2)
        rot_ax = numpy.array([0, y, z])
        return rotation_axis_angle(rot_ax, ang)
    else:
        ang = angle_between(ax1, ax2)
        rot_ax = numpy.cross(ax1, ax2)
        return rotation_axis_angle(rot_ax, ang)


def angle_between(vec1, vec2):
    '''
    Calculate the angle between two vectors
    Accepts: two vectors as numpy arrays
    Returns: the angle in radians
    '''
    return math.acos(float(vec1.dot(vec2)) /
                     (numpy.linalg.norm(vec1) * numpy.linalg.norm(vec2)))


def reflection_plane(vec1, vec2):
    '''
    Returns the Householder reflection matrix for reflection through
    a plane
    Accepts: two non-parallel vectors in the plane as numpy arrays
    Returns: the 3x3 reflection matrix as a numpy array
    '''
    norm = numpy.cross(vec1, vec2)
    a, b, c = normalise(norm)
    return numpy.array([[1 - 2*a*a, -2*a*b, -2*a*c],
                        [-2*a*b, 1-2*b*b, -2*b*c],
                        [-2*a*c, -2*b*c, 1-2*c*c]])
