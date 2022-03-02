"""Module to represent any dimensional vectors with a
dot product as an inner product """

from math import sqrt
from math import acos
from math import pi

class Vector:
    def __init__(self, v):
        """ v can be any finite ordered collection of numerical types"""
        self.v = v

    def __str__(self):
        return "(" + ", ".join(str(x) for x in self) + ")"
        
    def __repr__(self):
        return f"Vector{self.__str__()}"

    def __bool__(self):
        """False if self is the zero vector and true otherwise"""
        return not all(x == 0 for x in self)

    def __hash__(self):
        return hash(self.v)

    def __iter__(self):
        """For iterating through componentwise"""
        yield from self.v

    def __len__(self):
        """Returns the dimension of the vector space self belongs to."""
        return len(self.v)

    def __eq__(self, other):
        """Checks componentwise equality of vectors"""
        return self.v == other.v

    def __lt__(self, other):
        """Compares underlying tuples"""
        return self.v < other.v

    def __le__(self, other):
        return self == other or self < other

    def __neg__(self):
        """Returns the componentwise negation of self"""
        return (-1) * self

    def __add__(self, other):
        """Adds self and other"""
        if len(self) != len(other):
            raise NotImplementedError()
        return Vector([x1 + x2 for x1, x2 in zip(self, other)])

    def __sub__(self, other):
        """Subtracts other from self"""
        return self.__add__(other.__neg__())

    def __mul__(self, num):
        """Scalar multiplication by num."""
        return Vector([num * x for x in self])

    def __rmul__(self, num):
        """Scalar multiplication by num."""
        return self.__mul__(num)

    def length_sq(self):
        """ Returns the length-squared of this vector. This is usefull for 
            optimising programs (to avoid calling the square root function)"""
        return sum(x ** 2 for x in self)

    def length(self):
        """The Euclidean length of the vector"""
        return sqrt(self.length_sq())

    def dot(self, other):
        """Returns the dot product of this and other."""
        if len(self) != len(other):
            raise NotImplementedError()
        return sum(x1 * x2 for x1, x2 in zip(self, other))

    def cross(self, other):
        """Returns the cross product of self with other if both are 3-vectors.
           Raises a NotImplementedError exception otherwise"""
        if len(self) != 3 or len(other) != 3:
            raise NotImplementedError()
        return Vector([(self.v[1] * other.v[2]) - (self.v[2] * other.v[1]),
                      (self.v[2] * other.v[0]) - (self.v[0] * other.v[2]),
                      (self.v[0] * other.v[1]) - (self.v[1] * other.v[0])])

    def unit(self):
        """returns a unit vector in the direction of self"""
        return (1 / self.length()) * self

    def angle(self, other):
        """Returns the angle in radians [0, pi] (as defined by the dot product) between self and other"""

        x = self.dot(other) / (self.length() * other.length())

        #occasionally x may lie outside [-1, 1] by a very small amount for
        #floating-point arithematic reasons
        if abs(x) > 1:
            x = (1 if x > 0 else -1)

        return acos(x)

    def __getitem__(self, index):
        if index >= self.v.__len__():
            return None
        return self.v[index]
        
    # Class (static) methods **************************************************
    @staticmethod
    def zero(dim):
        return Vector(dim * [0])

    @staticmethod
    def standard_basis(dim):
        """Returns an iterator for the elements of the standard basis for R^dim"""
        return (Vector(i * [0,] + [1, ] + (dim - i - 1) * [0, ]) 
                for i in range(dim))
        
if __name__ == "__main__":
    for i in range(1, 10):
        print(tuple(Vector.standard_basis(i)))
    