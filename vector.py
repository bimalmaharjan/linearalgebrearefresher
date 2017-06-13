
import numpy as np
import math
from decimal import Decimal, getcontext
getcontext().prec = 30
class Vector(object):

    CANNOT_NORMALIZE_ZERO_VECTOR_MSG = "Cannot normalize zero vector"
    NO_UNIQUE_PARALLEL_COMPONENT_MSG = " no unique parallel component"
    ONLY_DEFINED_IN_TWO_THREE_DIMS_MSG = " only defined in two three dimensions"
    def __init__(self, coordinates):
        try:
            if not coordinates:
                raise ValueError
            self.coordinates = tuple([ Decimal(x) for x in coordinates])
            self.dimension = len(coordinates)

        except ValueError:
            raise ValueError('The coordinates must be nonempty')

        except TypeError:
            raise TypeError('The coordinates must be an iterable')


    def plus(self, v):
        new_coordinates = [x + y for x,y in zip(self.coordinates, v.coordinates)]
        return new_coordinates

    def minus(self,v):
        new_coordinates = [x - y for x,y in zip(self.coordinates, v.coordinates)]
        return new_coordinates

    def timescalar(self, c):
        new_coordinates = [c * x for x in self.coordinates]
        return new_coordinates

    def magnitude(self):
        coordinates_square = [ x ** 2 for x in self.coordinates]
        return math.sqrt(sum(coordinates_square))

    def normalized(self):
        try:
            magnitude = Decimal(self.magnitude())
            return self.timescalar(Decimal('1.0')/magnitude)

        except ZeroDivisionError:
            raise Exception (self.CANNOT_NORMALIZE_ZERO_VECTOR_MSG)

    def dot(self,v):
        return sum([ x * y for x, y in zip(self.coordinates,v.coordinates)])

    def angle_with(self,v, in_degrees = False):
        
        try:
            u1 = Vector(self.normalized())
            u2 = Vector(v.normalized())   

            # u1.dot(u2) must be between -1 and 1. rounding up after 10 digits for precision

            angle_radian = math.acos(round(u1.dot(u2),10))
            

            if in_degrees:
                return math.degrees(angle_radian)
            
            else:
                return angle_radian
        except Exception as e:
            if str(e) == self.CANNOT_NORMALIZE_ZERO_VECTOR_MSG:
                raise Exception('Cannot compute an angle with the zero vector')
            else:
                print e
                raise e


    def is_parallel_to(self,v):
        if (self.is_zero() or v.is_zero() or self.angle_with(v) == 0 or self.angle_with(v) == math.pi):
            return True
        else:
            return False

    def is_zero(self, tolerance =1e-10):
        return self.magnitude() < tolerance

    def is_orthogonal_to(self,v, tolerance =1e-10):
        if abs(self.dot(v)) < tolerance:
            return True
        else:
            return False

    def component_parallel_to(self, basis):
        
        try:
            u = Vector(basis.normalized())
            weight = self.dot(u)
            return u.timescalar(weight)

        except Exception as e:
            if str(e) == self.CANNOT_NORMALIZE_ZERO_VECTOR_MSG:
                raise Exception(self.NO_UNIQUE_PARALLEL_COMPONENT_MSG)
            else:
                raise e


    def component_orthogonal_to(self, basis):
        try:
            projection= Vector(self.component_parallel_to(basis))
            return self.minus(projection)

        except Exception as e:
            if str(e) == self.CANNOT_NORMALIZE_ZERO_VECTOR_MSG:
                raise Exception(self.NO_UNIQUE_PARALLEL_COMPONENT_MSG)
            else:
                raise e

    def cross_product(self,v):

        try:
            x1,y1,z1 = self.coordinates
            x2,y2,z2 = v.coordinates

            new_coordinates = [y1*z2-y2*z1, -(x1*z2-x2*z1),x1*y2-x2*y1]          
            return Vector(new_coordinates)
        except ValueError as e:
            msg=str(e)
            if msg =="need more than 2 values to unpack":
                self_embedded_in_R3 = Vector(self.coordinates + ('0,'))
                v_embedded_in_R3 = Vector(v.coordinates + ('0,'))
            elif msg =="too many values to unpack" or msg == "need more than 1 value to unpack":
                raise Exception(self.ONLY_DEFINED_IN_TWO_THREE_DIMS_MSG)
            else:
                raise e


    def area_of_parallelogram(self,v):
        cross_product = self.cross_product(v)
        return cross_product.magnitude()

    def area_of_triangle(self,v):
        return 0.5 * self.area_of_parallelogram(v)







    def __str__(self):
        return 'Vector: {}'.format(self.coordinates)


    def __eq__(self, v):
        return self.coordinates == v.coordinates

    def __iter__(self):
            return iter(self.coordinates)

    def __getitem__(self,index):
            return self.coordinates[index]




# my_vector = Vector([1,2,3])
# print my_vector
# print type(my_vector)

# my_vector2 = Vector([1,2,3])
# my_vector3 = Vector([-1,2,3])

# print my_vector == my_vector2
# print my_vector2 == my_vector3

# print zip(my_vector.coordinates, my_vector3.coordinates)

# # lesson 2.4
# v1 = Vector([8.218,-9.341])
# w1 = Vector([-1.129,2.111])

# v2 = Vector([7.119,8.215])
# w2 = Vector([-8.223,0.878])

# c = 7.41
# v3 = Vector([1.671,-1.012,-0.318])

# print v1.plus(w1)
# print v2.minus(w2)
# print v3.timescalar(c)


# # print 'vector addition = ',  np.array([8.218,-9.341]) + np.array([-1.129,2.111])
# # print 'vector substraction = ',  np.array([7.119,8.215]) - np.array([-8.223,0.878])
# # print 'scalar multiplication = ', 7.41 * np.array([1.671,-1.012,-0.318])

# # lesson 2.6

# v = Vector([-0.211,7.437])
# w = Vector([8.813,-1.331,-6.247])

# print 'magnitude of v =', v.magnitude()
# print 'magnitude of w =', w.magnitude()

# v4 = Vector([5.581,-2.136])
# w4 = Vector([1.996,3.108,-4.554])

# print 'normalization of v4 =', v4.normalized()
# print 'normalization of w4 =', w4.normalized()

# # lesson 2.8

# v5 = Vector([7.887,4.138])
# w5 = Vector([-8.802,6.776])

# v6 = Vector([-5.955,-4.904,-1.874])
# w6 = Vector([-4.496,-8.755,7.103])

# print "dot product of v5 and w5= ", v5.dot(w5)
# print "dot product of v6 and w6= ", v6.dot(w6)

# v7 = Vector([3.183,-7.627])
# w7 = Vector([-2.668,5.319])

# v8 = Vector([7.35,0.221,5.188])
# w8 = Vector([2.751,8.259,3.985])

# print "angle between v7 and w7 in rad=", v7.angle_with(w7)

# print "angle between v8 and w8 in deg=", v8.angle_with(w8, True)

# lesson 2.10

# v9 = Vector([-7.579, -7.88])
# w9 = Vector([22.737,23.64])

# v10 = Vector([-2.029,9.97,4.172])
# w10 = Vector([-9.231,-6.639,-7.245])

# v11 = Vector([-2.328,-7.284,-1.214])
# w11 = Vector([-1.821,1.072,-2.94])

# v12 = Vector([2.118,4.827])
# w12 = Vector([0,0])





# # print "dot product between v9 and w9 in= ", Vector((v9.normalized())).dot(Vector(w9.normalized()))
# # print "angle between v9 and w9 in rad=", v9.angle(w9)


# print "v9 is parallel to w9 :", v9.is_parallel_to(w9)
# print "v9 is orthogonal to w9:", v9.is_orthogonal_to(w9)


# print "v10 is parallel to w10 :", v10.is_parallel_to(w10)
# print "v10 is orthogonal to w10:", v10.is_orthogonal_to(w10)

# print "v11 is parallel to w11 :", v11.is_parallel_to(w11)
# print "v11 is orthogonal to w11:", v11.is_orthogonal_to(w11)

# print "v12 is parallel to w12 :", v12.is_parallel_to(w12)
# print "v12 is orthogonal to w12:", v12.is_orthogonal_to(w12)

# # 2.12

# v13 = Vector([3.039,1.879])
# b13 = Vector([0.825,2.036])

# print "v13 projection on b13 = ", v13.component_parallel_to(b13)

# v14 = Vector([-9.88,-3.264,-8.159])
# b14 = Vector([-2.155,-9.353,-9.473])

# print "v14 orthogonal on b14 =", v14.component_orthogonal_to(b14)

# v15 = Vector([3.009,-6.172,3.692,-2.51])
# b15 = Vector([6.404,-9.144,2.759,8.718])

# print "v15 prjection= ", v15.component_parallel_to(b15)
# print "v15 orthogonal=", v15.component_orthogonal_to(b15)

# # 2.14 

# v16 = Vector([8.462,7.893,-8.187])
# w16 = Vector([6.984,-5.975,4.778])

# v17 = Vector([-8.987,-9.838,5.031])
# w17 = Vector([-4.268,-1.861,-8.866])

# v18 = Vector([1.5,9.547,3.691])
# w18 = Vector([-6.007,0.124,5.772])

# print " \n cross product between v16 and w 16 is ", v16.cross_product(w16)
# print "\n area of parallelogram spanned by v17 and v18", v17.area_of_parallelogram(w17)
# print "\n area of triangle",v18.area_of_triangle(w18)





