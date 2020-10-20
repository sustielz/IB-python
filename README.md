# IB-python
An implementation of the immersed boundary method in python. The code is originally from Charles Peskin's course "Advanced Topics In Numerical Analysis: Immersed Boundary Method For Fluid Structure Interaction" (NYU, Spring 2019); for lecture notes and more information see https://www.math.nyu.edu/faculty/peskin/ib_lecture_notes/index.html

The core code is based on a vectorized 3D MATLAB implementation available here https://github.com/ModelingSimulation/IB-MATLAB, written by Guanhua Sun and Tristan Goodwill

The goal of this project is to write an object-oriented version of the available MATLAB code which uses numpy to implement the vectorization available in MATLAB; and to compare the python implementation to the original once it is functional to ensure there isn't a significatnt loss of speed. 

I also intend to include various jupyter notebooks intended to both benchmark the code and to provide an interactive way for those interested to learn more about the immersed boundary method. 

I also plan to include a penalty rigid body implementation, and possibly some other variations. 
