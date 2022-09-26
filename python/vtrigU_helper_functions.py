import numpy as np


def ants_locations():
    return np.array([[-0.0275, -0.0267, 0],
                     [-0.0253, -0.0267, 0],
                     [-0.0231, -0.0267, 0],
                     [-0.0209, -0.0267, 0],
                     [-0.0187, -0.0267, 0],
                     [-0.0165, -0.0267, 0],
                     [-0.0143, -0.0267, 0],
                     [-0.0122, -0.0267, 0],
                     [-0.0100, -0.0267, 0],
                     [-0.0078, -0.0267, 0],
                     [-0.0056, -0.0267, 0],
                     [-0.0034, -0.0267, 0],
                     [-0.0012, -0.0267, 0],
                     [ 0.0009, -0.0267, 0],
                     [ 0.0031, -0.0267, 0],
                     [ 0.0053, -0.0267, 0],
                     [ 0.0075, -0.0267, 0],
                     [ 0.0097, -0.0267, 0],
                     [ 0.0119, -0.0267, 0],
                     [ 0.0141, -0.0267, 0],
                     [ 0.0274, -0.0133, 0],
                     [ 0.0274, -0.0112, 0],
                     [ 0.0274, -0.0091, 0],
                     [ 0.0274, -0.0070, 0],
                     [ 0.0274, -0.0049, 0],
                     [ 0.0274, -0.0028, 0],
                     [ 0.0274, -0.0007, 0],
                     [ 0.0275,  0.0014, 0],
                     [ 0.0275,  0.0035, 0],
                     [ 0.0275,  0.0056, 0],
                     [ 0.0275,  0.0078, 0],
                     [ 0.0275,  0.0099, 0],
                     [ 0.0275,  0.0120, 0],
                     [ 0.0274,  0.0141, 0],
                     [ 0.0274,  0.0162, 0],
                     [ 0.0275,  0.0183, 0],
                     [ 0.0275,  0.0204, 0],
                     [ 0.0275,  0.0225, 0],
                     [ 0.0275,  0.0246, 0],
                     [ 0.0275,  0.0267, 0]])

def RadiationPattern(theta, phi):
    return np.sqrt((np.cos(theta/2)**2)*((np.sin(phi)**2)+(np.cos(theta)*np.cos(phi))**2))
