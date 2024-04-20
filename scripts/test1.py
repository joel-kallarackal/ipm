import numpy as np

binary_coordinates = np.column_stack(np.where(np.array([[255,0,0,0],[0,0,255,255],[255,255,255,0],[0,0,255,0]]) == 255))
print(binary_coordinates)