import numpy as np

A = np.array([[802, -400],[-400,200]])

eig = np.linalg.eigvals(A)

det = np.linalg.det(A)

print 'EigenValues: ', eig, '\nDeterminant: ', det
