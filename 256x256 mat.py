import numpy as np
nrr=np.random.random((256,256))
print(nrr)
print(np.linalg.inv(nrr)@nrr)