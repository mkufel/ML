import numpy, random, math
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# fixed input dataset for initial testing
fixedInputA = numpy.array([(1, 1), (2, 0.5)])
fixedInputB = numpy.array([(0,0), (1, 0.5)])

fixedInputs = numpy.concatenate((fixedInputA, fixedInputB))
fixedInputTargets = numpy.concatenate((
    numpy.ones(fixedInputA.shape[0]),
    -numpy.ones(fixedInputB.shape[0])
))

print(fixedInputs)
print(fixedInputTargets)

plt.plot([p[0] for p in fixedInputA],
         [p[1] for p in fixedInputA],
         'b.')
plt.plot([p[0] for p in fixedInputB],
         [p[1] for p in fixedInputB],
         'r.')

plt.axis('equal')
# plt.savefig('svmplot.pdf')
plt.show()
