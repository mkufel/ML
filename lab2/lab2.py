import numpy
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from math import e

# numpy.random.seed(100)
# Generate samples of disjoint classes
classA = numpy.concatenate(
    (numpy.random.randn(10, 2) * 0.2 + [1.5, 0.5],
    numpy.random.randn(10, 2) * 0.2 + [-1.5, 0.5]))
classB = numpy.random.randn(20, 2) * 0.2 + [0.0, -0.5]

# Concatenate the samples and create labels
inputs = numpy.concatenate((classA, classB))
targets = numpy.concatenate((
    numpy.ones(classA.shape[0]),
    -numpy.ones(classB.shape[0])))
N = inputs.shape[0]

# kernel definitions
def linearKernel(a, b):
    return numpy.dot(a, b)

p = 2
def polynomialKernel(a, b):
    return (numpy.dot(a, b) + 1) ** p

sigma = 2
def rbfKernel(a, b):
    numerator = numpy.linalg.norm(numpy.subtract(a, b)) ** 2
    denominator = 2 * (sigma ** 2)
    return e ** -(numerator/denominator)

chosenKernel = linearKernel

# Pre-computed matrix of values P_{i,j} = K(x_{i}, x_{j} * t_{i} * t_{j})
helperMatrix = numpy.zeros(shape=(N,N))
helperMatrix = [[chosenKernel(inputs[x], inputs[y]) * targets[x] * targets[y] for y in range(N)] for x in range(N)]

# Dual problem formula
def objective(alpha):
    return (1/2) * numpy.dot(alpha, numpy.dot(alpha, helperMatrix)) - numpy.sum(alpha)

# Dual formula constraint Sigma_{i} (alpha_{i} * target_{i}) = 0
def zerofun(alpha):
    return numpy.dot(alpha, targets)

C = 1000
bound = [(0, C) for b in range(N)]
constraint = {'type': 'eq', 'fun': zerofun}
start = numpy.zeros(N)

# try to minimize the objective function (SVM dual problem) given the start vector, bound for C and constraint
ret = minimize(objective, start, bounds=bound, constraints=constraint)
results = ret['x']
resultsNonZero = [(round(results[i], 5), inputs[i], targets[i]) for i in range(N) if abs(results[i]) > 10e-5]

def bvalue():
    sum = 0
    for result in resultsNonZero:
        sum += result[0] * result[2] * chosenKernel(result[1], resultsNonZero[0][1])
    return sum - resultsNonZero[0][2]

def indicator(x, y):
    value = 0
    for result in resultsNonZero:
        value += numpy.dot(numpy.dot(result[0], result[2]), chosenKernel([x, y], result[1]))
    return value - bvalue()

# plot the samples
plt.plot([p[0] for p in classA],
         [p[1] for p in classA],
         'b.')
plt.plot([p[0] for p in classB],
         [p[1] for p in classB],
         'r.')

xgrid = numpy.linspace(-5, 5)
ygrid = numpy.linspace(-4, 4)
grid = numpy.array([[indicator(x, y)
                     for x in xgrid]
                    for y in ygrid])
# plot the boundary line
plt.contour(xgrid, ygrid, grid, (-1.0, 0.0, 1.0), colors=('red', 'black', 'blue'), linewidth=(1, 3, 1))

plt.axis('equal')
plt.savefig('svmplot.pdf')
plt.show()
