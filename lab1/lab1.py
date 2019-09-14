import sys
sys.path.append('../dectrees/python')
import monkdata as m
import dtree as dt
import drawtree_qt5 as qt
import statistics
import pyqtgraph as pg
import random

# Print entropy of all datasets
monks = [m.monk1, m.monk2, m.monk3]
for idx,monk in enumerate(monks):
    print("Entropy of monk" + str(idx+1) + ": " + str(dt.entropy(monk)))

# Print average gain for alls monks for different attribute values
print("         ", end='')
for attribute in m.attributes:
    print(str(attribute) + "    ", end='')
print()

for idx, monk in enumerate(monks):
    print("MONK-" + str(idx+1) + " ", end='')
    for attribute in m.attributes:
        print("%.3f" % dt.averageGain(monk,attribute) + " ", end='')
    print()

# Prints a decision tree of given depth for the given dataset
def buildTreeCustom(dataset, depth):
    if (depth > 0):
        bestAttr = dt.bestAttribute(dataset, m.attributes)
        print(str(bestAttr), end='')

        # Select datasets splits for each value of the bestAttr
        splits = []
        for value in bestAttr.values:
            splits.append(dt.select(dataset, bestAttr, value))

        for split in splits:
            # If entropy of the split > 0, the split is impure and we can further split it. Recursive call with reduced depth
            if (dt.entropy(split) > 0):
                buildTreeCustom(split, depth - 1)
            else:
                print('+' if dt.mostCommon(split) else '-', end='')
    else:
        print('+' if dt.mostCommon(dataset) else '-', end='')

# Print the same tree using the custom and given functions for comparison
buildTreeCustom(m.monk1, 2)
print(dt.buildTree(m.monk1, m.attributes, 2))

# Build trees for all datasets and calculate performance for training and test datasets
t1 = dt.buildTree(m.monk1, m.attributes)
t2 = dt.buildTree(m.monk2, m.attributes)
t3 = dt.buildTree(m.monk3, m.attributes)

print("MONK-1 Training set: " + str(dt.check(t1, m.monk1)) + ", Test set: " + str(dt.check(t1, m.monk1test)))
print("MONK-2 Training set: " + str(dt.check(t2, m.monk2)) + ", Test set: " + str(dt.check(t2, m.monk2test)))
print("MONK-3 Training set: " + str(dt.check(t3, m.monk3)) + ", Test set: " + str(dt.check(t3, m.monk3test)))


def partition(data, fraction):
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata) * fraction)
    return ldata[:breakPoint], ldata[breakPoint:]

# For the given tree, returns a pruned tree with the highest performance on the given validation set
def prune(tree, valSet):
    currentTree = tree
    currentPerf = dt.check(currentTree, valSet)
    pTrees = dt.allPruned(currentTree)
    for pTree in pTrees:
        if (dt.check(pTree, valSet) > currentPerf):
            currentTree = prune(pTree, valSet)
            currentPerf = dt.check(currentTree, valSet)
    return currentTree

def prune2(tree, validation_set):
    currentTree = tree
    maxValue = 1
    oldValue = 0
    while maxValue > oldValue:
        maxValue = dt.check(currentTree, validation_set)
        oldValue = maxValue
        maxTree = currentTree

        for prunedTree in allPruned(currentTree):
            # Value for the pruned tree
            currentValue = dt.check(prunedTree, validation_set)
            if currentValue > maxValue:
                maxTree = prunedTree
                maxValue = currentValue
        currentTree = maxTree
    return maxTree

monk1train, monk1val = partition(m.monk1, 0.6)
# qt.drawTree(prune(t1, monk1val))

# Fractions for splitting the training set into training and validation sets
fractions = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
m1performance = []
m3performance = []

# Compute the mean performance for MONK-1 and MONK-3 for each fraction split on examples
for fraction in fractions:
    val1 = 0
    val3 = 0

    for i in range(100):
        monk1train, monk1val = partition(m.monk1, fraction)
        t1 = dt.buildTree(monk1train, m.attributes)
        val1 += dt.check(prune(t1, monk1val), m.monk1test)

        monk3train, monk3val = partition(m.monk3, fraction)
        t3 = dt.buildTree(monk3train, m.attributes)
        val3 += dt.check(prune(t3, monk3val), m.monk3test)

    val1 /= 100
    m1performance.append(val1)

    val3 /= 100
    m3performance.append(val3)

m1error = []
m3error = []
for i in range(0, 6):
    m1error.append(1 - m1performance[i])
    m3error.append(1 - m3performance[i])


print("Errors for pruned trees for different validation sets")
print("  ", end='')

for fraction in fractions:
    print(str(fraction) + "    ", end='')
print()

for error in m1error:
    print("%.4f" % error + " ", end='')
print()

for error in m3error:
    print("%.4f" % error + " ", end='')
print("\n")


print("Mean and Variance of error for pruned trees")
print("MONK-1 Mean error: " + str("%.4f" % statistics.mean(m1error)) + " Variance: " + str(
    "%.4f" % statistics.variance(m1error)))
print("MONK-3 Mean error: " + str("%.4f" % statistics.mean(m3error)) + " Variance: " + str(
    "%.4f" % statistics.variance(m3error)))
print()


print("Error for unpruned trees")
print("MONK-1: " + str("%.4f" % (1 - dt.check(t1, m.monk1test))))
print("MONK-3: " + str("%.4f" % (1 - dt.check(t3, m.monk3test))))


# print("M1 performance " + str(m1performance))
# print("M3 performance " + str(m3performance))
#
# print("Variance of M1 results: " + str(statistics.variance(m1performance)))
# print("Mean of M1 results: " + str(statistics.mean(m1performance)))
# print("Variance of M3 results: " + str(statistics.variance(m3performance)))
# print("Mean of M3 results: " + str(statistics.mean(m3performance)))

# Draw scatter plots for MONK-1 and MONK-3 fraction vs. Mean error of a pruned tree
app = pg.mkQApp()

plt = pg.plot(fractions, m1error, title='Mean error for MONK-1', symbol='o', pen=None, left='Classification error', bottom='Fraction of the training set used for training')
plt.showGrid(x=True,y=True)

plt2 = pg.plot(fractions, m3error, title='Mean error for MONK-3', symbol='o', pen=None, left='Classification error', bottom='Fraction of the training set used for training')
plt2.showGrid(x=True,y=True)

status = app.exec_()
sys.exit(status)