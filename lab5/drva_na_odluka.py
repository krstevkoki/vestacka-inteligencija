class decisionnode:
    def __init__(self, col=-1, value=None, results=None, tb=None, fb=None):
        self.col = col
        self.value = value
        self.results = results
        self.tb = tb
        self.fb = fb


def sporedi_broj(row, column, value):
    return row[column] >= value


def sporedi_string(row, column, value):
    return row[column] == value


# Divides a set on a specific column. Can handle numeric
# or nominal values
def divideset(rows, column, value):
    # Make a function that tells us if a row is in
    # the first group (true) or the second group (false)
    split_function = None
    if isinstance(value, int) or isinstance(value, float):  # ako vrednosta so koja sporeduvame e od tip int ili float
        # split_function=lambda row:row[column]>=value # togas vrati funkcija cij argument e row i vrakja vrednost
        # true ili false
        split_function = sporedi_broj
    else:
        # split_function=lambda row:row[column]==value # ako vrednosta so koja sporeduvame e od drug tip (string)
        split_function = sporedi_string

    # Divide the rows into two sets and return them
    # set1=[row for row in rows if split_function(row)]  # za sekoj row od rows za koj split_function vrakja true
    # set2=[row for row in rows if not split_function(row)] # za sekoj row od rows za koj split_function vrakja false
    set1 = [row for row in rows if
            split_function(row, column, value)]  # za sekoj row od rows za koj split_function vrakja true
    set2 = [row for row in rows if
            not split_function(row, column, value)]  # za sekoj row od rows za koj split_function vrakja false
    return (set1, set2)


# Create counts of possible results (the last column of
# each row is the result)
def uniquecounts(rows):
    results = {}
    for row in rows:
        # The result is the last column
        r = row[len(row) - 1]
        if r not in results: results[r] = 0
        results[r] += 1
    return results


# Probability that a randomly placed item will
# be in the wrong category
def giniimpurity(rows):
    total = len(rows)
    counts = uniquecounts(rows)
    imp = 0
    for k1 in counts:
        p1 = float(counts[k1]) / total
        for k2 in counts:
            if k1 == k2: continue
            p2 = float(counts[k2]) / total
            imp += p1 * p2
    return imp


# Entropy is the sum of p(x)log(p(x)) across all
# the different possible results
def entropy(rows):
    from math import log
    log2 = lambda x: log(x) / log(2)
    results = uniquecounts(rows)
    # Now calculate the entropy
    ent = 0.0
    for r in list(results.keys()):
        p = float(results[r]) / len(rows)
        ent = ent - p * log2(p)
    return ent


def buildtree(rows, scoref=entropy):
    if len(rows) == 0: return decisionnode()
    current_score = scoref(rows)

    # Set up some variables to track the best criteria
    best_gain = 0.0
    best_criteria = None
    best_sets = None

    column_count = len(rows[0]) - 1
    for col in range(0, column_count):
        # Generate the list of different values in
        # this column
        column_values = {}
        for row in rows:
            column_values[row[col]] = 1
        # Now try dividing the rows up for each value
        # in this column
        for value in list(column_values.keys()):
            (set1, set2) = divideset(rows, col, value)

            # Information gain
            p = float(len(set1)) / len(rows)
            gain = current_score - p * scoref(set1) - (1 - p) * scoref(set2)
            if gain > best_gain and len(set1) > 0 and len(set2) > 0:
                best_gain = gain
                best_criteria = (col, value)
                best_sets = (set1, set2)

    # Create the subbranches
    if best_gain > 0:
        trueBranch = buildtree(best_sets[0])
        falseBranch = buildtree(best_sets[1])
        return decisionnode(col=best_criteria[0], value=best_criteria[1], tb=trueBranch, fb=falseBranch)
    else:
        return decisionnode(results=uniquecounts(rows))


def printtree(tree, indent=''):
    # Is this a leaf node?
    if tree.results is not None:
        print(str(tree.results))
    else:
        # Print the criteria
        print(str(tree.col) + ':' + str(tree.value) + '? ')
        # Print the branches
        print(indent + 'T->', end=' ')
        printtree(tree.tb, indent + '  ')
        print(indent + 'F->', end=' ')
        printtree(tree.fb, indent + '  ')


def classify(observation, tree):
    if tree.results is not None:
        return tree.results
    else:
        vrednost = observation[tree.col]
        branch = None

        if isinstance(vrednost, int) or isinstance(vrednost, float):
            if vrednost >= tree.value:
                branch = tree.tb
            else:
                branch = tree.fb
        else:
            if vrednost == tree.value:
                branch = tree.tb
            else:
                branch = tree.fb

        return classify(observation, branch)


def podeli_set(set):
    middle = int(len(set) / 2)
    set1, set2 = set[:middle], set[middle:]
    return set1, set2


trainingData = [
    [6.3, 2.9, 5.6, 1.8, 'I. virginica'],
    [6.5, 3.0, 5.8, 2.2, 'I. virginica'],
    [7.6, 3.0, 6.6, 2.1, 'I. virginica'],
    [4.9, 2.5, 4.5, 1.7, 'I. virginica'],
    [7.3, 2.9, 6.3, 1.8, 'I. virginica'],
    [6.7, 2.5, 5.8, 1.8, 'I. virginica'],
    [7.2, 3.6, 6.1, 2.5, 'I. virginica'],
    [6.5, 3.2, 5.1, 2.0, 'I. virginica'],
    [6.4, 2.7, 5.3, 1.9, 'I. virginica'],
    [6.8, 3.0, 5.5, 2.1, 'I. virginica'],
    [5.7, 2.5, 5.0, 2.0, 'I. virginica'],
    [5.8, 2.8, 5.1, 2.4, 'I. virginica'],
    [6.4, 3.2, 5.3, 2.3, 'I. virginica'],
    [6.5, 3.0, 5.5, 1.8, 'I. virginica'],
    [7.7, 3.8, 6.7, 2.2, 'I. virginica'],
    [7.7, 2.6, 6.9, 2.3, 'I. virginica'],
    [6.0, 2.2, 5.0, 1.5, 'I. virginica'],
    [6.9, 3.2, 5.7, 2.3, 'I. virginica'],
    [5.6, 2.8, 4.9, 2.0, 'I. virginica'],
    [7.7, 2.8, 6.7, 2.0, 'I. virginica'],
    [6.3, 2.7, 4.9, 1.8, 'I. virginica'],
    [6.7, 3.3, 5.7, 2.1, 'I. virginica'],
    [7.2, 3.2, 6.0, 1.8, 'I. virginica'],
    [6.2, 2.8, 4.8, 1.8, 'I. virginica'],
    [6.1, 3.0, 4.9, 1.8, 'I. virginica'],
    [6.4, 2.8, 5.6, 2.1, 'I. virginica'],
    [7.2, 3.0, 5.8, 1.6, 'I. virginica'],
    [7.4, 2.8, 6.1, 1.9, 'I. virginica'],
    [7.9, 3.8, 6.4, 2.0, 'I. virginica'],
    [6.4, 2.8, 5.6, 2.2, 'I. virginica'],
    [6.3, 2.8, 5.1, 1.5, 'I. virginica'],
    [6.1, 2.6, 5.6, 1.4, 'I. virginica'],
    [7.7, 3.0, 6.1, 2.3, 'I. virginica'],
    [6.3, 3.4, 5.6, 2.4, 'I. virginica'],
    [5.1, 3.5, 1.4, 0.2, 'I. setosa'],
    [4.9, 3.0, 1.4, 0.2, 'I. setosa'],
    [4.7, 3.2, 1.3, 0.2, 'I. setosa'],
    [4.6, 3.1, 1.5, 0.2, 'I. setosa'],
    [5.0, 3.6, 1.4, 0.2, 'I. setosa'],
    [5.4, 3.9, 1.7, 0.4, 'I. setosa'],
    [4.6, 3.4, 1.4, 0.3, 'I. setosa'],
    [5.0, 3.4, 1.5, 0.2, 'I. setosa'],
    [4.4, 2.9, 1.4, 0.2, 'I. setosa'],
    [4.9, 3.1, 1.5, 0.1, 'I. setosa'],
    [5.4, 3.7, 1.5, 0.2, 'I. setosa'],
    [4.8, 3.4, 1.6, 0.2, 'I. setosa'],
    [4.8, 3.0, 1.4, 0.1, 'I. setosa'],
    [4.3, 3.0, 1.1, 0.1, 'I. setosa'],
    [5.8, 4.0, 1.2, 0.2, 'I. setosa'],
    [5.7, 4.4, 1.5, 0.4, 'I. setosa'],
    [5.4, 3.9, 1.3, 0.4, 'I. setosa'],
    [5.1, 3.5, 1.4, 0.3, 'I. setosa'],
    [5.7, 3.8, 1.7, 0.3, 'I. setosa'],
    [5.1, 3.8, 1.5, 0.3, 'I. setosa'],
    [5.4, 3.4, 1.7, 0.2, 'I. setosa'],
    [5.1, 3.7, 1.5, 0.4, 'I. setosa'],
    [4.6, 3.6, 1.0, 0.2, 'I. setosa'],
    [5.1, 3.3, 1.7, 0.5, 'I. setosa'],
    [4.8, 3.4, 1.9, 0.2, 'I. setosa'],
    [5.0, 3.0, 1.6, 0.2, 'I. setosa'],
    [5.0, 3.4, 1.6, 0.4, 'I. setosa'],
    [5.2, 3.5, 1.5, 0.2, 'I. setosa'],
    [5.2, 3.4, 1.4, 0.2, 'I. setosa'],
    [5.5, 2.3, 4.0, 1.3, 'I. versicolor'],
    [6.5, 2.8, 4.6, 1.5, 'I. versicolor'],
    [5.7, 2.8, 4.5, 1.3, 'I. versicolor'],
    [6.3, 3.3, 4.7, 1.6, 'I. versicolor'],
    [4.9, 2.4, 3.3, 1.0, 'I. versicolor'],
    [6.6, 2.9, 4.6, 1.3, 'I. versicolor'],
    [5.2, 2.7, 3.9, 1.4, 'I. versicolor'],
    [5.0, 2.0, 3.5, 1.0, 'I. versicolor'],
    [5.9, 3.0, 4.2, 1.5, 'I. versicolor'],
    [6.0, 2.2, 4.0, 1.0, 'I. versicolor'],
    [6.1, 2.9, 4.7, 1.4, 'I. versicolor'],
    [5.6, 2.9, 3.6, 1.3, 'I. versicolor'],
    [6.7, 3.1, 4.4, 1.4, 'I. versicolor'],
    [5.6, 3.0, 4.5, 1.5, 'I. versicolor'],
    [5.8, 2.7, 4.1, 1.0, 'I. versicolor'],
    [6.2, 2.2, 4.5, 1.5, 'I. versicolor'],
    [5.6, 2.5, 3.9, 1.1, 'I. versicolor'],
    [5.9, 3.2, 4.8, 1.8, 'I. versicolor'],
    [6.1, 2.8, 4.0, 1.3, 'I. versicolor'],
    [6.3, 2.5, 4.9, 1.5, 'I. versicolor'],
    [6.1, 2.8, 4.7, 1.2, 'I. versicolor'],
    [6.4, 2.9, 4.3, 1.3, 'I. versicolor'],
    [6.6, 3.0, 4.4, 1.4, 'I. versicolor'],
    [6.8, 2.8, 4.8, 1.4, 'I. versicolor'],
    [6.7, 3.0, 5.0, 1.7, 'I. versicolor'],
    [6.0, 2.9, 4.5, 1.5, 'I. versicolor'],
    [5.7, 2.6, 3.5, 1.0, 'I. versicolor'],
    [5.5, 2.4, 3.8, 1.1, 'I. versicolor'],
    [5.5, 2.4, 3.7, 1.0, 'I. versicolor'],
    [5.8, 2.7, 3.9, 1.2, 'I. versicolor'],
    [6.0, 2.7, 5.1, 1.6, 'I. versicolor'],
    [5.4, 3.0, 4.5, 1.5, 'I. versicolor'],
    [6.0, 3.4, 4.5, 1.6, 'I. versicolor'],
    [6.7, 3.1, 4.7, 1.5, 'I. versicolor'],
    [6.3, 2.3, 4.4, 1.3, 'I. versicolor'],
    [5.6, 3.0, 4.1, 1.3, 'I. versicolor'],
    [5.5, 2.5, 4.0, 1.3, 'I. versicolor'],
    [5.5, 2.6, 4.4, 1.2, 'I. versicolor'],
    [6.1, 3.0, 4.6, 1.4, 'I. versicolor'],
    [5.8, 2.6, 4.0, 1.2, 'I. versicolor'],
    [5.0, 2.3, 3.3, 1.0, 'I. versicolor'],
    [5.6, 2.7, 4.2, 1.3, 'I. versicolor'],
    [5.7, 3.0, 4.2, 1.2, 'I. versicolor'],
    [5.7, 2.9, 4.2, 1.3, 'I. versicolor'],
    [6.2, 2.9, 4.3, 1.3, 'I. versicolor'],
    [5.1, 2.5, 3.0, 1.1, 'I. versicolor'],
    [5.7, 2.8, 4.1, 1.3, 'I. versicolor'],
    [6.4, 3.1, 5.5, 1.8, 'I. virginica'],
    [6.0, 3.0, 4.8, 1.8, 'I. virginica'],
    [6.9, 3.1, 5.4, 2.1, 'I. virginica'],
    [6.7, 3.1, 5.6, 2.4, 'I. virginica'],
    [6.9, 3.1, 5.1, 2.3, 'I. virginica'],
    [5.8, 2.7, 5.1, 1.9, 'I. virginica'],
    [6.8, 3.2, 5.9, 2.3, 'I. virginica'],
    [6.7, 3.3, 5.7, 2.5, 'I. virginica'],
    [6.7, 3.0, 5.2, 2.3, 'I. virginica'],
    [6.3, 2.5, 5.0, 1.9, 'I. virginica'],
    [6.5, 3.0, 5.2, 2.0, 'I. virginica'],
    [6.2, 3.4, 5.4, 2.3, 'I. virginica'],
    [4.7, 3.2, 1.6, 0.2, 'I. setosa'],
    [4.8, 3.1, 1.6, 0.2, 'I. setosa'],
    [5.4, 3.4, 1.5, 0.4, 'I. setosa'],
    [5.2, 4.1, 1.5, 0.1, 'I. setosa'],
    [5.5, 4.2, 1.4, 0.2, 'I. setosa'],
    [4.9, 3.1, 1.5, 0.2, 'I. setosa'],
    [5.0, 3.2, 1.2, 0.2, 'I. setosa'],
    [5.5, 3.5, 1.3, 0.2, 'I. setosa'],
    [4.9, 3.6, 1.4, 0.1, 'I. setosa'],
    [4.4, 3.0, 1.3, 0.2, 'I. setosa'],
    [5.1, 3.4, 1.5, 0.2, 'I. setosa'],
    [5.0, 3.5, 1.3, 0.3, 'I. setosa'],
    [4.5, 2.3, 1.3, 0.3, 'I. setosa'],
    [4.4, 3.2, 1.3, 0.2, 'I. setosa'],
    [5.0, 3.5, 1.6, 0.6, 'I. setosa'],
    [5.1, 3.8, 1.9, 0.4, 'I. setosa'],
    [4.8, 3.0, 1.4, 0.3, 'I. setosa'],
    [5.1, 3.8, 1.6, 0.2, 'I. setosa'],
    [5.9, 3.0, 5.1, 1.8, 'I. virginica']
]


def sort_function(item):
    return -item[1], item[0]


if __name__ == "__main__":
    att1 = input()
    att2 = input()
    att3 = input()
    att4 = input()
    planttype = input()
    testCase = [att1, att2, att3, att4, planttype]

    set1, set2 = podeli_set(trainingData)

    tree1 = buildtree(set1)
    tree2 = buildtree(set2)

    res1 = dict(classify(testCase, tree1))
    res2 = dict(classify(testCase, tree2))

    if (len(res1) == 1) and (len(res2) == 1):
        if res1.keys() != res2.keys():
            print("KONTRADIKCIJA")
        else:
            print(list(res1.keys())[0])
    else:
        sorted1 = sorted(res1.items(), key=sort_function, reverse=False)
        sorted2 = sorted(res2.items(), key=sort_function, reverse=False)

        if sorted1[0][0] != sorted2[0][0]:
            print("KONTRADIKCIJA")
        else:
            print(sorted1[0])
