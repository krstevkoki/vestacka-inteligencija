# Python modul vo koj se implementirani algoritmite za neinformirano i informirano prebaruvanje

# ______________________________________________________________________________________________
# Improtiranje na dopolnitelno potrebni paketi za funkcioniranje na kodovite

import sys
import bisect

infinity = float('inf')  # sistemski definirana vrednost za beskonecnost


# ______________________________________________________________________________________________
# Definiranje na pomosni strukturi za cuvanje na listata na generirani, no neprovereni jazli

class Queue:
    """Queue is an abstract class/interface. There are three types:
        Stack(): A Last In First Out Queue.
        FIFOQueue(): A First In First Out Queue.
        PriorityQueue(order, f): Queue in sorted order (default min-first).
    Each type supports the following methods and functions:
        q.append(item)  -- add an item to the queue
        q.extend(items) -- equivalent to: for item in items: q.append(item)
        q.pop()         -- return the top item from the queue
        len(q)          -- number of items in q (also q.__len())
        item in q       -- does q contain item?
    Note that isinstance(Stack(), Queue) is false, because we implement stacks
    as lists.  If Python ever gets interfaces, Queue will be an interface."""

    def __init__(self):
        raise NotImplementedError

    def extend(self, items):
        for item in items:
            self.append(item)


def Stack():
    """A Last-In-First-Out Queue."""
    return []


class FIFOQueue(Queue):
    """A First-In-First-Out Queue."""

    def __init__(self):
        self.A = []
        self.start = 0

    def append(self, item):
        self.A.append(item)

    def __len__(self):
        return len(self.A) - self.start

    def extend(self, items):
        self.A.extend(items)

    def pop(self):
        e = self.A[self.start]
        self.start += 1
        if self.start > 5 and self.start > len(self.A) / 2:
            self.A = self.A[self.start:]
            self.start = 0
        return e

    def __contains__(self, item):
        return item in self.A[self.start:]


class PriorityQueue(Queue):
    """A queue in which the minimum (or maximum) element (as determined by f and
    order) is returned first. If order is min, the item with minimum f(x) is
    returned first; if order is max, then it is the item with maximum f(x).
    Also supports dict-like lookup. This structure will be most useful in informed searches"""

    def __init__(self, order=min, f=lambda x: x):
        self.A = []
        self.order = order
        self.f = f

    def append(self, item):
        bisect.insort(self.A, (self.f(item), item))

    def __len__(self):
        return len(self.A)

    def pop(self):
        if self.order == min:
            return self.A.pop(0)[1]
        else:
            return self.A.pop()[1]

    def __contains__(self, item):
        return any(item == pair[1] for pair in self.A)

    def __getitem__(self, key):
        for _, item in self.A:
            if item == key:
                return item

    def __delitem__(self, key):
        for i, (value, item) in enumerate(self.A):
            if item == key:
                self.A.pop(i)


# ______________________________________________________________________________________________
# Definiranje na klasa za strukturata na problemot koj ke go resavame so prebaruvanje
# Klasata Problem e apstraktna klasa od koja pravime nasleduvanje za definiranje na osnovnite karakteristiki
# na sekoj eden problem sto sakame da go resime


class Problem:
    """The abstract class for a formal problem.  You should subclass this and
    implement the method successor, and possibly __init__, goal_test, and
    path_cost. Then you will create instances of your subclass and solve them
    with the various search functions."""

    def __init__(self, initial, goal=None):
        """The constructor specifies the initial state, and possibly a goal
        state, if there is a unique goal.  Your subclass's constructor can add
        other arguments."""
        self.initial = initial
        self.goal = goal

    def successor(self, state):
        """Given a state, return a dictionary of {action : state} pairs reachable
        from this state. If there are many successors, consider an iterator
        that yields the successors one at a time, rather than building them
        all at once. Iterators will work fine within the framework. Yielding is not supported in Python 2.7"""
        raise NotImplementedError

    def actions(self, state):
        """Given a state, return a list of all actions possible from that state"""
        raise NotImplementedError

    def result(self, state, action):
        """Given a state and action, return the resulting state"""
        raise NotImplementedError

    def goal_test(self, state):
        """Return True if the state is a goal. The default method compares the
        state to self.goal, as specified in the constructor. Implement this
        method if checking against a single self.goal is not enough."""
        return state == self.goal

    def path_cost(self, c, state1, action, state2):
        """Return the cost of a solution path that arrives at state2 from
        state1 via action, assuming cost c to get up to state1. If the problem
        is such that the path doesn't matter, this function will only look at
        state2.  If the path does matter, it will consider c and maybe state1
        and action. The default method costs 1 for every step in the path."""
        return c + 1

    def value(self):
        """For optimization problems, each state has a value.  Hill-climbing
        and related algorithms try to maximize this value."""
        raise NotImplementedError


# ______________________________________________________________________________
# Definiranje na klasa za strukturata na jazel od prebaruvanje
# Klasata Node ne se nasleduva

class Node:
    """A node in a search tree. Contains a pointer to the parent (the node
    that this is a successor of) and to the actual state for this node. Note
    that if a state is arrived at by two paths, then there are two nodes with
    the same state.  Also includes the action that got us to this state, and
    the total path_cost (also known as g) to reach the node.  Other functions
    may add an f and h value; see best_first_graph_search and astar_search for
    an explanation of how the f and h values are handled. You will not need to
    subclass this class."""

    def __init__(self, state, parent=None, action=None, path_cost=0):
        "Create a search tree Node, derived from a parent by an action."
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1

    def __repr__(self):
        return "<Node %s>" % (self.state,)

    def __lt__(self, node):
        return self.state < node.state

    def expand(self, problem):
        "List the nodes reachable in one step from this node."
        return [self.child_node(problem, action)
                for action in problem.actions(self.state)]

    def child_node(self, problem, action):
        "Return a child node from this node"
        next = problem.result(self.state, action)
        return Node(next, self, action,
                    problem.path_cost(self.path_cost, self.state,
                                      action, next))

    def solution(self):
        "Return the sequence of actions to go from the root to this node."
        return [node.action for node in self.path()[1:]]

    def solve(self):
        "Return the sequence of states to go from the root to this node."
        return [node.state for node in self.path()[0:]]

    def path(self):
        "Return a list of nodes forming the path from the root to this node."
        x, result = self, []
        while x:
            result.append(x)
            x = x.parent
        return list(reversed(result))

    # We want for a queue of nodes in breadth_first_search or
    # astar_search to have no duplicated states, so we treat nodes
    # with the same state as equal. [Problem: this may not be what you
    # want in other contexts.]

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __hash__(self):
        return hash(self.state)


# ________________________________________________________________________________________________________
# Neinformirano prebaruvanje vo ramki na drvo
# Vo ramki na drvoto ne razresuvame jamki

def tree_search(problem, fringe):
    """Search through the successors of a problem to find a goal.
    The argument fringe should be an empty queue."""
    fringe.append(Node(problem.initial))
    while fringe:
        node = fringe.pop()
        print((node.state))
        if problem.goal_test(node.state):
            return node
        fringe.extend(node.expand(problem))
    return None


def breadth_first_tree_search(problem):
    "Search the shallowest nodes in the search tree first."
    return tree_search(problem, FIFOQueue())


def depth_first_tree_search(problem):
    "Search the deepest nodes in the search tree first."
    return tree_search(problem, Stack())


# ________________________________________________________________________________________________________
# Neinformirano prebaruvanje vo ramki na graf
# Osnovnata razlika e vo toa sto ovde ne dozvoluvame jamki t.e. povtoruvanje na sostojbi

def graph_search(problem, fringe):
    """Search through the successors of a problem to find a goal.
    The argument fringe should be an empty queue.
    If two paths reach a state, only use the best one."""
    closed = {}
    fringe.append(Node(problem.initial))
    while fringe:
        node = fringe.pop()
        if problem.goal_test(node.state):
            return node
        if node.state not in closed:
            closed[node.state] = True
            fringe.extend(node.expand(problem))
    return None


def breadth_first_graph_search(problem):
    "Search the shallowest nodes in the search tree first."
    return graph_search(problem, FIFOQueue())


def depth_first_graph_search(problem):
    "Search the deepest nodes in the search tree first."
    return graph_search(problem, Stack())


def uniform_cost_search(problem):
    "Search the nodes in the search tree with lowest cost first."
    return graph_search(problem, PriorityQueue(lambda a, b: a.path_cost < b.path_cost))


def depth_limited_search(problem, limit=50):
    "depth first search with limited depth"

    def recursive_dls(node, problem, limit):
        "helper function for depth limited"
        cutoff_occurred = False
        if problem.goal_test(node.state):
            return node
        elif node.depth == limit:
            return 'cutoff'
        else:
            for successor in node.expand(problem):
                result = recursive_dls(successor, problem, limit)
                if result == 'cutoff':
                    cutoff_occurred = True
                elif result is not None:
                    return result
        if cutoff_occurred:
            return 'cutoff'
        else:
            return None

    # Body of depth_limited_search:
    return recursive_dls(Node(problem.initial), problem, limit)


def iterative_deepening_search(problem):
    for depth in range(sys.maxsize):
        result = depth_limited_search(problem, depth)
        if result is not 'cutoff':
            return result


class Stars(Problem):
    def __init__(self, initial, goal=None):
        self.initial = initial
        self.goal = goal

    def successor(self, state):
        successors = dict()
        H_i, H_j, B_i, B_j, stars = state

        """ Horse """
        # H1
        H_i_new, H_j_new = horse1(state)
        if (H_i_new, H_j_new) in stars:
            Statenew = (H_i_new, H_j_new, B_i, B_j,
                        tuple((S_i, S_j) for (S_i, S_j) in stars if S_i != H_i_new and S_j != H_j_new))
        else:
            Statenew = (H_i_new, H_j_new, B_i, B_j, stars)
        successors["Horse1"] = Statenew

        # H2
        H_i_new, H_j_new = horse2(state)
        if (H_i_new, H_j_new) in stars:
            Statenew = (H_i_new, H_j_new, B_i, B_j,
                        tuple((S_i, S_j) for (S_i, S_j) in stars if S_i != H_i_new and S_j != H_j_new))
        else:
            Statenew = (H_i_new, H_j_new, B_i, B_j, stars)
        successors["Horse2"] = Statenew

        # H3
        H_i_new, H_j_new = horse3(state)
        if (H_i_new, H_j_new) in stars:
            Statenew = (H_i_new, H_j_new, B_i, B_j,
                        tuple((S_i, S_j) for (S_i, S_j) in stars if S_i != H_i_new and S_j != H_j_new))
        else:
            Statenew = (H_i_new, H_j_new, B_i, B_j, stars)
        successors["Horse3"] = Statenew

        # H4
        H_i_new, H_j_new = horse4(state)
        if (H_i_new, H_j_new) in stars:
            Statenew = (H_i_new, H_j_new, B_i, B_j,
                        tuple((S_i, S_j) for (S_i, S_j) in stars if S_i != H_i_new and S_j != H_j_new))
        else:
            Statenew = (H_i_new, H_j_new, B_i, B_j, stars)
        successors["Horse4"] = Statenew

        # H5
        H_i_new, H_j_new = horse5(state)
        if (H_i_new, H_j_new) in stars:
            Statenew = (H_i_new, H_j_new, B_i, B_j,
                        tuple((S_i, S_j) for (S_i, S_j) in stars if S_i != H_i_new and S_j != H_j_new))
        else:
            Statenew = (H_i_new, H_j_new, B_i, B_j, stars)
        successors["Horse5"] = Statenew

        # H6
        H_i_new, H_j_new = horse6(state)
        if (H_i_new, H_j_new) in stars:
            Statenew = (H_i_new, H_j_new, B_i, B_j,
                        tuple((S_i, S_j) for (S_i, S_j) in stars if S_i != H_i_new and S_j != H_j_new))
        else:
            Statenew = (H_i_new, H_j_new, B_i, B_j, stars)
        successors["Horse6"] = Statenew

        # H7
        H_i_new, H_j_new = horse7(state)
        if (H_i_new, H_j_new) in stars:
            Statenew = (H_i_new, H_j_new, B_i, B_j,
                        tuple((S_i, S_j) for (S_i, S_j) in stars if S_i != H_i_new and S_j != H_j_new))
        else:
            Statenew = (H_i_new, H_j_new, B_i, B_j, stars)
        successors["Horse7"] = Statenew

        # H8
        H_i_new, H_j_new = horse8(state)
        if (H_i_new, H_j_new) in stars:
            Statenew = (H_i_new, H_j_new, B_i, B_j,
                        tuple((S_i, S_j) for (S_i, S_j) in stars if S_i != H_i_new and S_j != H_j_new))
        else:
            Statenew = (H_i_new, H_j_new, B_i, B_j, stars)
        successors["Horse8"] = Statenew

        """ Bishop """
        # B1
        B_i_new, B_j_new = bishop1(state)
        if (B_i_new, B_j_new) in stars:
            Statenew = (H_i, H_j, B_i_new, B_j_new,
                        tuple((S_i, S_j) for (S_i, S_j) in stars if S_i != B_i_new and S_j != B_j_new))
        else:
            Statenew = (H_i, H_j, B_i_new, B_j_new, stars)
        successors["Bishop1"] = Statenew

        # B2
        B_i_new, B_j_new = bishop2(state)
        if (B_i_new, B_j_new) in stars:
            Statenew = (H_i, H_j, B_i_new, B_j_new,
                        tuple((S_i, S_j) for (S_i, S_j) in stars if S_i != B_i_new and S_j != B_j_new))
        else:
            Statenew = (H_i, H_j, B_i_new, B_j_new, stars)
        successors["Bishop2"] = Statenew

        # B3
        B_i_new, B_j_new = bishop3(state)
        if (B_i_new, B_j_new) in stars:
            Statenew = (H_i, H_j, B_i_new, B_j_new,
                        tuple((S_i, S_j) for (S_i, S_j) in stars if S_i != B_i_new and S_j != B_j_new))
        else:
            Statenew = (H_i, H_j, B_i_new, B_j_new, stars)
        successors["Bishop3"] = Statenew

        # B4
        B_i_new, B_j_new = bishop4(state)
        if (B_i_new, B_j_new) in stars:
            Statenew = (H_i, H_j, B_i_new, B_j_new,
                        tuple((S_i, S_j) for (S_i, S_j) in stars if S_i != B_i_new and S_j != B_j_new))
        else:
            Statenew = (H_i, H_j, B_i_new, B_j_new, stars)
        successors["Bishop4"] = Statenew

        return successors

    def actions(self, state):
        return self.successor(state).keys()

    def result(self, state, action):
        possible = self.successor(state)
        return possible[action]

    def goal_test(self, state):
        stars = state[4]
        return len(stars) == 0

    def value(self):
        raise NotImplementedError


# up + left
def horse1(state):
    H_i, H_j = state[0], state[1]

    if 0 <= (H_i - 2) <= 7 and 0 <= (H_j - 1) <= 7 and \
            (H_i - 2, H_j - 1) != (state[2], state[3]):
        H_i -= 2
        H_j -= 1

    return H_i, H_j


# up + right
def horse2(state):
    H_i, H_j = state[0], state[1]

    if 0 <= (H_i - 2) <= 7 and 0 <= (H_j + 1) <= 7 and \
            (H_i - 2, H_j + 1) != (state[2], state[3]):
        H_i -= 2
        H_j += 1

    return H_i, H_j


# right + up
def horse3(state):
    H_i, H_j = state[0], state[1]

    if 0 <= (H_i - 1) <= 7 and 0 <= (H_j + 2) <= 7 and \
            (H_i - 1, H_j + 2) != (state[2], state[3]):
        H_i -= 1
        H_j += 2

    return H_i, H_j


# right + down
def horse4(state):
    H_i, H_j = state[0], state[1]

    if 0 <= (H_i + 1) <= 7 and 0 <= (H_j + 2) <= 7 and \
            (H_i + 1, H_j + 2) != (state[2], state[3]):
        H_i += 1
        H_j += 2

    return H_i, H_j


# down + right
def horse5(state):
    H_i, H_j = state[0], state[1]

    if 0 <= (H_i + 2) <= 7 and 0 <= (H_j + 1) <= 7 and \
            (H_i + 2, H_j + 1) != (state[2], state[3]):
        H_i += 2
        H_j += 1

    return H_i, H_j


# down + left
def horse6(state):
    H_i, H_j = state[0], state[1]

    if 0 <= (H_i + 2) <= 7 and 0 <= (H_j - 1) <= 7 and \
            (H_i + 2, H_j - 1) != (state[2], state[3]):
        H_i += 2
        H_j -= 1

    return H_i, H_j


# left + down
def horse7(state):
    H_i, H_j = state[0], state[1]

    if 0 <= (H_i + 1) <= 7 and 0 <= (H_j - 2) <= 7 and \
            (H_i + 1, H_j - 2) != (state[2], state[3]):
        H_i += 1
        H_j -= 2

    return H_i, H_j


# left + up
def horse8(state):
    H_i, H_j = state[0], state[1]

    if 0 <= (H_i - 1) <= 7 and 0 <= (H_j - 2) <= 7 and \
            (H_i - 1, H_j - 2) != (state[2], state[3]):
        H_i -= 1
        H_j -= 2

    return H_i, H_j


# up + left
def bishop1(state):
    B_i, B_j = state[2], state[3]
    if 0 <= (B_i - 1) <= 7 and 0 < (B_j - 1) < 7 and \
            (B_i - 1, B_j - 1) != (state[0], state[1]):
        B_i -= 1
        B_j -= 1

    return B_i, B_j


# up + right
def bishop2(state):
    B_i, B_j = state[2], state[3]
    if 0 <= (B_i - 1) <= 7 and 0 < (B_j + 1) < 7 and \
            (B_i - 1, B_j + 1) != (state[0], state[1]):
        B_i -= 1
        B_j += 1

    return B_i, B_j


# down + left
def bishop3(state):
    B_i, B_j = state[2], state[3]
    if 0 <= (B_i + 1) <= 7 and 0 < (B_j - 1) < 7 and \
            (B_i + 1, B_j - 1) != (state[0], state[1]):
        B_i += 1
        B_j -= 1

    return B_i, B_j


# down + right
def bishop4(state):
    B_i, B_j = state[2], state[3]
    if 0 <= (B_i + 1) <= 7 and 0 < (B_j + 1) < 7 and \
            (B_i + 1, B_j + 1) != (state[0], state[1]):
        B_i += 1
        B_j += 1

    return B_i, B_j


if __name__ == "__main__":
    KRedica = int(input())
    KKolona = int(input())
    LRedica = int(input())
    LKolona = int(input())
    Z1Redica = int(input())
    Z1Kolona = int(input())
    Z2Redica = int(input())
    Z2Kolona = int(input())
    Z3Redica = int(input())
    Z3Kolona = int(input())

    stars = ((Z1Redica, Z1Kolona), (Z2Redica, Z2Kolona), (Z3Redica, Z3Kolona))
    initial_state = (KRedica, KKolona, LRedica, LKolona, stars)
    problem = Stars(initial_state)

    answer = breadth_first_graph_search(problem)
    print(answer.solution())

    pass
