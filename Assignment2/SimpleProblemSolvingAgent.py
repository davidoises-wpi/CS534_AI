from search import *

def my_hill_climbing(problem):
    """
    [Figure 4.2]
    From the initial node, keep choosing the neighbor with highest value,
    stopping when no neighbor is better.
    """
    current = Node(problem.initial)
    while True:
        neighbors = current.expand(problem)
        if not neighbors:
            break
        neighbor = argmax_random_tie(neighbors, key=lambda node: problem.value(node.state))
        if problem.value(neighbor.state) <= problem.value(current.state):
            break
        current = neighbor

    return current

def my_simulated_annealing(problem, schedule=exp_schedule()):
    """ This version returns all the states encountered in reaching 
    the goal state."""
    current = Node(problem.initial)
    for t in range(sys.maxsize):
        T = schedule(t)
        if T == 0:
            return current
        neighbors = current.expand(problem)
        if not neighbors:
            return current
        next_choice = random.choice(neighbors)
        delta_e = problem.value(next_choice.state) - problem.value(current.state)
        if delta_e > 0 or probability(np.exp(delta_e / T)):
            current = next_choice

class TravelingSalesmanProblem(Problem):
    """The problem of searching a graph from one node to another."""

    def __init__(self, initial, goal, graph):
        super().__init__(initial, goal)
        self.graph = graph

    def actions(self, A):
        """The actions at a graph node are just its neighbors."""
        return list(self.graph.get(A).keys())

    def result(self, state, action):
        """The result of going to a neighbor is just that neighbor."""
        return action

    def path_cost(self, cost_so_far, A, action, B):
        return cost_so_far + (self.graph.get(A, B) or np.inf)

    def find_min_edge(self):
        """Find minimum value of edges."""
        m = np.inf
        for d in self.graph.graph_dict.values():
            local_min = min(d.values())
            m = min(m, local_min)

        return m

    def h(self, node):
        """h function is straight-line distance from a node's state to goal."""
        locs = getattr(self.graph, 'locations', None)
        if locs:
            if type(node) is str:
                return int(distance(locs[node], locs[self.goal]))

            return int(distance(locs[node.state], locs[self.goal]))
        else:
            return np.inf
    
    def value(self, state):
        """For optimization problems, each state has a value. Hill Climbing
        and related algorithms try to maximize this value."""
        locs = getattr(self.graph, 'locations', None)
        straigh_line_distance_initial_to_goal = int(distance(locs[self.initial], locs[self.goal]))

        straigh_line_distance_current_to_goal = self.h(state)

        return (straigh_line_distance_initial_to_goal - straigh_line_distance_current_to_goal)
    
class SimpleProblemSolvingAgent(SimpleProblemSolvingAgentProgram):
    """
    Attributes:
    state - The name of the city in which the agent currently is
    goal - The name fo the city to which the agent wants to go
    grapth - An object with the map graph and the locations of each city
    """

    def __init__(self, initial_state, goal, graph, search_algorithm='Greedy'):
        """State is an abstract representation of the state
        of the world, and seq is the list of actions required
        to get to a particular state from the initial state(root)."""
        self.state = initial_state
        self.goal = goal
        self.graph = graph
        self.search_algorithm = search_algorithm

    def __call__(self):
        """[Figure 3.1] Formulate a goal and problem, then
        search for a sequence of actions to solve it.
        
        Modified from original repo, once the search function returns the
        goal node, we track back all the parents of that node to regenerate
        the path that lead to the goal"""

        goal = self.formulate_goal()
        problem = self.formulate_problem(self.state, goal)
        resulting_node = self.search(problem)

        if not resulting_node:
            return None
        
        path = []
        path.insert(0, resulting_node)

        next_node = resulting_node.parent
        while next_node:
            path.insert(0, next_node)
            next_node = next_node.parent
        
        return path

    def update_state(self, percept):
        return percept

    def formulate_goal(self):
        return self.goal

    def formulate_problem(self, state, goal):
        TSP = TravelingSalesmanProblem(state, goal, self.graph)
        return TSP

    def search(self, problem):
        """
        When this function is called, problem contains in itself:
        initial - Name of the starting city
        goal - Name of the destination city
        graph - Map graph and each city location

        Returns a linked list of child and parent nodes starting from the goal.
        """
        if self.search_algorithm == 'Greedy':
            return best_first_graph_search(problem, problem.h, False)
        elif self.search_algorithm == 'Astar':
            return astar_search(problem, problem.h, False)
        elif self.search_algorithm == 'Hill_climbing':
            return my_hill_climbing(problem)
        elif self.search_algorithm == 'Simulated_annealing':
            return my_simulated_annealing(problem)
        
        return None
