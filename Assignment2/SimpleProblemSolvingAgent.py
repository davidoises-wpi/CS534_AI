from search import *

def node_hill_climbing(problem):
    """
    [Figure 4.2]
    From the initial node, keep choosing the neighbor with highest value,
    stopping when no neighbor is better.

    Returns the node object with links to parent nodes that form the path.
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

def node_simulated_annealing(problem, schedule=exp_schedule()):
    """
    [Figure 4.5]
    From the initial node, keep choosing a random neighbor,
    add it to the explored path if it has a higer value or the temperature probability allows it,
    stop when temperature is 0.

    Returns the node object with links to parent nodes that form the path.
    """
    
    current = Node(problem.initial)
    for t in range(sys.maxsize):
        T = schedule(t)
        if T == 0:
            return current
        neighbors = current.expand(problem)
        if not neighbors:
            return current
        next_choice = random.choice(neighbors)
        # Note that this subtraction is in the correct order since problem.value is higher
        # as it gets closer to the goal. The search algorithm optimizes for highers value.
        delta_e = problem.value(next_choice.state) - problem.value(current.state)
        if delta_e > 0 or probability(np.exp(delta_e / T)):
            current = next_choice

class TravelingSalesmanProblem(Problem):
    """The problem of searching a graph from one node to another.

    Custom definition of GraphProblem but with an extra funcion value(state).
    This function returns a bigger number when a node is closer to the goal. Used
    for optimization in local search algorithms.
    """

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
    Used for searching problems with different search algorithms.
    """

    def __init__(self, initial_state, goal, graph, search_algorithm='Greedy'):
        """
        state - The name of the city in which the agent currently is
        goal - The name fo the city to which the agent wants to go
        grapth - An object with the map graph and the locations of each city
        search_algorithm - The algorithm to be used to solve the search problem
        """
        self.state = initial_state
        self.goal = goal
        self.graph = graph
        self.search_algorithm = search_algorithm

    def __call__(self):
        """
        [Figure 3.1] Formulate a goal and problem, then
        search for a sequence of actions to solve it.
        
        It traces the parent nodes that lead to the goal and returns the
        traveled path.
        """

        # Formulate goal and problem, then find the solution
        goal = self.formulate_goal()
        problem = self.formulate_problem(self.state, goal)
        resulting_node = self.search(problem)
        if not resulting_node:
            return None
        
        # List of nodes that led to the goal
        path = []

        # The search algorithm returns the goal node, therefore keep inserting
        # parent nodes at the begining of the path to get the correct order from
        # start to end
        path.insert(0, resulting_node)

        # Explore all parent nodes in link list of cities
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
        Returns a linked list of child and parent nodes starting from the goal.

        Depending on the defined search algorithm this calls different search functions
        """
        if self.search_algorithm == 'Greedy':
            return best_first_graph_search(problem, problem.h, False)
        elif self.search_algorithm == 'Astar':
            return astar_search(problem, problem.h, False)
        elif self.search_algorithm == 'Hill_climbing':
            return node_hill_climbing(problem)
        elif self.search_algorithm == 'Simulated_annealing':
            return node_simulated_annealing(problem)
        
        return None
