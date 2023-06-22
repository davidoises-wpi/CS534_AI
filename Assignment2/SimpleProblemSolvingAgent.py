from search import *
    
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
        TSP = GraphProblem(state, goal, self.graph)
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
        
        return None
