
from search import romania_map
from SimpleProblemSolvingAgent import SimpleProblemSolvingAgent

def display_path(path):
    """
    This function takes in a list of nodes and returns
    a string showing the path in a sequential way and linking
    nodes with arrows ->.
    """

    path_string = ""
    for node in path:
        # Append each node name to the string along with an arrow
        path_string += node.state + " -> "

    # Return the full string without the last arrow characters
    return path_string[:-4]

def run_problem_solving_agents(origin, destination):
    """
    This function runs 4 different types of search algorithms to
    solve the traveling salesman problem and displays their results:
    1. Greedy best first search
    2. A* search
    3. Hill climbing search
    4. Simulated annealing search
    """

    # Create a greedy problem solving agent
    greedy = SimpleProblemSolvingAgent(origin, destination, romania_map, 'Greedy')

    # Dispaly search results
    resulting_path = greedy()
    print("Greedy Best-First Search")
    print(display_path(resulting_path))
    print("Total Cost: ", end="")
    print(resulting_path[-1].path_cost)
    print()

    # Create an A* problem solving agent
    astar = SimpleProblemSolvingAgent(origin, destination, romania_map, 'Astar')

    # Dispaly search results
    resulting_path = astar()
    print("A* Search")
    print(display_path(resulting_path))
    print("Total Cost: ", end="")
    print(resulting_path[-1].path_cost)
    print()

    # Create a hill climbing problem solving agent
    hill_climbing = SimpleProblemSolvingAgent(origin, destination, romania_map, 'Hill_climbing')

    # Dispaly search results
    resulting_path = hill_climbing()
    print("Hill Climbing Search")
    print(display_path(resulting_path))
    print("Total Cost: ", end="")
    print(resulting_path[-1].path_cost)
    print()

    # Create a simulated annealing problem solving agent
    simulated_annealing = SimpleProblemSolvingAgent(origin, destination, romania_map, 'Simulated_annealing')

    # Dispaly search results
    resulting_path = simulated_annealing()
    print("Simulated Annealing Search")
    print(display_path(resulting_path))
    print("Total Cost: ", end="")
    print(resulting_path[-1].path_cost)
    print()

def valid_user_input(city1, city2):
    """
    This function checks if the provided cities are valid.

    Valid input meets:
    The first city provided as argument exists in the map. (This helps prevent double checking).
    City 1 and city 2 are diferent.
    """

    # Check if city1 is in the map
    if city1 not in romania_map.graph_dict.keys():
        error_string = "Could not find " + city1 + ", please try again: "
        return (False, error_string)
    
    # Checkf if origin and destination are the same
    if city1 == city2:
        error_string = "The same city can't be both origin and destination, please try again: "
        return (False, error_string)
    
    return (True, "")

def main():

    keep_running = True

    while keep_running:

        # Empty strings to hold city names
        origin = ""
        destination = ""

        # Get origin city
        origin = input("Please enter the origin city: ")
        # Check if valid
        (valid, prompt) = valid_user_input(origin, destination)
        # Keep requesting user input until valid
        while not valid:
            origin = input(prompt)
            (valid, prompt) = valid_user_input(origin, destination)

        # Get destination city
        destination = input("Please enter the destination city: ")
        # Check if valid
        (valid, prompt) = valid_user_input(destination, origin)
        # Keep requesting user input until valid
        while not valid:
            destination = input(prompt)
            (valid, prompt) = valid_user_input(destination, origin)

        print()

        # Find the solutions with the solving agent
        run_problem_solving_agents(origin, destination)

        # Ask if user wants to keep running the program
        retry_request = input("Would you like to find the best path between another two cities? ")
        if not (retry_request == "yes" or retry_request == "Yes"):
            keep_running = False

    # Exit program
    print("Thank You for Using Our App")

if __name__ == "__main__":
    main()