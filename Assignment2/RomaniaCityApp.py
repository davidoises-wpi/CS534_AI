from search import *
from SimpleProblemSolvingAgent import *

def main():
    greedy = SimpleProblemSolvingAgent('Arad', 'Bucharest', romania_map, 'Greedy')

    resulting_path = greedy()
    print("The path taken was: ", end="")
    print(resulting_path)
    print("The path cost was: ", end="")
    print(resulting_path[-1].path_cost)

    greedy = SimpleProblemSolvingAgent('Arad', 'Bucharest', romania_map, 'Astar')

    resulting_path = greedy()
    print("The path taken was: ", end="")
    print(resulting_path)
    print("The path cost was: ", end="")
    print(resulting_path[-1].path_cost)

    greedy = SimpleProblemSolvingAgent('Arad', 'Bucharest', romania_map, 'Hill_climbing')

    resulting_path = greedy()
    print("The path taken was: ", end="")
    print(resulting_path)
    print("The path cost was: ", end="")
    print(resulting_path[-1].path_cost)

if __name__ == "__main__":
    main()