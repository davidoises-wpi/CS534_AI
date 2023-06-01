
"""Hello World

Initial basic script for CS534 assignment 1.
Requests user input for the user's name and then prints it back
saying hello.

Author: David Dominguez
Date: 05/31/2023
"""

def main():
    print("Hello World")

    # Receive input from the user and store the provided name
    name = input("Enter your name: ")

    # Print the concatenated stirng "hello" + the user name
    print("Hello " + name)

if __name__ == "__main__":
    main()