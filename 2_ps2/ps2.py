# 6.100B Problem Set 2 Fall 2023
# Graph Optimization
# Name: Rafael Moreno Ribeiro
# Collaborators: None

# Problem Set 2
# =============
# Finding shortest paths to drive from home to work on a road network

from graph import DirectedRoad, Node, RoadMap


# PROBLEM 2: Building the Road Network
#
# PROBLEM 2.1: Designing your Graph
#
#   What do the graph's nodes represent in this problem? What
#   do the graph's edges represent? Where are the times
#   represented?
#
# Write your answer below as a comment:
# The nodes represent different crossroads and points along your journey; The edges are the roads connecting different nodes; The times represent the weight of each edge of the graph
#

# PROBLEM 2.2: Implementing create_graph
def create_graph(map_filename):
    """
    Parses the map file and constructs a road map (graph).

    Travel time and traffic multiplier should be each cast to a float.

    Parameters:
        map_filename : str
            Name of the map file.

    Assumes:
        Each entry in the map file consists of the following format, separated by spaces:
            source_node destination_node travel_time road_type traffic_multiplier

        Note: hill road types specified in the text file represent travel time uphill
              in the source to destination direction. Downhill travel takes 1/3 as long
              as uphill travel.

        e.g.
            N0 N1 10 highway 1
        This entry would become two directed roads; one from 'N0' to 'N1' on a highway with
        a weight of 10.0, and another road from 'N1' to 'N0' on a highway using the same weight.

        e.g.
            N2 N3 9 hill 2
        This entry would become two directed roads; one from 'N2' to 'N3' on a hill road with
        a weight of 9.0, and another road from 'N3' to 'N2' on a hill road with a weight of 3.0.
        Note that the directed roads created should both have type 'hill', not 'uphill'/ 'downhill'!

    Returns:
        RoadMap
            A directed road map representing the given map.
    """
    # Initiate RoadMap instance
    road_map = RoadMap()

    # Open the file
    with open(map_filename, 'r') as file:

        # Read through every line
        for line in file:
            # Convert str into list, excluding possible spaces in the beginning/end and splitting by spaces
            data = line.strip().split()

            # Each entry of data corresponds to a different information, namely:
            source_node = Node(data[0])
            destination_node = Node(data[1])
            time = int(data[2])
            type = data[3]
            multiplier = int(data[4])

            # Check that source node is not yet in road_map
            # If not, adds it
            if source_node not in road_map.get_all_nodes():
                road_map.insert_node(source_node)

            # Check that destination node is not yet in road_map
            # If not, adds it
            if destination_node not in road_map.get_all_nodes():
                road_map.insert_node(destination_node)

            # For each road in file, create DirectedRoad instance from source to destination and vice-versa and add it to road_map

            # For hill type, we must treat each orientation of the road differently
            # Uphill = time -> Downhill = time/3
            if type == 'hill':
                road_uphill = DirectedRoad(source_node, destination_node, time, type, multiplier)
                road_downhill = DirectedRoad(destination_node, source_node, time/3, type, multiplier)
                road_map.insert_road(road_uphill)
                road_map.insert_road(road_downhill)

            else:
                road_going = DirectedRoad(source_node, destination_node, time, type, multiplier)
                road_returning = DirectedRoad(destination_node, source_node, time, type, multiplier)
                road_map.insert_road(road_going)
                road_map.insert_road(road_returning)

    return road_map


# PROBLEM 2.3: Testing create_graph
#
#   Go to the bottom of this file, look for the section under FOR PROBLEM 2.3,
#   and follow the instructions in the handout.


# PROBLEM 3: Finding the Shortest Path using Depth-First Search

# Problem 3.1: Objective function
#
#   What is the objective function for this problem? What are the constraints?
#
# Answer: We want to find the shortest path possible between two nodes, constrained to the non-restricted roads that are given to us
#
#
#

# PROBLEM 3.2: Implement find_shortest_path
def find_shortest_path(roadmap, start, end, restricted_roads=None, has_traffic=False):
    """
    Finds the shortest path between start and end nodes on the road map,
    without using any restricted roads, following traffic conditions.
    If restricted_roads is None, assume there are no restricted roads.
    Use the depth first search algorithm (DFS).

    Parameters:
        roadmap: RoadMap
            The graph on which to carry out the search.
        start: Node
            Node at which to start.
        end: Node
            Node at which to end.
        restricted_roads: list of str or None
            Road Types not allowed on path. If None, all are roads allowed
        has_traffic: bool
            Flag to indicate whether to get shortest path during traffic or not.

    Returns:
        A two element tuple of the form (best_path, best_time).
            The first item is a list of Nodes, the shortest path from start to end.
            The second item is a float, the length (time traveled) of the best path.
        If there exists no path that satisfies constraints, then return None.
    """
    # Let's make use of a helper function to implement our DFS algorithm
    # It has been modified to take road times into consideration as weights
    def dfs_timed_with_shortest(graph, source, target, current_path, current_time, best_path, best_time, restricted_roads):

        # Check if current source node is the target node
        if source == target:

             # Check if the current path is better than the best path found so far
            if not best_path or current_time < best_time:
                return current_path, current_time
            else:
                return best_path, best_time

        # Iterate through every reachable road from the current source node
        for road in graph.get_reachable_roads_from_node(source, restricted_roads):
            next_node = road.get_destination_node()

            # Check if the next node is not already in the current path, to avoid cycles
            if next_node not in current_path:

                # Find new time by adding the travel time of current road
                new_time = current_time + road.get_travel_time(has_traffic)

                # Recursive call to explore the next node
                new_path, new_time = dfs_timed_with_shortest(graph, next_node, target, current_path + [next_node], new_time, best_path, best_time, restricted_roads)

                # Update best path and best time if a shorter path is found
                if new_path is not None and (best_path is None or new_time < best_time):
                    best_path, best_time = new_path, new_time

        # Return tuple containing best_path and best_time
        return best_path, best_time

    # Initialize restricted_roads if None has been imputed instead of []
    if restricted_roads is None:
        restricted_roads = []

    # Perform DFS with time tracking and shortest path identification
    best_path, best_time = dfs_timed_with_shortest(roadmap, start, end, [start], 0.0, None, float('inf'), restricted_roads)

    # If no valid path was found, return None
    if best_path == None:
        return None

    # Else, return the best path found and its corresponding time
    else:
        return best_path, best_time


# PROBLEM 4.1: Implement find_shortest_path_no_traffic
def find_shortest_path_no_traffic(filename, start, end):
    """
    Finds the shortest path from start to end during conditions of no traffic.
    Assume there are no restricted roads.

    You must use find_shortest_path.

    Parameters:
        filename: str
            Name of the map file that contains the graph
        start: Node
            Node object at which to start.
        end: Node
            Node object at which to end.

    Returns:
        A two element tuple of the form (best_path, best_time).
            The first item is a list of Nodes, the shortest path from start to end with no traffic.
            The second item is a float, the length (time traveled) of the best path.
        If there exists no path that satisfies constraints, then return None.
    """
    road_map = create_graph(filename)
    return find_shortest_path(road_map, start, end, None, False)


# PROBLEM 4.2: Implement find_shortest_path_restricted
def find_shortest_path_restricted(filename, start, end):
    """
    Finds the shortest path from start to end when local roads and hill roads cannot be used.
    Assume no traffic.

    You must use find_shortest_path.

    Parameters:
        filename: str
            Name of the map file that contains the graph
        start: Node
            Node object at which to start.
        end: Node
            Node object at which to end.

    Returns:
        A two element tuple of the form (best_path, best_time).
            The first item is a list of Nodes, the shortest path from start to end given the aforementioned conditions.
            The second item is a float, the length (time traveled) of the best path.
        If there exists no path that satisfies constraints, then return None.
    """
    road_map = create_graph(filename)
    return find_shortest_path(road_map, start, end, ["local", "hill"], False)


# PROBLEM 4.3: Implement find_shortest_path_in_traffic
def find_shortest_path_in_traffic(filename, start, end):
    """
    Finds the shortest path from start to end in traffic,
    i.e. when all roads' travel times are multiplied by their traffic multipliers.

    You must use find_shortest_path.

    Parameters:
        filename: str
            Name of the map file that contains the graph
        start: Node
            Node object at which to start.
        end: Node
            Node object at which to end.

    Returns:
        A two element tuple of the form (best_path, best_time).
            The first item is a list of Nodes, the shortest path from start to end given the aforementioned conditions.
            The second item is a float, the length (time traveled) of the best path.
        If there exists no path that satisfies constraints, then return None.
    """
    road_map = create_graph(filename)
    return find_shortest_path(road_map, start, end, None, True)


if __name__ == '__main__':

    # UNCOMMENT THE LINES BELOW TO DEBUG OR TO EXECUTE PROBLEM 2.3
    pass

    # small_map = create_graph('./maps/small_map.txt')

    # # ------------------------------------------------------------------------
    # # FOR PROBLEM 2.3
    #road_map = create_graph("maps/test_create_graph.txt")
    #print(road_map)
    # # ------------------------------------------------------------------------

    # start = Node('N0')
    # end = Node('N4')
    # restricted_roads = []
    # print(find_shortest_path(small_map, start, end, restricted_roads))
