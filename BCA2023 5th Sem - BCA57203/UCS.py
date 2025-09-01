from queue import PriorityQueue

def printPath(node, parent):
    if parent[node] != -1:
        printPath(parent[node], parent)
        print(" -> ", end="")
    print(node, end="")

def UCS(graph, start, goal_set, n_nodes):
    visited = [False] * n_nodes
    parent = [-1] * n_nodes
    distances = [float('inf')] * n_nodes
    pq = PriorityQueue()
    pq.put((0,start))

    while not pq.empty():
        dist, curr_node = pq.get()
        if not visited[curr_node]:
            visited[curr_node] = True
            if curr_node in goal_set:
                print("Path is: ")
                printPath(curr_node,parent)
                return dist
            for neighbour, cost in graph[curr_node]:
                if not visited[neighbour]:
                    new_dist = dist + cost
                    if new_dist < distances[neighbour]:
                        distances[neighbour] = new_dist
                        parent[neighbour] = curr_node
                        pq.put((new_dist,neighbour))
    
    return -1

graph = {
    0: [(1, 4), (2, 2)],
    1: [(3, 1), (4, 2), (7, 3)],
    2: [(1, 1)],
    3: [(0, 3), (4, 3)],
    4: [(5, 2), (6, 3), (7, 2), (8, 4)],
    5: [(6, 2), (8, 1), (9, 2)],
    6: [(9, 1)],
    7: [(2, 3), (8, 2)],
    8: [],
    9: [(8, 1)]
}

start = 0
goal_set = {8,9}
n_nodes = len(graph)

print(f"Running UCS on the given graph for Start: {start} and Goal Set: {goal_set}")
print(f"\nResulting Cost: {UCS(graph,start,goal_set,n_nodes)}")