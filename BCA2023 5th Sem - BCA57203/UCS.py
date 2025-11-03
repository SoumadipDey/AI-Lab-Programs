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
    0: [(1, 5), (2, 10)],
    1: [(0, 5), (3, 200), (4, 40)],
    2: [(0, 10), (3, 8), (6, 20)],
    3: [(1, 200), (2, 8), (5, 13), (6, 15)],
    4: [(1, 40), (5, 1), (7, 9)],
    5: [(3, 13), (4, 1), (6, 6), (7, 5)],
    6: [(2, 20), (3, 15), (5, 6), (7, 2), (8, 18)],
    7: [(4, 9), (5, 5), (6, 2), (8, 1)],
    8: [(6, 18), (7, 1)],
}

start = 0
goal_set = {8}
n_nodes = len(graph)

print(f"Running UCS on the given graph for Start: {start} and Goal Set: {goal_set}")
print(f"\nResulting Cost: {UCS(graph,start,goal_set,n_nodes)}")