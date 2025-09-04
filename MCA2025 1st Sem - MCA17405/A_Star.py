from queue import PriorityQueue

def printPath(node, parent):
    if parent[node] != -1:
        printPath(parent[node], parent)
        print(" -> ", end="")
    print(node, end="")

def AStarSearch(graph, heuristics, start, goal_set, n_nodes):
    visited = [False] * n_nodes
    parent = [-1] * n_nodes
    distances = [float('inf')] * n_nodes

    pq = PriorityQueue()
    distances[start] = 0
    pq.put((distances[start]+heuristics[start],distances[start],start))

    while not pq.empty():
        _, actual_dist, curr_node = pq.get()
        if not visited[curr_node]:
            visited[curr_node] = True
            if curr_node in goal_set:
                print("Path is: ")
                printPath(curr_node,parent)
                return actual_dist
            for neighbour, cost in graph[curr_node]:
                if not visited[neighbour]:
                    new_actual_dist = actual_dist + cost
                    new_pred_dist = new_actual_dist + heuristics[neighbour]
                    if new_actual_dist < distances[neighbour]:
                        distances[neighbour] = new_actual_dist
                        parent[neighbour] = curr_node
                        pq.put((new_pred_dist,new_actual_dist,neighbour))
    
    return -1

graph = {
    0: [(1, 4), (2, 3)],
    1: [(0, 4), (4, 12), (5, 5)],
    2: [(0, 3), (3, 7), (4, 10)],
    3: [(2, 7), (4, 2)],
    4: [(2, 10), (3, 2), (1, 12), (6, 5)],
    5: [(1, 5), (6, 16)],
    6: [(4, 5), (5, 16)]
}
heuristics = {
    0: 14,
    1: 12,
    2: 11,
    3: 6,
    4: 4,
    5: 11,
    6: 0
}
start = 0
goal_set = {6}
n_nodes = len(graph)

print(f"Running A Star on the given graph for Start: {start} and Goal Set: {goal_set}")
print(f"\nResulting Cost: {AStarSearch(graph,heuristics,start,goal_set,n_nodes)}")