from queue import PriorityQueue

def printPath(node, parent):
    if parent[node] != -1:
        printPath(parent[node], parent)
        print(" -> ", end="")
    print(node, end="")

def UCS(graph, start, goal_set, n_nodes):
    visited = [False] * n_nodes
    parent = [-1] * n_nodes
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
                    parent[neighbour] = curr_node
                    pq.put((new_dist,neighbour))
    
    return -1

graph = {
    0: [(1, 1), (2, 4)],
    1: [(2, 2), (3, 5)],
    2: [(4, 3)],
    3: [(5, 2), (6, 1)],
    4: [(6, 4), (3, 1)],
    5: [],
    6: [(7, 6)],
    7: []
}

start = 0
goal_set = {5,7}
n_nodes = 8

print(f"Running UCS on the given graph for Start: {start} and Goal Set: {goal_set}")
print(f"\nResulting Cost: {UCS(graph,start,goal_set,n_nodes)}")