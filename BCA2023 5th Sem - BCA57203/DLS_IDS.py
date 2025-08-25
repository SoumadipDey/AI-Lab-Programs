def DLS(graph, start, goal, visited, limit):
    print(f"DLS(start = {start}, goal= {goal}, limit = {limit})")
    if(start == goal): return 1
    elif(limit == 0): return -1
    else:
        visited[start] = True
        cutoff_flag = False
        for neighbour in graph[start]:
            if not visited[neighbour]:
                result = DLS(graph, neighbour, goal, visited, limit - 1)
                if result == 1:
                    return 1
                elif result == -1:
                    cutoff_flag = True
        
        return -1 if cutoff_flag else 0

def IDS(graph, start, goal, n_nodes):
    limit = 0
    while True:
        visited = [False] * n_nodes
        print(f"Current Depth Level: {limit}")
        result = DLS(graph, start, goal, visited, limit)
        print()
        if result != -1:
            if(result == 0):
                return "Not Found!"
            else:
                return f"Found at Depth Level {limit}"
        limit += 1


graph = {
    0: [1, 2],
    1: [0, 3, 4],
    2: [0, 5, 6],
    3: [1],
    4: [1, 7],
    5: [2],
    6: [2],
    7: [4, 8, 9],
    8: [7, 10, 11],
    9: [7],
    10: [8],
    11: [8]
}

n_nodes = len(graph)
start = 0
goal = 56

print(f"The result of running IDS on the graph for Start: {start}, Goal: {goal} is: {IDS(graph,start,goal,n_nodes)}")