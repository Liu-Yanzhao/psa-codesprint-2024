import multiprocessing as mp
from datetime import datetime
import heapq
from typing import List, Dict, Tuple

class Spot:
    def __init__(self, x: int, y: int, t: int):
        self.x = x
        self.y = y
        self.t = t
        self.g = float('inf')
        self.h = 0
        self.f = 0
        self.parent = None
        self.neighbors = []
        self.wall = False
        self.charge = False
        self.park = False
    
    def __lt__(self, other):
        return self.f < other.f

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.t == other.t

    def __hash__(self):
        return hash((self.x, self.y, self.t))

def create_grid_and_nodes(shape: tuple) -> Dict[tuple, Spot]:
    nodes = {}
    for x in range(shape[0]):
        for y in range(shape[1]):
            for t in range(shape[2]):
                nodes[(x, y, t)] = Spot(x, y, t)
    return nodes

def compute_neighbors(nodes: Dict[tuple, Spot], shape: tuple):
    directions = [(0, 1, 1), (1, 0, 1), (0, -1, 1), (-1, 0, 1), (0, 0, 1), (1, 1, 1), (-1, -1, 1), (1, -1, 1), (-1, 1, 1)]
    for (x, y, t), node in nodes.items():
        for dx, dy, dt in directions:
            new_x, new_y, new_t = x + dx, y + dy, t + dt
            if 0 <= new_x < shape[0] and 0 <= new_y < shape[1] and 0 <= new_t < shape[2]:
                neighbor = nodes.get((new_x, new_y, new_t))
                if neighbor and (not (neighbor in node.neighbors)):
                    node.neighbors.append(neighbor)

def update_neighbors(nodes, recompute, shape):
    directions = [(0, 1, 1), (1, 0, 1), (0, -1, 1), (-1, 0, 1), (0, 0, 1), (1, 1, 1), (-1, -1, 1), (1, -1, 1), (-1, 1, 1)]
    for (x, y, t) in recompute:
        for dx, dy, dt in directions:
            new_x, new_y, new_t = x + dx, y + dy, t + dt
            if 0 <= new_x < shape[0] and 0 <= new_y < shape[1] and 0 <= new_t < shape[2]:
                neighbor = nodes.get((new_x, new_y, new_t))
                if neighbor and (not (neighbor in nodes[(x,y,t)].neighbors)):
                    nodes[(x, y, t)].neighbors.append(neighbor)


def heuristic(node: Spot, goal: Spot) -> float:
    return ((node.x - goal.x)**2 + (node.y - goal.y)**2)**0.5

def astar_3d(start: Spot, goal: Spot) -> List[Tuple[int, int, int]]:
    startTime = datetime.now()
    open_list = []
    closed_set = set()

    start.g = 0
    start.h = heuristic(start, goal)
    start.f = start.g + start.h

    heapq.heappush(open_list, (start.f, start))

    while open_list:
        current = heapq.heappop(open_list)[1]

        if current.x == goal.x and current.y == goal.y:
            path = []
            while current:
                path.append((current.x, current.y, current.t))
                current = current.parent
            print(f"Path found in: {datetime.now() - startTime}s")
            return path[::-1]

        closed_set.add(current)

        for neighbor in current.neighbors:
            if neighbor in closed_set or neighbor.wall:
                continue

            if current.x != goal.x and 0 < current.y < 5:
                tentative_g = current.g + 1
            elif current.x == goal.x and current.y > 5:
                tentative_g = current.g + 1
            else:
                tentative_g = current.g + 5  

            if tentative_g < neighbor.g:
                neighbor.parent = current
                neighbor.g = tentative_g
                neighbor.h = heuristic(neighbor, goal)
                neighbor.f = neighbor.g + neighbor.h

                if neighbor not in [node for _, node in open_list]:
                    heapq.heappush(open_list, (neighbor.f, neighbor))

    print(goal.x, goal.y)
    print("Path Not Found")
    return []



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    shape = (20, 20, 100)
    nodes = create_grid_and_nodes(shape)
    compute_neighbors(nodes, shape)

    voxelarray = np.zeros(shape, dtype=bool)
    colors = np.empty(voxelarray.shape, dtype=object)

    for i in range(0, 19):
        for t in range(100):
            voxelarray[10, i, t] = True
            colors[10, i, t] = "red"
            nodes[(10, i, t)].wall = True

    # Use a manager to create a shared queue
    with mp.Manager() as manager:
        q = manager.Queue()
        processes = []
        
        # Start the process
        start = nodes[(0, 0, 0)]
        goal = nodes[(19, 19, 99)]
        p = mp.Process(target=astar_3d, args=(start, goal, q, -1))
        processes.append(p)

        start = nodes[(0, 19, 0)]
        goal = nodes[(19, 0, 99)]
        p = mp.Process(target=astar_3d, args=(start, goal, q, -1))
        processes.append(p)

        for p in processes:
            p.start()
        
        thing = 0
        while True:
            if not q.empty():
                object_instance_, path = q.get()
                print("Path received from queue:", path)
                if path:
                    for x, y, t in path:
                        voxelarray[x, y, t] = True
                        colors[x, y, t] = "blue"
                        nodes[(x, y, t)].wall = True
                thing += 1
            if thing == 2:
                # write reoccuring code
                break
            # else:
                # print("No path found or queue is empty")

    # Visualization
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(voxelarray, edgecolor='k', facecolors=colors)
    plt.show()
