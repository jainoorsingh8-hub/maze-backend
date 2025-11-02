# --- Python Backend using Flask ---
# To run locally:
# 1. pip install Flask flask-cors
# 2. python app.py
# The server runs on http://127.0.0.1:5000/

import json
import time
import random
from flask import Flask, request, jsonify
from flask_cors import CORS

# Initialize Flask application
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend connections

# Constants
WALL = 1
PATH = 0
START = 2
END = 3

# --- MAZE GENERATION LOGIC (Recursive Backtracking) ---
def generate_maze_logic(size):
    """Generates a maze using the Recursive Backtracking algorithm."""
    new_maze = [[WALL] * size for _ in range(size)]
    stack = []
    start_x, start_y = 1, 1
    
    new_maze[start_y][start_x] = PATH
    stack.append((start_x, start_y))
    directions = [(0, -2), (2, 0), (0, 2), (-2, 0)]

    while stack:
        x, y = stack[-1]
        neighbors = []

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 < nx < size - 1 and 0 < ny < size - 1 and new_maze[ny][nx] == WALL:
                neighbors.append((nx, ny, x + dx // 2, y + dy // 2))

        if neighbors:
            nx, ny, wx, wy = random.choice(neighbors)
            new_maze[wy][wx] = PATH
            new_maze[ny][nx] = PATH
            stack.append((nx, ny))
        else:
            stack.pop()

    new_maze[1][1] = START
    new_maze[size - 2][size - 2] = END
    return new_maze


# --- SOLVERS ---

def solve_dfs(maze, size):
    start_time = time.time()
    stack = [(1, 1, [(1, 1)])]
    visited = set(['1,1'])
    explored = 0
    visited_steps = []

    while stack:
        x, y, path = stack.pop()
        explored += 1
        visited_steps.append((x, y))

        if maze[y][x] == END:
            return explored, path, int((time.time() - start_time) * 1000), visited_steps

        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            key = f"{nx},{ny}"
            if 0 <= nx < size and 0 <= ny < size and maze[ny][nx] != WALL and key not in visited:
                visited.add(key)
                stack.append((nx, ny, path + [(nx, ny)]))

    return explored, [], 0, visited_steps


def solve_bfs(maze, size):
    start_time = time.time()
    queue = [(1, 1, [(1, 1)])]
    visited = set(['1,1'])
    explored = 0
    visited_steps = []

    while queue:
        x, y, path = queue.pop(0)
        explored += 1
        visited_steps.append((x, y))

        if maze[y][x] == END:
            return explored, path, int((time.time() - start_time) * 1000), visited_steps

        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            key = f"{nx},{ny}"
            if 0 <= nx < size and 0 <= ny < size and maze[ny][nx] != WALL and key not in visited:
                visited.add(key)
                queue.append((nx, ny, path + [(nx, ny)]))

    return explored, [], 0, visited_steps


def solve_astar(maze, size):
    start_time = time.time()
    end_x, end_y = size - 2, size - 2

    def heuristic(x, y):
        return abs(x - end_x) + abs(y - end_y)

    open_set = [(heuristic(1, 1), 1, 1, 0, [(1, 1)])]
    g_costs = {'1,1': 0}
    visited_steps = []
    explored = 0

    while open_set:
        open_set.sort(key=lambda item: item[0])
        f, x, y, g, path = open_set.pop(0)
        explored += 1
        visited_steps.append((x, y))

        if maze[y][x] == END:
            return explored, path, int((time.time() - start_time) * 1000), visited_steps

        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            key = f"{nx},{ny}"

            if 0 <= nx < size and 0 <= ny < size and maze[ny][nx] != WALL:
                new_g = g + 1
                if key not in g_costs or new_g < g_costs[key]:
                    g_costs[key] = new_g
                    new_f = new_g + heuristic(nx, ny)
                    open_set = [item for item in open_set if item[1] != nx or item[2] != ny]
                    open_set.append((new_f, nx, ny, new_g, path + [(nx, ny)]))

    return explored, [], 0, visited_steps


# --- API ROUTES ---

@app.route('/')
def home():
    return "<h1>Maze Solver API</h1><p>Endpoints:<br>/api/generate_maze<br>/api/solve_maze</p>"


@app.route('/api/generate_maze', methods=['GET'])
def generate_maze_api():
    try:
        size = int(request.args.get('size', 15))
        size = max(11, min(31, size))
        if size % 2 == 0:
            size += 1
        maze = generate_maze_logic(size)
        return jsonify({'maze': maze, 'size': size})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/solve_maze', methods=['POST'])
def solve_maze_api():
    try:
        data = request.get_json()
        maze = data.get('maze')
        size = data.get('size')
        algorithm = data.get('algorithm')

        if not maze or not size or not algorithm:
            return jsonify({'error': 'Missing maze data, size, or algorithm.'}), 400

        if algorithm == 'dfs':
            explored, path, time_ms, visited_steps = solve_dfs(maze, size)
        elif algorithm == 'bfs':
            explored, path, time_ms, visited_steps = solve_bfs(maze, size)
        elif algorithm == 'astar':
            explored, path, time_ms, visited_steps = solve_astar(maze, size)
        else:
            return jsonify({'error': 'Invalid algorithm specified.'}), 400

        return jsonify({
            'explored': explored,
            'path': path,
            'time': time_ms,
            'visited_steps': visited_steps
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
