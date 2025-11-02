# --- Python Backend using Flask ---
# To run this file:
# 1. Install Flask: pip install Flask
# 2. Run the script: python app.py
# 3. The server will start at http://127.0.0.1:5000/

import json
import time
import random
from flask import Flask, request, jsonify
from flask_cors import CORS

# Initialize Flask application
app = Flask(__name__)
# Enable CORS for frontend running on a different port (like React dev server)
CORS(app)

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
                # Store coordinates of the next cell and the wall cell in between
                neighbors.append((nx, ny, x + dx // 2, y + dy // 2)) 

        if neighbors:
            nx, ny, wx, wy = random.choice(neighbors)
            new_maze[wy][wx] = PATH # Carve the wall
            new_maze[ny][nx] = PATH # Move to the new cell
            stack.append((nx, ny))
        else:
            stack.pop()

    # Set Start and End points
    new_maze[1][1] = START
    new_maze[size - 2][size - 2] = END
    
    return new_maze

# --- MAZE SOLVING LOGIC ---

def solve_dfs(maze, size):
    """Depth-First Search implementation."""
    start_time = time.time()
    stack = [(1, 1, [(1, 1)])]
    visited_set = set(['1,1'])
    explored = 0
    visited_steps = []

    while stack:
        x, y, current_path = stack.pop()
        explored += 1
        visited_steps.append((x, y)) # Log exploration step

        if maze[y][x] == END:
            return explored, current_path, int((time.time() - start_time) * 1000), visited_steps

        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            key = f'{nx},{ny}'

            if 0 <= nx < size and 0 <= ny < size and \
               maze[ny][nx] != WALL and key not in visited_set:
                visited_set.add(key)
                stack.append((nx, ny, current_path + [(nx, ny)]))
    
    return explored, [], 0, visited_steps # Maze unsolvable


def solve_bfs(maze, size):
    """Breadth-First Search implementation."""
    start_time = time.time()
    queue = [(1, 1, [(1, 1)])]
    visited_set = set(['1,1'])
    explored = 0
    visited_steps = []

    while queue:
        x, y, current_path = queue.pop(0) # FIFO queue
        explored += 1
        visited_steps.append((x, y))

        if maze[y][x] == END:
            return explored, current_path, int((time.time() - start_time) * 1000), visited_steps

        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            key = f'{nx},{ny}'

            if 0 <= nx < size and 0 <= ny < size and \
               maze[ny][nx] != WALL and key not in visited_set:
                visited_set.add(key)
                queue.append((nx, ny, current_path + [(nx, ny)]))
    
    return explored, [], 0, visited_steps


def solve_astar(maze, size):
    """A* Search implementation."""
    start_time = time.time()
    end_x, end_y = size - 2, size - 2
    
    def heuristic(x, y):
        return abs(x - end_x) + abs(y - end_y)

    # open_set: [f_cost (g + h), x, y, g (cost from start), currentPath]
    open_set = [(heuristic(1, 1), 1, 1, 0, [(1, 1)])]
    g_costs = {'1,1': 0}
    visited_steps = []
    explored = 0

    while open_set:
        # Sort by f_cost (f_cost is the first element for easy sorting)
        open_set.sort(key=lambda item: item[0])
        f, x, y, g, current_path = open_set.pop(0) 
        
        key = f'{x},{y}'
        
        if maze[y][x] == END:
            return explored, current_path, int((time.time() - start_time) * 1000), visited_steps
            
        explored += 1
        visited_steps.append((x, y))

        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            n_key = f'{nx},{ny}'

            if 0 <= nx < size and 0 <= ny < size and maze[ny][nx] != WALL:
                new_g = g + 1
                
                # If path to neighbor is shorter or unvisited
                if n_key not in g_costs or new_g < g_costs[n_key]:
                    g_costs[n_key] = new_g
                    new_h = heuristic(nx, ny)
                    new_f = new_g + new_h
                    
                    # Remove old entry from open_set if it exists
                    open_set = [item for item in open_set if item[1] != nx or item[2] != ny]
                    
                    # Add new, improved entry
                    open_set.append((new_f, nx, ny, new_g, current_path + [(nx, ny)]))

    return explored, [], 0, visited_steps

# --- FLASK ROUTES (API Endpoints) ---

@app.route('/api/generate_maze', methods=['GET'])
def generate_maze_api():
    """Endpoint to generate and return a new maze."""
    try:
        # Get size parameter from URL query string, default to 15
        size = int(request.args.get('size', 15))
        # Ensure size is an odd number and within bounds
        size = max(11, min(31, size))
        if size % 2 == 0:
            size += 1

        new_maze = generate_maze_logic(size)
        
        # Return the maze data as JSON
        return jsonify({
            'maze': new_maze,
            'size': size
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/solve_maze', methods=['POST'])
def solve_maze_api():
    """Endpoint to solve the maze using a specified algorithm."""
    try:
        data = request.get_json()
        maze = data.get('maze')
        size = data.get('size')
        algorithm = data.get('algorithm')

        if not maze or not size or not algorithm:
            return jsonify({'error': 'Missing maze data, size, or algorithm.'}), 400

        # Choose the solving function based on the algorithm requested
        if algorithm == 'dfs':
            explored, path, time_ms, visited_steps = solve_dfs(maze, size)
        elif algorithm == 'bfs':
            explored, path, time_ms, visited_steps = solve_bfs(maze, size)
        elif algorithm == 'astar':
            explored, path, time_ms, visited_steps = solve_astar(maze, size)
        else:
            return jsonify({'error': 'Invalid algorithm specified.'}), 400

        # Return the solution data
        return jsonify({
            'explored': explored,
            'path': path,
            'time': time_ms,
            'visited_steps': visited_steps # Return the steps for frontend animation
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run the Flask server. debug=True allows for automatic restarts on code changes.
    app.run(debug=True)

