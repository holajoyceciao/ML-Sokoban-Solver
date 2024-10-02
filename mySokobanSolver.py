'''
IFN680 Sokoban Assignment

The functions and classes defined in this module will be called by a marker script. 
You should complete the functions and classes according to their specified interfaces.

You are not allowed to change the defined interfaces.
That is, changing the formal parameters of a function will break the 
interface and triggers to a fail for the test of your code.
'''


import search
import sokoban
import time 

def my_team():
    '''
    Return the list of the team members of this assignment submission as a list
    of triplet of the form (student_number, first_name, last_name)
    e.g.  [ (1234567, 'Ada', 'Lovelace'), (1234568, 'Grace', 'Hopper'), (1234569, 'Eva', 'Tardos') ]
    '''

    raise NotImplementedError()
 

def taboo_cells(warehouse):
    '''  
    Identify the taboo cells of a warehouse. A cell inside a warehouse is 
    called 'taboo' if whenever a box get pushed on such a cell then the puzzle 
    becomes unsolvable.  
    When determining the taboo cells, you must ignore all the existing boxes, 
    simply consider the walls and the target cells.  
    Use only the following two rules to determine the taboo cells;
     Rule 1: if a cell is a corner inside the warehouse and not a target, 
             then it is a taboo cell.
     Rule 2: all the cells between two corners inside the warehouse along a 
             wall are taboo if none of these cells is a target.
    
    @param warehouse: a Warehouse object

    @return
       A string representing the puzzle with only the wall cells marked with 
       an '#' and the taboo cells marked with an 'X'.  
       The returned string should NOT have marks for the worker, the targets,
       and the boxes.  
    '''
    ##         "INSERT YOUR CODE HERE"  

    # retrieving warehouse
    warehouse = str(warehouse).split('\n')
    warehouse = [list(row) for row in warehouse]
    rows, cols = len(warehouse), len(warehouse[0])

    # mark out of boundary position as NA
    ## left and right detection
    for r in range(rows):
        # from left
        l_ptr = 0
        while l_ptr < cols and warehouse[r][l_ptr] != '#':
            if warehouse[r][l_ptr] == ' ':
                warehouse[r][l_ptr] = 'NA'
            l_ptr += 1

        # from right
        r_ptr = cols - 1
        while r_ptr >= 0 and warehouse[r][r_ptr] != '#':
            if warehouse[r][r_ptr] == ' ':
                warehouse[r][r_ptr] = 'NA'
            r_ptr -= 1

    ## top and bottom detection
    for c in range(cols):
        # from top
        t_ptr = 0
        while t_ptr < rows and warehouse[t_ptr][c] != '#':
            if warehouse[t_ptr][c] == ' ':
                warehouse[t_ptr][c] = 'NA'
            t_ptr += 1

        # from bottom
        b_ptr = rows - 1
        while b_ptr >= 0 and warehouse[b_ptr][c] != '#':
            if warehouse[b_ptr][c] == ' ':
                warehouse[b_ptr][c] = 'NA'
            b_ptr -= 1

    # rule 1: mark taboo cell at corners
    for r in range(rows):
        for c in range(cols):
            if warehouse[r][c] not in ('.', 'NA', '#'):
                # top-left
                if r > 0 and c > 0 and warehouse[r - 1][c] == '#' and warehouse[r][c - 1] == '#':
                    warehouse[r][c] = 'X'
                # top-right
                if r > 0 and c + 1 < cols and warehouse[r - 1][c] == '#' and warehouse[r][c + 1] == '#':
                    warehouse[r][c] = 'X'
                # bottom-left
                if r + 1 < rows and c > 0 and warehouse[r + 1][c] == '#' and warehouse[r][c - 1] == '#':
                    warehouse[r][c] = 'X'
                # bottom-right
                if r + 1 < rows and c + 1 < cols and warehouse[r + 1][c] == '#' and warehouse[r][c + 1] == '#':
                    warehouse[r][c] = 'X'

    # Rule 2: mark taboo cells between two corners along walls
    ## horizontal detection (left to right)
    for r in range(rows):
        left_corner = -1
        for c in range(cols):
            if warehouse[r][c] == 'X':  # Found a corner
                if left_corner == -1:
                    left_corner = c
                else:
                    # check if all cells between the corners are empty spaces and not targets
                    if all(warehouse[r][i] == ' ' for i in range(left_corner + 1, c)) and all(warehouse[r][i] not in '.' for i in range(left_corner + 1, c)):
                        for i in range(left_corner + 1, c):
                            warehouse[r][i] = 'X'
                    left_corner = c

    # vertical detection (top to bottom)
    for c in range(cols):
        top_corner = -1
        for r in range(rows):
            if warehouse[r][c] == 'X':  # Found a corner
                if top_corner == -1:
                    top_corner = r
                else:
                    # check if all cells between the corners are empty spaces and not targets
                    if all(warehouse[i][c] == ' ' for i in range(top_corner + 1, r)) and all(warehouse[i][c] not in '.' for i in range(top_corner + 1, r)):
                        for i in range(top_corner + 1, r):
                            warehouse[i][c] = 'X'
                    top_corner = r

    # remove other symbols and swap 'NA' back to empty spaces
    for r in range(rows):
        for c in range(cols):
            if warehouse[r][c] in '.*@$':
                warehouse[r][c] = ' '
            if warehouse[r][c] == 'NA':
                warehouse[r][c] = ' '

    return '\n'.join([''.join(row) for row in warehouse])  
 


class SokobanPuzzle(search.Problem):
    '''
    An instance of the class 'SokobanPuzzle' represents a Sokoban puzzle.
    An instance contains information about the walls, the targets, the boxes
    and the worker.

    Your implementation should be fully compatible with the search functions of 
    the provided module 'search.py'. It uses search.Problem as a sub-class. 
    That means, it should have a:
    - self.actions() function
    - self.result() function
    - self.goal_test() function
    See the Problem class in search.py for more details on these functions.
    
    Each instance should have at least the following attributes:
    - self.allow_taboo_push
    - self.macro
    
    When self.allow_taboo_push is set to True, the 'actions' function should 
    return all possible legal moves including those that move a box on a taboo 
    cell. If self.allow_taboo_push is set to False, those moves should not be
    included in the returned list of actions.
    
    If self.macro is set True, the 'actions' function should return 
    macro actions. If self.macro is set False, the 'actions' function should 
    return elementary actions.
    
    
    '''
    
    def __init__(self, warehouse, goal=None, allow_taboo_push=False, macro=False, allow_push=True):
        # directions to move in (x, y) -> NOTE: origin at TOP-LEFT
        self.directions = { 'Left': (-1, 0), 'Right': (1, 0), 'Up': (0, -1), 'Down': (0, 1) }

        # retrieve key state of the warehouse
        self.walls = set()
        self.targets = set()
        self.boxes = set()
        self.worker = tuple()
        self.define_game_state(warehouse)
        self.allow_push = allow_push
        #need the warehouse for the can_go_there, at the moment
        self.warehouse = warehouse
        
        
        # get maximum game boundaries
        ## compute number of rows and columns from the layout by joining sets and get maximum
        all_positions = self.walls | self.boxes | self.targets | {self.worker}
        self.nys = max(y for _, y in all_positions) + 1
        self.nxs = max(x for x, _ in all_positions) + 1

        # get taboo cell from the function to know if a position is 'X'
        self.taboo_layout = taboo_cells(warehouse).splitlines()
        #need taboo cell for the lookup in solve_macro
        self.taboo_cells = set(sokoban.find_2D_iterator(self.taboo_layout , "X"))
          


        # save the state for type of movement
        self.allow_taboo_push = allow_taboo_push
        #if true, return push actions
        self.macro = macro

        # save the original state for the warehouse
        super().__init__((self.worker, frozenset(self.boxes)), goal)
        self.goal = goal
        
    def define_game_state(self, warehouse):
        # retrieve states from the warehouse
        self.walls = set(warehouse.walls)
        self.targets = set(warehouse.targets)
        self.boxes = set(warehouse.boxes)
        self.worker = warehouse.worker
    
    def actions(self, state):
        """
        Return the list of actions that can be executed in the given state.
        """
        # get CURRENT worker position from the state
        ## worker = (x, y); boxes (hashset) = {(x, y), (x, y) ...} -> They are the LATEST state not initial
        worker, boxes = state
        x, y = worker
        # legal action for the worker
        possible_actions = []
        if self. macro:
    ##################   MACRO ACTION ################
            for box in boxes:
                #check all tiles adjacent to the box
                for key, direction in self.directions.items():
                    #position of the worker next to the box
                    worker_pos = tuple(a-b for a,b in zip(box, direction))
                    #check it the move is valide, saves time
                    new_box_pos = tuple(a+b for a,b in zip(box, direction))
                    #cannot push a box on a wall or an other box
                    if new_box_pos in self.walls.union(boxes):
                        continue
                    #if allow_taboo push is false
                    if new_box_pos in self.taboo_cells:
                        if  not self.allow_taboo_push:
                            continue
                    
                    #create a new warehouse with the current position of the worker before any move
                    new_warehouse = self.warehouse.copy(worker=worker, boxes=boxes)
                    #check if the worker can go there, targer is y, x so needs to be inverted
                    if can_go_there(new_warehouse, tuple((worker_pos[1], worker_pos[0]))):
                            possible_actions.append(tuple(((box[1], box[0]), key)))

###################   ELEMENT ACTIONS #################
        else:
            # iterate over 4 directions { 'Left': (-1, 0), 'Right': (1, 0), 'Up': (0, -1), 'Down': (0, 1) }
            for move, coordinates in self.directions.items():
                dx, dy = coordinates
            #check if wnant macro
                # next potential position by adding the updating (x, y) position
                ## eg. current = (4, 2) ; go up one grid = (4, 1)
                nx, ny = x + dx, y + dy
                # check if the next position is in bound
                if 0 <= nx < self.nxs and 0 <= ny < self.nys:
                      # if the next position is a wall (efficient lookup from hashset)
                    if (nx, ny) in self.walls:
                        continue
                          # if the next position is a box (efficient lookup from hashset)
                    if (nx, ny) in boxes:
                        if not self.allow_push:
                            continue
                        # want to push box, so need to identify the 'next next position'
                        nnx, nny = nx + dx, ny + dy

                        # if the 'next next position' is in game boundary
                        if 0 <= nnx < self.nxs and 0 <= nny < self.nys:
                            # if the 'next next position' is a vacant space or target cell
                            if (nnx, nny) in boxes or  (nnx, nny) in self.walls:
                                continue
                                # check if that vacant cell might be a taboo from taboo_layout
                                ## if 'taboo_push' is False, cannot push to taboo cell
                            if self.allow_taboo_push and self.taboo_layout[nny][nnx] == 'X':
                                continue  # skip and continue directly from next iteration

                            possible_actions.append(move)
    
                    # if the next position is a vacant space or the target spot -> move worker
                    possible_actions.append(move)  
        return possible_actions

    def result(self, state, action): # now worker is ABOUT to perform an action
        # get current worker position from the state
        worker, boxes = state
        x, y = worker
        
        if self.macro:
######### MACRO RESULT ##########################################################
            box, direction = action
            #the box location format is y, x
            box = (box[1], box[0])
            #The position the box will be after the push
            new_box = tuple(a+b for a , b in zip(self.directions[direction], box))
            boxes_list = list(boxes)
            #remove previous box position
            boxes_list.remove(box)
            #add next box position
            boxes_list.append(new_box)
            #worker goes where the box previously was
            new_worker = box
     
            return (new_worker, tuple(boxes_list))
######## MICRO RESULT ###########################################################
        
        else:
            # directions = { 'Left': (-1, 0), 'Right': (1, 0), 'Up': (0, -1), 'Down': (0, 1) }
            dx, dy = self.directions[action]
        
            # next position for the worker (x, y) -> (nx, ny)
            nx, ny = x + dx, y + dy 
    
            # if the new location is a box, worker will push the box
            if (nx, ny) in boxes:
                # find the 'next next position'
                nnx, nny = nx + dx, ny + dy
    
                # push the box to the new position
                new_boxes = set(boxes)
                new_boxes.remove((nx, ny))
                new_boxes.add((nnx, nny))
    
                # return new state with updated worker and boxes positions
                return ((nx, ny), frozenset(new_boxes))
    
            # if worker is moving to a vacant space
            else:
                # return new state with updated worker position
                return ((nx, ny), boxes)
        
            # if action is not legal, should not happen
            raise Exception('Invalid action')

    def goal_test(self, state):
        # get current boxes position
        worker, boxes = state
        if self.goal is not None:
            return worker == self.goal
        # check if all boxes are in target positions
        else:
            return all(box in self.targets for box in boxes)

    def path_cost(self, c, state1, action, state2):
        # record length of path
        return c + 1
        
    def h_1(self, state):
        return self.manhattan_distance(state)
    def h_2(self, state):
        return self.euclidean_distance(state)
        
    def euclidean_distance(self, node):
     # simple heuristic used for can_get_there, is the distance to the goal
        worker, _ = node.state
        player_x, player_y = worker
        return np.sqrt((player_x - self.goal[0])**2+ (player_y - self.goal[1])**2)   

    def manhattan_distance(self, state):
    # simple heuristic used for the macro_search, uses the min distance betwenn each box and target as a cost
        worker, boxes = state.state
        player_x, player_y = worker
        
        boxes_cost = 0
        player_cost = 0
        
        for box_x, box_y in boxes:
        # Cost of moving each box to the nearest goal
            min_box_goal_distance = min(
                abs(box_x - goal_x) + abs(box_y - goal_y) for goal_x, goal_y in self.targets
            )
            boxes_cost += min_box_goal_distance
        
        #if boxes:
        #cost of moving to the nearest box, not implemented beacuse would require the path to reach a point
         #   player_cost = min(
         #       abs(box_x - player_x) + abs(box_y - player_y) for box_x, box_y in boxes
         #   )
        
        return boxes_cost #+ player_cost
    
    def value(self, state):
        """For optimization problems, each state has a value.  Hill-climbing
        and related algorithms try to maximize this value."""
        raise NotImplementedError
        
    def print_solution(self, goal_node):

        path = goal_node.path()

        print( f"Solution takes {len(path)-1} steps from the initial state to the goal state\n")
        print( "Below is the sequence of moves\n")
        moves = []
        for node in path:
            if node.action:
                moves.append(node.action)
        return(moves)
      
        

def check_action_seq(warehouse, action_seq):
    '''
    
    Determine if the sequence of actions listed in 'action_seq' is legal or not.
    
    Important notes:
      - a legal sequence of actions does not necessarily solve the puzzle.
      - an action is legal even if it pushes a box onto a taboo cell.
        
    @param warehouse: a valid Warehouse object

    @param action_seq: a sequence of legal actions.
           For example, ['Left', 'Down', Down','Right', 'Up', 'Down']
           
    @return
        The string 'Failure', if one of the action was not successul.
           For example, if the agent tries to push two boxes at the same time,
                        or push one box into a wall, or walk into a wall.
        Otherwise, if all actions were successful, return                 
               A string representing the state of the puzzle after applying
               the sequence of actions.  This must be the same string as the
               string returned by the method  Warehouse.__str__()
    '''
    
    ##         "INSERT YOUR CODE HERE"
    
    raise NotImplementedError()



def solve_sokoban_elem(warehouse):
    '''    
    This function should solve using elementary actions 
    the puzzle defined in a file.
    
    @param warehouse: a valid Warehouse object

    @return
        If puzzle cannot be solved return the string 'Impossible'
        If a solution was found, return a list of elementary actions that solves
            the given puzzle coded with 'Left', 'Right', 'Up', 'Down'
            For example, ['Left', 'Down', Down','Right', 'Up', 'Down']
            If the puzzle is already in a goal state, simply return []
    '''
    
    ##         "INSERT YOUR CODE HERE"

    
    solver = SokobanPuzzle(warehouse)

    sol_ts = search.breadth_first_graph_search(solver)
    
    return sol_ts


import numpy as np

            
def can_go_there(warehouse, dst):
    '''    
    Determine whether the worker can walk to the cell dst=(row,column) 
    switch because the row is y and column i s x
    without pushing any box.

    overRide the Sokoban Solver, changes the goal and the goal_test
    
    @param warehouse: a valid Warehouse object

    @return
      True if the worker can walk to cell dst=(row,column) without pushing any box
      False otherwise
    '''
   
    ##         "INSERT YOUR CODE HERE"
    goal = (dst[1], dst[0])
    goThereSolver = SokobanPuzzle(warehouse, goal=goal,allow_push=False)      
    sol_ts = search.astar_graph_search(goThereSolver, goThereSolver.h_2)
    return sol_ts is not None

# Define a timeout exception
class TimeoutException(Exception):
    pass

# Define the signal handler
def timeout_handler(signum, frame):
    raise TimeoutException("Computation took too long!")

import signal

def solve_sokoban_macro(warehouse):
    '''    
    Solve using macro actions the puzzle defined in the warehouse passed as
    a parameter. A sequence of macro actions should be 
    represented by a list M of the form
            [ ((r1,c1), a1), ((r2,c2), a2), ..., ((rn,cn), an) ]
    For example M = [ ((3,4),'Left') , ((5,2),'Up'), ((12,4),'Down') ] 
    means that the worker first goes the box at row 3 and column 4 and pushes it left,
    then goes to the box at row 5 and column 2 and pushes it up, and finally
    goes the box at row 12 and column 4 and pushes it down.
    
    @param warehouse: a valid Warehouse object

    @return
        If puzzle cannot be solved return the string 'Impossible'
        Otherwise return M a sequence of macro actions that solves the puzzle.
        If the puzzle is already in a goal state, simply return []
    '''
     
    
    ##         "INSERT YOUR CODE HERE"
    #find reachable box       
    
    # Set the timeout handler for a 3-minute limit (180 seconds)
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(180)  # Set the alarm for 3 minutes
    try:
        # Find reachable boxes and solve using macro actions
        macroSolver = SokobanPuzzle(warehouse, allow_taboo_push=False, macro=True)        
        # Perform A* search
        sol_ts = search.astar_graph_search(macroSolver, macroSolver.h_1)
        
        # Check if a solution was found
        if sol_ts is not None:
            return macroSolver.print_solution(sol_ts)
        else:
            return "Impossible"

    except TimeoutException:
        return "Timeout"
    
    finally:
        # Cancel the alarm after completion or timeout
        signal.alarm(0)
    

