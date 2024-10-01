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
    # retrieving warehouse
    warehouse = str(warehouse).split('\n')
    warehouse = [list(row) for row in warehouse]
    rows, cols = len(warehouse), len(warehouse[0])
    
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

    ##         "INSERT YOUR CODE HERE"    
 


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
    
    def __init__(self, warehouse, goal=None, can_push=True, use_macro=False):
        
        self.warehouse = warehouse
        self.can_push = can_push
        self.x_max = max([x for x, y in self.warehouse.walls])
        self.x_min = min([x for x, y in self.warehouse.walls])

        self.y_max = max([y for x, y in self.warehouse.walls])
        self.y_min = min([y for x, y in self.warehouse.walls])

        self.worker = warehouse.worker
        self.boxes = warehouse.boxes
        self.walls = set(warehouse.walls)
        self.targets = set(warehouse.targets)
        self.taboo_cells = set()

        self.action_dict = {
            'Up' : tuple((0,-1)),
            'Down' : tuple((0,1)),
            'Right' : tuple((1,0)),
            'Left' : tuple((-1,0))
        }

        self.goal = goal
        self.initial =  tuple((self.worker, tuple(self.boxes)))# use tuple to make the state hashable
        self.targets = tuple(self.targets) # use tuple to make the state hashable

    def find_taboo(self, taboo_warehouse):
        if (len(self.taboo_cells) == 0):
            lines = taboo_warehouse.split("\n")
            self.taboo_cells =set(sokoban.find_2D_iterator(lines, "X"))
            
    def h(self, state):
        return self.manhattan_distance(state)
        
    def move_result(self, direction, state):
        """
        executes a move action
        """
        worker = state[0]
        
        new_worker_position= tuple(a+b for a, b in zip(worker, direction))
        
        return tuple((new_worker_position, state[1]))
    
    def push_result(self,direction,state):
        """
        A push move, needs a direction, the boxe moves and then the player.
        """
        worker, boxes = state
        new_worker_position= self.move_result(direction, state)[0]
        
        new_box_position = tuple(a + b for a, b in zip(new_worker_position, direction))
    
        boxes.remove(new_worker_position)
        boxes.append(new_box_position)
        
        return tuple((new_worker_position, boxes))
                                
    
    def actions(self, state):
        """
        Return the list of actions that can be executed in the given state.
        
        As specified in the header comment of this class, the attributes
        'self.allow_taboo_push' and 'self.macro' should be tested to determine
        what type of list of actions is to be returned.
        """
        worker, boxes = state
        actions = []

        #(row, column)
        # Directions: (-1, 0) = left, (1, 0) = right, (0, -1) = down, (0, 1) = up
        directions = [(-1,0), (1,0), (0,-1), (0,1)]
        for direction in directions:
            new_worker_pos = self.move_result(direction, state)[0]
            # If the new worker position is in a wall, skip to the next direction
            if new_worker_pos in self.walls:
                continue
            # If the new worker position is in a box
            if new_worker_pos in boxes:
                # Check if pushing the box is allowed
                if not self.can_push:
                    continue
                
                # Determine the new box position after pushing
                new_box_pos = tuple(a + b for a, b in zip(new_worker_pos, direction))
                
                # If the new box position is in a wall or another box, or a taboo cell, skip this direction
                if new_box_pos in self.walls or new_box_pos in boxes or new_box_pos in taboo_cells:
                    continue
                
                # The move is legal, append it as a push action
                actions.append((direction, 'p'))
            else:
                # If the new position is not a wall or a box, append it as a move action
                actions.append((direction, 'm'))
        return actions
        
    def result(self,state, action):
        direction, actionType  = action
        if actionType=='p':
            return self.push_result(direction, state)
        elif actionType=="m":
            return self.move_result(direction, state)

    def goal_test(self, state):
        worker, boxes = state
        if self.goal is not None:
            return worker == self.goal
        else:
            return set(boxes) == set(self.targets)
            
    def path_cost(self, c, state1, action, state2):
        return c+1

    def manhattan_distance(self, state):
        
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
        
        if boxes:
        #same, should be changes to box not on a target
            player_cost = min(
                abs(box_x - player_x) + abs(box_y - player_y) for box_x, box_y in boxes
            )
        
        return boxes_cost + player_cost
    
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


class canGoPuzzle(SokobanPuzzle):
   
    def euclidean_distance(self, node):
        worker = node.state[0]
        player_x, player_y = worker
        return np.sqrt((player_x - self.goal[0])**2+ (player_y - self.goal[1])**2)

    def h(self,state):
        return self.manhattan_distance(state)
        
    
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
    
    goThereSolver = canGoPuzzle(warehouse, goal=goal, can_push=False)
            
    sol_ts = search.breadth_first_graph_search(goThereSolver)
    
    return sol_ts is not None


class solveSokoMacro(SokobanPuzzle):

     
        def result(self, state, action):
            worker, boxes = state
            box, direction = action
            box = (box[1], box[0])
            
        
            new_box = tuple(a+b for a , b in zip(self.action_dict[direction], box))
            boxes_list = list(boxes)
            boxes_list.remove(box)
            boxes_list.append(new_box)
            new_worker = box
            return tuple((new_worker, tuple(boxes_list)))
        
        def actions(self, state):
            worker, boxes =state
            actions = []
            
            for box in boxes:
                #check all tiles adjacent to the box
                for key, direction in self.action_dict.items():
                    #position of the worker next to the box
                    worker_pos = tuple(a-b for a,b in zip(box, direction))
                    #create a new warehouse with the current position of the worker before any move
                    new_warehouse = self.warehouse.copy(worker=worker, boxes=boxes)
                    #check if the worker can go there, targer is y, x so needs to be inverted
                    if can_go_there(new_warehouse, tuple((worker_pos[1], worker_pos[0]))):
                        
                        new_box_pos = tuple(a+b for a,b in zip(box, direction))
        
                        if new_box_pos not in self.walls and new_box_pos not in boxes and new_box_pos not in self.taboo_cells:
                            #again, the output needs to be y, x
                            actions.append(tuple(((box[1], box[0]), key)))
            return actions
            
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
    
    taboo_warehouse = taboo_cells(warehouse)
    macroSolver = solveSokoMacro(warehouse)
    #add the taboo cells
    macroSolver.find_taboo(taboo_warehouse)
    
    sol_ts = search.astar_graph_search(macroSolver,macroSolver.h)
    if sol_ts is not None:
        return macroSolver.print_solution(sol_ts)
    else:
        return "Impossible"
    
    

