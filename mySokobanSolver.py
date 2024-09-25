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
    raise NotImplementedError()


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
    
    def __init__(self, warehouse, goal=None, can_push=True):
        
        self.warehouse = warehouse
        self.can_push = can_push
        self.x_max = max([x for x, y in self.warehouse.walls])
        self.x_min = min([x for x, y in self.warehouse.walls])

        self.y_max = max([y for x, y in self.warehouse.walls])
        self.y_min = min([y for x, y in self.warehouse.walls])

        self.worker = warehouse.worker
        self.boxes = set(warehouse.boxes)
        self.walls = set(warehouse.walls)
        self.targets = set(warehouse.targets)

        self.goal = goal
        self.initial =  tuple((self.worker)), tuple((self.boxes))# use tuple to make the state hashable
        self.targets = tuple(self.targets) # use tuple to make the state hashable
        
    def move_result(self, direction, state):
        """
        executes a move action
        """
        worker = state[0]
        
        new_worker_position= tuple(a+b for a, b in zip(worker, direction))
        
        return tuple((new_worker_position, state[1]))
    
    def push_result(self,direction,state):
        """
        A push move, needs a direction, the boxe moves and the player.
        """
        worker, boxes = state
        new_worker_position= self.move_result(direction, state)[0]
        
        new_box_position = tuple(a + b for a, b in zip(new_worker_position, direction))
    
        boxes.remove(new_worker_position)
        boxes.append(new_box_position)
        
    
        return tuple((new_worker_position, tuple(boxes)))
    
    
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
                
                # If the new box position is in a wall or another box, skip this direction
                if new_box_pos in self.walls or new_box_pos in boxes:
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
            return set(boxes) == self.targets

    def print_solution(self, goal_node):

        path = goal_node.path()

        print( f"Solution takes {len(path)-1} steps from the initial state to the goal state\n")
        print( "Below is the sequence of moves\n")
        moves = []
        for node in path:
            if node.action:
                moves += [f"{node.action}, "]
        print(moves)


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
    goal = (dst[1], dst[0])
    ##         "INSERT YOUR CODE HERE"
    goThereSolver = SokobanPuzzle(warehouse, goal=goal, can_push=False)
            
    sol_ts = search.breadth_first_graph_search(goThereSolver)
    return sol_ts is not None



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

    
    raise NotImplementedError()

