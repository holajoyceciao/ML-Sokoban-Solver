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
    
    def __init__(self, warehouse):

        self.warehouse = warehouse

        self.x_max = max([x for x, y in self.warehouse.walls])
        self.x_min = min([x for x, y in self.warehouse.walls])

        self.y_max = max([y for x, y in self.warehouse.walls])
        self.y_min = min([y for x, y in self.warehouse.walls])

        self.worker = warehouse.worker
        self.boxes = list(warehouse.boxes)
        self.walls = set(warehouse.walls)
        self.targets = list(warehouse.targets)

        self.goal = warehouse.targets 
        self.initial =  (self.worker, self.boxes)
    
        self.initial = tuple(self.initial)# use tuple to make the state hashable
        self.goal = tuple(self.goal) # use tuple to make the state hashable
        #raise NotImplementedError()
        
  
    def move_result(self, direction, state):
        """
        executes a move action
        """
        worker = state.worker
        
        new_worker_position= tuple(a+b for a, b in zip(worker, direction))
        
        return tuple((new_worker_position, state.boxes))
    
    def push_result(self,direction,state):
        """
        A push move, needs a direction, the boxe moves and the player.
        """
        worker = state.worker
        boxes = state.boxes
        
        new_worker_position= self.move_result(direction, state)[0]
        
        new_box_position = tuple(a + b for a, b in zip(new_worker_position, direction))
    
        new_box_positions = boxes.remove(new_worker_position).append(new_box_position)
    
        return tuple((new_worker_position, new_box_positions))
    
    
    def actions(self, state):
        """
        Return the list of actions that can be executed in the given state.
        
        As specified in the header comment of this class, the attributes
        'self.allow_taboo_push' and 'self.macro' should be tested to determine
        what type of list of actions is to be returned.
        """
        worker = state.worker
        boxes = state.boxes
        targets = state.targets
        
        actions = []
        
        directions = [(-1,0), (1,0), (0,-1), (0,1)]
        for direction in directions:
            new_worker_pos = self.move_result(direction, state)[0]
            if new_worker_pos not in self.walls:
                if new_worker_pos in boxes:
                    if new_worker_pos not in targets:
                        new_box_pos = tuple(a + b for a, b in zip(new_worker_pos, direction))
                        print('new_box_pos', new_box_pos)
                        if new_box_pos not in self.walls:
                            actions.append((direction, 'p'))
                else:
                    actions.append((direction, 'm'))
            
        return actions
        
    def result(self,state, action):
        direction, actionType  = action
        if actionType=='p':
            worker, boxes = self.push_result(direction, state)
            return state.copy(worker, boxes)
        elif actionType=="m":
            worker, boxes = self.move_result(direction, state)
            return state.copy(worker, boxes)

    def goal_test(self,state):
        for boxe in state.boxes:
            if boxe not in state.targets:
                return False
        return True



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
    
    raise NotImplementedError()

def dfs(matrix, source, target, visited=None):
    '''
    A function that search a matrix of movements, return true if the target is accessible from the origin
    '''
    x, y = source

    print(f"Source: {source}, Target: {target}, Equal: {source == target}")
    if visited is None:
        visited = set()
    if source == target:
        return True
    visited.add(source)
    print(visited)
    
    rows = len(matrix)
    cols = len(matrix[0])
    
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    for direction in directions:
        nx, ny = x + direction[0], y + direction[1]
        if 0<=nx <rows and 0 <= ny < cols:
            neighbor = (nx, ny)
            if matrix[nx][ny] == 0 and neighbor not in visited:  
                if dfs(matrix, neighbor, target, visited): 
                    return True
    return False


def can_go_there(warehouse, dst):
    '''    
    Determine whether the worker can walk to the cell dst=(row,column) 
    without pushing any box.
    
    @param warehouse: a valid Warehouse object

    @return
      True if the worker can walk to cell dst=(row,column) without pushing any box
      False otherwise
    '''
    ##         "INSERT YOUR CODE HERE"
    y, x = warehouse.worker
    x1, y1 = dst
    target = tuple((x+x1, y+y1))
    x_max = max([x for x, y in warehouse.walls])
    x_min = min([x for x, y in warehouse.walls])

    y_max = max([y for x, y in warehouse.walls])
    y_min = min([y for x, y in warehouse.walls])

    dist = tuple((x_max-x_min, y_max-y_min))
     
    matrix = [[0 for _ in range(dist[0] + 1)] for _ in range(dist[1] + 1)]

    for wall_x, wall_y in warehouse.walls:
        matrix[wall_y - y_min][wall_x - x_min]  = 1 

    for box_x, box_y in warehouse.boxes:
        matrix[box_y - y_min][box_x - x_min] = 1 
    
    
    for row in matrix:
        print(row)
            
    return dfs(matrix, tuple((x,y)), target)

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

