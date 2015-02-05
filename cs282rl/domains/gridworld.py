import numpy as np
from ..utils import check_random_state


# Maze state is represented as a 2-element NumPy array: (Y, X). Increasing Y is South.

# Possible actions, expressed as (delta-y, delta-x).
maze_actions = {
    'N': np.array([-1, 0]),
    'S': np.array([1, 0]),
    'E': np.array([0, 1]),
    'W': np.array([0, -1]),
}

def parse_topology(topology):
    return np.array([list(row) for row in topology])


class Maze(object):
    """
    Simple wrapper around a NumPy 2D array to handle indexing and staying in bounds.
    """
    def __init__(self, topology):
        self.topology = parse_topology(topology)
        self.shape = self.topology.shape

    def in_bounds(self, position):
        position = np.asarray(position)
        return np.all(position >= 0) and np.all(position < self.shape)

    def __getitem__(self, position):
        if not self.in_bounds(position):
            raise IndexError("Position out of bounds: {}".format(position))
        return self.topology[tuple(position)]

    def positions_containing(self, x):
        return self._tuplify(self.topology == x)

    def positions_not_containing(self, x):
        return self._tuplify(self.topology != x)

    def _tuplify(self, arr):
        return [tuple(position) for position in np.transpose(np.nonzero(arr))]

    def __str__(self):
        return '\n'.join(''.join(row) for row in self.topology.tolist())

    def __repr__(self):
        return 'Maze({})'.format(repr(self.topology.tolist()))


def move_avoiding_walls(maze, position, action):
    """
    Return the new position after moving, and the event that happened ('hit-wall' or 'moved')
    """
    # Compute new position
    new_position = position + action

    # Compute collisions with walls, including implicit walls at the ends of the world.
    if not maze.in_bounds(new_position) or maze[new_position] == '#':
        return position, 'hit-wall'

    return new_position, 'moved'



class GridWorld(object):
    """
    A simple task in a maze: get to the goal.

    Parameters
    ----------

    maze : list of strings or lists
        maze topology (see below)

    rewards: dict of string to number. default: {'*': 10}.
        Rewards obtained by being in a maze grid with the specified contents,
        or experiencing the specified event (either 'hit-wall' or 'moved'). The
        contributions of content reward and event reward are summed. For
        example, you might specify a cost for moving by passing
        rewards={'*': 10, 'moved': -1}.

    terminal_markers: sequence of chars, default '*'
        A grid cell containing any of these markers will be considered a
        "terminal" state.

    action_error_prob: float
        With this probability, the requested action is ignored and a random
        action is chosen instead.

    random_state: None, int, or RandomState object
        For repeatable experiments, you can pass a random state here. See
        http://scikit-learn.org/stable/modules/generated/sklearn.utils.check_random_state.html

    Notes
    -----

    Maze topology is expressed textually. Key:
     '#': wall
     '.': open (really, anything that's not '#')
     '*': goal
     'o': origin
    """

    def __init__(self, maze, rewards={'*': 10}, terminal_markers='*', action_error_prob=0, random_state=None, directions="NSEW"):

        self.maze = Maze(maze) if not isinstance(maze, Maze) else maze
        self.rewards = rewards
        self.terminal_markers = terminal_markers
        self.action_error_prob = action_error_prob
        self.random_state = check_random_state(random_state)

        self.actions = [maze_actions[direction] for direction in directions]
        self.num_actions = len(self.actions)
        self.state = None
        self.reset()
        self.num_states = self.maze.shape[0] * self.maze.shape[1]

    def __repr__(self):
        return 'GridWorld(maze={maze!r}, rewards={rewards}, terminal_markers={terminal_markers}, action_error_prob={action_error_prob})'.format(**self.__dict__)

    def reset(self):
        """
        Reset the position to a starting position (an 'o'), chosen at random.
        """
        options = self.maze.positions_containing('o')
        self.state = options[self.random_state.choice(len(options))]

    def observe(self):
        """
        Return the current state as an integer, or None if the episode is over.

        The state is the index into the flattened maze.
        """
        if self.state is None:
            return None
        else:
            return np.ravel_multi_index(self.state, self.maze.shape)

    def perform_action(self, action_idx):
        """Perform an action (specified by index), yielding a new state and reward."""
        # In the absorbing end state, nothing does anything.
        if self.state is None:
            return self.observe(), 0

        if self.action_error_prob and self.random_state.rand() < self.action_error_prob:
            action_idx = self.random_state.choice(self.num_actions)
        action = self.actions[action_idx]
        new_state, result = move_avoiding_walls(self.maze, self.state, action)
        self.state = new_state

        reward = self.rewards.get(self.maze[new_state], 0) + self.rewards.get(result, 0)
        if self.maze[new_state] in self.terminal_markers:
            # Episode complete.
            self.state = None
        return self.observe(), reward


    samples = {
        'trivial': [
            '###',
            '#o#',
            '#.#',
            '#*#',
            '###'],

        'larger': [
            '#########',
            '#..#....#',
            '#..#..#.#',
            '#..#..#.#',
            '#..#.##.#',
            '#....*#.#',
            '#######.#',
            '#o......#',
            '#########']
    }
