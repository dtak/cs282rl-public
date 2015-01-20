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


def move_avoiding_walls(maze, position, action):
    # Compute new position
    new_position = position + action

    # Compute collisions with walls, including implicit walls at the ends of the world.
    if not maze.in_bounds(new_position) or maze[new_position] == '#':
        return position, 'hit-wall'

    return new_position, 'moved'



class FullyObservableSimpleMazeTask(object):
    """
    A simple task in a maze: get to the goal.

    Parameters
    ----------

    maze : list of strings or lists
        maze topology (see below)

    absorbing_end_state: boolean.
        If True, after reaching the goal, we go into an absorbing zero-reward end state.

    rewards: dict of string to number. default: {'*': 10}.
        Rewards obtained by being in a maze grid with the specified contents, or experiencing the
        specified event (e.g., 'hit-wall', 'moved'). The contributions of content reward and event
        reward are summed.

    Notes
    -----

    Maze topology is expressed textually. Key:
     '#': wall
     '.': open (really, anything that's not '#')
     '*': goal
     'o': origin
    """

    GOAL_MARKER = '*'

    # Internally, we represent being in the special absorbing end state as state=None.

    def __init__(self, maze, absorbing_end_state=False, rewards={'*': 10},
        action_error_prob=0, random_state=None):

        self.maze = maze
        self.absorbing_end_state = absorbing_end_state
        self.rewards = rewards
        self.action_error_prob = action_error_prob
        self.random_state = check_random_state(random_state)

        self.actions = [maze_actions[direction] for direction in "NSEW"]
        self.state = None
        self.reset()
        self.num_states = self.maze.shape[0] * self.maze.shape[1]
        if absorbing_end_state:
            self.num_states += 1

    def reset(self):
        """
        Reset the position to a starting position (an 'o'), chosen at random.
        """
        options = self.maze.positions_containing('o')
        self.state = options[self.random_state.choice(len(options))]

    def observe(self):
        """
        Return the current state as an integer.

        The state is the index into the flattened maze. If the end state is
        absorbing, the n*m+1th state is that absorbing state.
        """
        if self.state is None:
            return self.num_states - 1
        else:
            return np.ravel_multi_index(self.state, self.maze.shape)

    def perform_action(self, action_idx):
        """Perform an action (specified by index), yielding a new state and reward."""
        # In the absorbing end state, nothing does anything.
        if self.state is None:
            return self.observe(), 0

        if self.action_error_prob and self.random_state.rand() < self.action_error_prob:
            action_idx = self.random_state.choice(len(self.actions))
        action = self.actions[action_idx]
        new_state, result = move_avoiding_walls(self.maze, self.state, action)
        self.state = new_state

        reward = self.rewards.get(self.maze[new_state], 0) + self.rewards.get(result, 0)
        if self.maze[new_state] == self.GOAL_MARKER:
            # Reached goal.
            if self.absorbing_end_state:
                self.state = None
            else:
                self.reset()
        return self.observe(), reward
