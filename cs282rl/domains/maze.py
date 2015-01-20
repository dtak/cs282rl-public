import numpy as np
#from sklearn.util import check_random_state


# Maze state is represented as a 2-element NumPy array: (Y, X). Increasing Y is North.

# Possible actions, expressed as (delta-y, delta-x).
maze_actions = {
    'N': np.array([1, 0]),
    'S': np.array([-1, 0]),
    'E': np.array([0, 1]),
    'W': np.array([0, -1]),
}

def parse_topology(topology):
    return np.array([list(row) for row in topology])


def move_avoiding_walls(maze, state, action):
    # Compute new state
    new_state = state + action

    # Compute collisions with walls, including implicit walls at the ends of the world.
    if np.any(new_state < 0) or np.any(new_state >= maze.shape) or maze[new_state] == '#':
        return state, 'hit-wall'

    return new_state, 'moved'


def free_positions(maze):
    return np.transpose(np.nonzero(maze != '#'))


def random_initial_state(maze):
    """Choose randomly from the free positions in the maze."""
    options = free_positions(maze)
    return options[np.random.choice(len(options))]


class FullyObservableSimpleMaze(object):
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
    The task may define additional
     '*': goal
     'o': origin (if applicable.)
    """
    GOAL_MARKER = '*'
    # Internally, we represent being in the special absorbing end state as state=None.
    def __init__(self, maze, absorbing_end_state=False, rewards={'*': 10}):
        self.maze = parse_topology(maze)
        self.absorbing_end_state = absorbing_end_state
        self.rewards = rewards

        self.actions = [maze_actions[direction] for direction in "NSEW"]
        self.state = None
        self.reset()
        self.num_states = self.maze.shape[0] * self.maze.shape[1]
        if absorbing_end_state:
            self.num_states += 1

    def reset(self):
        self.state = random_initial_state(self.maze)

    def observe(self):
        """
        Fully observable. Use flattened indices.

        If the end state is absorbing, the n*m+1th state is that absorbing state.
        """
        if self.state is None:
            return self.num_states - 1
        else:
            return np.ravel_multi_index(self.state, self.maze.shape)

    def perform_action(self, action_idx):
        """Perform an action (specified by index), yielding a new state and reward."""
        action = self.actions[action_idx]
        new_state, result = perform_action(self.maze, self.state, action)
        self.state = new_state

        reward = self.rewards.get(self.maze[new_state], 0) + self.rewards.get(result, 0)
        if self.maze[new_state] == self.GOAL_MARKER:
            # Reached goal.
            if self.absorbing_end_state:
                self.state = None
            else:
                self.reset()
        return self.observe(), reward
