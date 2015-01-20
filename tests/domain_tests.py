from __future__ import absolute_import, print_function, unicode_literals, division

from nose.tools import *
from cs282rl import domains

maze_structure = structure_raw = [
    'xxxxxxxxx',
    'x..x...*x',
    'x..x..x.x',
    'x..x..x.x',
    'x..x.xx.x',
    'x.....x.x',
    'xxxxxxx.x',
    'x.......x',
    'xxxxxxxxx']


def test_maze():
    maze = domains.FullyObservableSimpleMaze(maze_structure)
