from continual_learning_utils import *
import random
import pandas as pd
import networkx as nx
import logging
from random import shuffle
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import random
import string
import os
import re
import glob
import csv
import torch
from wonderwords import RandomWord
import os
import gc
import pickle
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
from itertools import permutations
import logging
from random import shuffle
from matplotlib import pyplot as plt
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import math

r = RandomWord()

def create_unique_random_grid(nouns, size=3):
    """Creates a size x size grid with unique random nouns."""
    random_nouns = random.sample(nouns, size * size)
    return [random_nouns[i * size:(i + 1) * size] for i in range(size)]

def find_shortest_paths(grid, start_name, end_name):
    """Finds all shortest paths from start_name to end_name in a grid. """
    # Find coordinates of start and end points
    start = end = None
    for i, row in enumerate(grid):
        for j, name in enumerate(row):
            if name == start_name:
                start = (i, j)
            if name == end_name:
                end = (i, j)
    
    # Check if start or end points were not found
    if start is None or end is None:
        print ("start or end not found")
        return []

    paths = []
    start_x, start_y = start
    end_x, end_y = end

    # Total horizontal and vertical distances
    x_dist = end_x - start_x
    y_dist = end_y - start_y

    # Generate a list of directions taken in the shortest path
    # We know that the shortest route is x_dist EAST or WESTs, and y_dist NORTH or SOUTHs
    hor_moves = ['EAST' if x_dist > 0 else 'WEST'] * abs(x_dist)
    ver_moves = ['SOUTH' if y_dist > 0 else 'NORTH'] * abs(y_dist)
    all_moves = hor_moves + ver_moves

    # We have a list, e.g. [NORTH, NORTH, EAST, EAST] and we want to find all possible orderings
    # Each ordering (i.e. permutation) is a possible shortest path from start_name to end_name
    for path in set(permutations(all_moves, len(all_moves))):
        sequence = [f'FROM: {start_name}, TO: {end_name}, PATH: {start_name}']
        x, y = start
        for direction in path:
            if direction == 'EAST' and x < 2:
                x += 1
            elif direction == 'WEST' and x > 0:
                x -= 1
            elif direction == 'SOUTH' and y < 2:
                y += 1
            elif direction == 'NORTH' and y > 0:
                y -= 1
            else:
                # Invalid move, skip this path
                break
            sequence.append(f"{direction} {grid[x][y]}")

            # add the path when it successfully reaches the end point
            if (x, y) == end:
                paths.append(' '.join(sequence))

    return paths

def shuffle_stimuli(stimuli):
    random.shuffle(stimuli)
    return stimuli

def get_all_paths_for_grid(grid):
    all_paths = []
    items = [item for sublist in grid for item in sublist]
    for start in items:
        for end in items:
            if start != end:
                all_paths.extend(find_shortest_paths(grid, start, end))
    return shuffle_stimuli(all_paths)

def prepare_data(default=False, horizontal_and_vertical=False, short_paths=False):
    training_strs = []
    testing_strs = []
    for i in range(5):
        nouns = [r.word(include_parts_of_speech=["nouns"]).replace(" ", "_") for _ in range(9)]
        grid = create_unique_random_grid(nouns)
        print(grid)
        pths = get_all_paths_for_grid(grid)
        if default == True:
            training_strs.append(pths[0:120])
            testing_strs.append(pths[120:])
        if short_paths == True:
            sorted_pths = sorted(pths, key=len)
            training_strs.append(sorted_pths[0:120])
            testing_strs.append(sorted_pths[120:])
        if horizontal_and_vertical == True:
            special_paths = construct_special_paths(grid)
            training_strs.append(special_paths)
            testing_strs.append([p for p in pths if p not in special_paths][0:20])
    
    for env in range(5):
        if os.path.exists(f"spatial_model_{env}") is False:
            os.mkdir(f"spatial_model_{env}")
        text_file = open(f"spatial_model_{env}/test.txt", "w")
        n = text_file.write('\n'.join(testing_strs[env]))
        text_file.close()
    
    return training_strs, testing_strs

def construct_special_paths(grid):
    shortest_paths = []
    
    # Horizontal and Vertical Paths
    for i in range(3):
        # Horizontal
        horizontal_path = f"FROM: {grid[i][0]}, TO: {grid[i][2]}, PATH: {grid[i][0]} EAST {grid[i][1]} EAST {grid[i][2]}"
        shortest_paths.append(horizontal_path)
        horizontal_path = f"FROM: {grid[i][2]}, TO: {grid[i][0]}, PATH: {grid[i][2]} WEST {grid[i][1]} WEST {grid[i][0]}"
        shortest_paths.append(horizontal_path)
        # Vertical
        vertical_path = f"FROM: {grid[0][i]}, TO: {grid[2][i]}, PATH: {grid[0][i]} SOUTH {grid[1][i]} SOUTH {grid[2][i]}"
        shortest_paths.append(vertical_path)
        vertical_path = f"FROM: {grid[2][i]}, TO: {grid[0][i]}, PATH: {grid[2][i]} NORTH {grid[1][i]} NORTH {grid[0][i]}"
        shortest_paths.append(vertical_path)
    
    return shortest_paths
