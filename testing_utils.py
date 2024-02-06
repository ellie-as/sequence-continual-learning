from continual_learning_utils import *
from grid_environment_utils import * 
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

def test_accuracy(model, test_data):
    correct_predictions = 0
    total_predictions = 0
    directions = ['NORTH', 'EAST', 'SOUTH', 'WEST']

    for sequence in test_data:
        # Find the first direction in the PATH and create the input sequence up to that point
        first_direction_index = next((i for i, word in enumerate(sequence.split()) if word in directions), None)
        if first_direction_index is not None:
            # Prepare the input sequence up to and including the first direction
            input_sequence = ' '.join(sequence.split()[:first_direction_index + 1])
            
            # Generate the model's prediction
            full_predicted_sequence = model.continue_input(input_sequence)
            # Remove the input part from the predicted sequence
            predicted_sequence = full_predicted_sequence[len(input_sequence):].strip()
            predicted_token = predicted_sequence.split()[0]  # First word of the generation

            # Extract the corresponding true token
            target_token = sequence.split()[first_direction_index + 1]

            # Compare the predicted token with the true token
            total_predictions += 1
            print(f"Correct location: {target_token}, Predicted location: {predicted_token}")
            if predicted_token == target_token:
                correct_predictions += 1

    # Calculate accuracy
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    return accuracy

def get_unique_locations(seqs):
    words_to_remove = {'NORTH', 'EAST', 'SOUTH', 'WEST', 'PATH:', 'TO:', 'FROM:'}
    unique_words = set(word for phrase in seqs for word in phrase.replace(',', '').split())
    unique_words = unique_words.difference(words_to_remove)
    return unique_words

def analyze_sequences(results):
    # Extract necessary data
    current_env = results['model']  # The current environment index
    training_strs = results['training_strs']
    testing_strs = results['testing_strs']
    train_size = results['train_size']
    seqs = results['seqs']

    # Initialize results storage
    analysis_results = {'real': [], 'valid': [], 'neither': []}

    # Iterate through each sequence in seqs
    for seq in seqs:
        found_as_real = False
        found_as_valid = False

        # Check for real sequences in previous phases
        for i in range(current_env):
            if seq in training_strs[i][0:train_size]:
                analysis_results['real'].append(seq)
                found_as_real = True
                break  # Stop searching if found

        if not found_as_real:
            # Check for valid sequences if not found as real
            for i in range(current_env):
                if seq in training_strs[i][train_size:] or seq in testing_strs[i]:
                    analysis_results['valid'].append(seq)
                    found_as_valid = True
                    break  # Stop searching if found

        if not (found_as_real or found_as_valid):
            # If the sequence is neither real nor valid
            analysis_results['neither'].append(seq)

    return analysis_results

def shortest_path_accuracy(model, test_data, all_data):
    results = []
    for seq in test_data:
        seq = model.continue_input(seq[:seq.index('PATH:')+5], do_sample=False)
        seq = seq[:seq.index('\n')]
            
        if seq in all_data:
            print(f"Valid: {seq}")
            results.append(1)
        else:
            print(f"Invalid: {seq}")
            results.append(0)
    return results.count(1) / len(results)

def test_data_subset(test_data, train_data):
    train_starts = [train_seq.split('PATH:')[0] for train_seq in train_data]
    # Filter test_data where the start is not in train_starts
    subset = [test_seq for test_seq in test_data if test_seq.split('PATH:')[0] not in train_starts]
    return subset