### ENVIRONMENT FOR TESTING flat agents (Flat and Flat-ATT)

"""Initializes the SAR environment. It sets up the grid, start state, victim locations, hazards (ditches and fires),
points of interest (POIs), information location, and various rewards and penalties. It also prepares visualization tools.
"""
import torch
import matplotlib.pyplot as plt
from IPython.display import clear_output
from PIL import Image
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm.auto import tqdm
import pandas as pd
from IPython.display import clear_output
import time
from PIL import Image
from io import BytesIO
import torch
from termcolor import colored
import matplotlib.colors as mcolors
import os
import gym
from gym import spaces
from ClassDisasterResponseAssistant import DisasterResponseAssistant
from utils_functions import get_file_type



### environment setup used for flat agents integrated with LLM
class SARenvLLM:
    def __init__(self, gridsize, startState, victimStates, ditches, fires, POIs, infoLocation, image_path, document_path, mode='prod'):
        self.gridsize = gridsize              # size of the grid environment
        self.startState = startState          # The starting state of the agent
        self.victimStates = victimStates      # The locations where victims are present.
        self.ditches = ditches                # The locations representing obstacles (ditches) -- can't go through
        self.fires = fires                    # The locations representing fires.
        self.POIs = POIs                      #  Points of interest in the environment.
        self.infoLocation = infoLocation  # Location where the agent needs to ask for information
        self.maxSteps = 1 * (self.gridsize[0] * self.gridsize[1])  # Example step limit
        self.ditchPenalty = -30
        self.savePenalty = -5
        self.turnPenalty = -1
        self.askingReward = 6
        self.wrongAskPenalty = -5
        self.exceedStepPenalty = -5
        self.winReward = 100
        self.image_path = image_path
        self.document_path = document_path
        self.mode = mode
        
        document_type = get_file_type(self.document_path)
        self.assistant = DisasterResponseAssistant(self.document_path, document_type)
        self.ask_action_counter = 0
        self.hazards = []
        self.pois = []
        # Define the correct order of information collection
        self.referenceSequence = ['X', 'Y', 'Z']
        # Extend the initialization method to include the new info locations
        self.infoLocations = {
            (6, 1): 'X',
            (4, 4): 'Y',
            (2, 6): 'Z'
        }
        # Initialize a list to keep track of the types of information collected
        self.infoCollected = []



        self.desired_cell_size = 50  # The size of cells in the grid for visualization
        self.generate_annotated_image()

        self.create_statespace()
        self.create_statespace_VISUALISATION()
        self.stateCount = self.get_statespace_len()
        self.stateDict = {k: v for k, v in zip(self.statespace, range(self.stateCount))}
        self.currentState = (self.startState[0], False, False, False)  # State includes position and info statusX, and info statusY
        self.actionspace = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]  # Actions: 'UP', 'DOWN', 'LEFT', 'RIGHT', 'ASK', 'SAVE'
        self.actionDict = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT', 4: 'ASK', 5: 'SAVE', 6: 'ASK_Y', 7: 'ASK_Z', 
                           8: 'ASK_A', 9: 'ASK_B', 10: 'ASK_C', 11: 'use', 12: 'remove', 13: 'carry'}
        self.actionCount = self.get_actionspace_len()
        

        self.last_reward_state = None    # Attribute to remember the last reward state
        
        self.victim_saved = False
        self.isGameEnd = False
        self.visited_information_state = False
        self.response = None
        self.totalTurns = 0
        self.sensor_readings = {state: 1.0 for state in self.get_statespace()}
    

        self.keywords_for_POIs = ["victim", "trail", "potential sighting", "Screams", "shelter", "high ground", "water source",
                                  "access route", "last known position", "high probability area", "safe"]
        self.keywords_for_danger = ["fire", "heat", "smoke", "restricted", "no access allowed", "flames", "dangerous", "steep terrain",
                                    "dense vegetation", "unstable structures", "unstable buildings", "hazardous material", "unsafe"]

        self.locationsDict = {
            'hospital': (0, 3),
            'train station': (5, 6), ## (6, 5)
            'school': (3, 0),
            'mall': (4, 1),  # (1, 1)
            'bank': (6, 5),
            'restaurant': (2, 0),
            'shop': (1, 2),
            'bakery': (3, 6),
            'petrol station': (2, 5)
        }

        if self.mode == 'debug':
            print("State Space", self.statespace)
            print("State Dict", self.stateDict)
            print("Action Space", self.actionspace)
            print("Action Dict", self.actionDict)
            print("Start State", self.startState)
            print("Terminal States", self.victimStates)
            print("Ditches", self.ditches)
            print("WinReward:{}, TurnPenalty:{}, DitchPenalty:{}, savePenalty:{}, AskingReward:{}, \
                WrongAskPenalty:{}".format(self.winReward, self.turnPenalty, self.ditchPenalty, \
                    self.savePenalty, self.askingReward, self.wrongAskPenalty))

    def set_mode(self, mode):
        self.mode = mode
    def create_statespace(self):
        self.statespace = [((row, col), info_statusX, info_statusY, info_statusZ)
                           for row in range(self.gridsize[0])
                           for col in range(self.gridsize[1])
                           for info_statusX in [False, True]
                           for info_statusY in [False, True]
                           for info_statusZ in [False, True]]
    def get_statespace(self): return self.statespace
    def get_actionspace(self): return self.actionspace
    def get_statespace_len(self): return len(self.statespace)
    def get_actionspace_len(self): return len(self.actionspace)
    def get_actiondict(self): return self.actionDict
    def create_statespace_VISUALISATION(self):
        self.statespace_vis = [((row, col)) for row in range(self.gridsize[0]) for col in range(self.gridsize[1])]
    def get_statespace_VISUALISATION(self): return self.statespace_vis


    ### Handles the 'ASK' action, providing the agent with environmental information
    def ask_action(self, state):
        self.ask_action_counter += 1
        position, info_statusX, info_statusY, info_statusZ = state
        verbal_inputs = []
        if not info_statusZ:
            VERBAL_INPUT1 = "Hey, there's a victim at the hospital and fire was reported at the train station and the bank. A safe area is the mall and make sure you keep an eye on the access route in the school, restaurant, and shop. There are also reports of significant instances of heat at the bakery. Police told us that no access allowed around the petrol station."
            verbal_inputs.append(VERBAL_INPUT1)
            
            if self.ask_action_counter <= 1:
                print(f"real LLM is about to start handling the input {VERBAL_INPUT1}")
                for input_text in verbal_inputs:
                    response = self.assistant.generate_response(input_text)
                    if response:
                        self.visited_information_state = True
                    self.hazards, self.pois = self.assistant.refine_response(response)
                    print(f"real LLM is about to end handling the input {VERBAL_INPUT1}")
                    self.update_environment_REAL(self.hazards, self.pois)
            else:
                # #print(f"input will be handled hereby by pseudoLLM")
                # print(self.hazards, self.pois)
                self.visited_information_state = True
                self.update_environment_REAL(self.hazards, self.pois)
                
    
    def update_environment_REAL(self, haz, poi):
        for hazardous_location in haz:
            self.sensor_readings[(hazardous_location, True, True, True)] = -10.0
            self.fires.append(hazardous_location)
        for safe_location in poi:
            self.sensor_readings[(safe_location, True, True, True)] = 10.0
            self.POIs.append(safe_location)
    

    ### Determines the next state of the agent based on the current state and action. It handles different actions
    ### like movement, asking for information, and saving victims
    def next_state_vision(self, current_state, action):
        position, info_statusX, info_statusY, info_statusZ = current_state
        s_row, s_col = position
        # Handle movement actions
        if action == 0:  # Move Up
            next_row = max(0, s_row - 1)
        elif action == 1:  # Move Down
            next_row = min(self.gridsize[0] - 1, s_row + 1)
        else:
            next_row = s_row
        if action == 2:  # Move Left
            next_col = max(0, s_col - 1)
        elif action == 3:  # Move Right
            next_col = min(self.gridsize[1] - 1, s_col + 1)
        else:
            next_col = s_col
        # Stay in the cell for ASK and SAVE actions
        if action in [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]:
            next_row, next_col = position
        next_position = (next_row, next_col)
        next_info_statusX = info_statusX
        next_info_statusY = info_statusY
        next_info_statusZ = info_statusZ
        # # Update info_status on ASK action
        # if action == 4 and position == self.infoLocation[0]:
        #     self.ask_action(current_state)
        #     info_status = True
        return (next_position, next_info_statusX, next_info_statusY, next_info_statusZ)


    ### Determines the next state of the agent based on the current state and action. It handles different actions
    ### like movement, asking for information, and saving victims
    def next_state(self, current_state, action):
        position, info_statusX, info_statusY, info_statusZ = current_state
        s_row, s_col = position
        next_info_statusX = info_statusX
        next_info_statusY = info_statusY
        next_info_statusZ = info_statusZ
        next_row, next_col = s_row, s_col
        # Handle movement actions
        if action == 0:  # Move Up
            next_row = max(0, s_row - 1)
        elif action == 1:  # Move Down
            next_row = min(self.gridsize[0] - 1, s_row + 1)
        elif action == 2:  # Move Left
            next_col = max(0, s_col - 1)
        elif action == 3:  # Move Right
            next_col = min(self.gridsize[1] - 1, s_col + 1)
        
        next_position = (next_row, next_col)
        # Update info_status on ASK action  
        if action == 4 and position == self.infoLocation[0] and not info_statusX:
            # self.ask_action(current_state)
            next_info_statusX = True
        
        # Update info_status on ASK action  
        elif action == 6 and position == self.infoLocation[1] and info_statusX:
            # self.ask_action(current_state)
            next_info_statusY = True
        
        elif action == 7 and position == self.infoLocation[2] and info_statusX and info_statusY:
            self.ask_action(current_state)
            next_info_statusZ = True
            
        
        if action == 5 and position == self.victimStates[0] and next_position == self.victimStates[0] and next_info_statusX and next_info_statusY and info_statusZ:
            self.victim_saved = True
            self.isGameEnd = True
        if next_position in self.ditches:
            self.isGameEnd = True
        if self.totalTurns >= self.maxSteps:
            self.isGameEnd = True
        return (next_position, next_info_statusX, next_info_statusY, next_info_statusZ)


    ### Calculates the reward based on the agent's actions and the current state.
    def compute_reward(self, state, next_state, action):
        _, info_statusX, info_statusY, info_statusZ = state
        pos_next, next_info_statusX, next_info_statusY, next_info_statusZ = next_state
        reward = self.turnPenalty
        if action == 5 and pos_next == self.victimStates[0]:
            if next_info_statusX and next_info_statusY and next_info_statusZ:
                reward += self.winReward  # winReward (+100)
            else:
                reward += self.savePenalty  # savePenalty (-10)
        elif action == 4:
            if pos_next == self.infoLocation[0] and not info_statusX:
                reward += self.askingReward 
            else:
                reward += self.wrongAskPenalty  # askingReward (+5) or wrongAskPenalty (-1)
        elif action == 6:
            if pos_next == self.infoLocation[1] and not info_statusY and info_statusX:
                reward += self.askingReward
            else:
                reward += self.wrongAskPenalty
        elif action == 7:
            if pos_next == self.infoLocation[2] and not info_statusZ and info_statusX and info_statusY:
                reward += self.askingReward
            else:
                reward += self.wrongAskPenalty
        elif pos_next in self.ditches:
            reward += self.ditchPenalty  # ditchPenalty (-20)
        elif self.totalTurns > self.maxSteps:
            reward += self.exceedStepPenalty
        else:
            reward
        return reward


    ### Advances the environment by one time step based on the agent's action
    def step(self, action):
        if self.isGameEnd:
            #print('game over')
            raise Exception('Game is Over')
        if action not in self.actionspace:
            raise ValueError('Invalid action taken')
        next_state = self.next_state(self.currentState, action)
        reward = self.compute_reward(self.currentState, next_state, action) #self.currentState
        self.currentState = next_state
        done = self.isGameEnd
        self.totalTurns += 1
        return self.currentState, reward, done, self.totalTurns


    ### Resets the environment to its initial state.
    def reset(self, start_state=None):
        self.isGameEnd = False
        self.visited_information_state = False
        self.victim_saved = False
        self.totalTurns = 0
        self.infoCollected = []
        self.currentState = start_state if start_state else (self.startState[0], False, False, False)
        self.POIs = []
        self.fires = []
        return self.currentState

    def reset_for_animation(self, start):
        self.isGameEnd = False
        self.visited_information_state = False
        self.totalTurns = 0
        self.currentState = (start, False)
        return self.currentState


    ### Methods for annotating and generating images of the environment grid
    def annotate_cell(self, ax, row, col, text, color='white'):
        """ Annotates a specific cell in the grid with given text and color. """
        ax.text(col * self.desired_cell_size + self.desired_cell_size/2,
                row * self.desired_cell_size + self.desired_cell_size/2,
                text,
                ha='center',
                va='center',
                color=color,
                fontsize=15,
                weight='bold')


    def generate_annotated_image(self):
        # Load and process the image
        img = Image.open(self.image_path)
        # Resize and crop image to fit the grid
        aspect_ratio = img.width / img.height
        if aspect_ratio > 1:
            new_size = (self.gridsize[0] * img.height // self.gridsize[1], img.height)
        else:
            new_size = (img.width, self.gridsize[1] * img.width // self.gridsize[0])
        img = img.resize(new_size, Image.Resampling.LANCZOS)
        crop_x = (img.width - self.gridsize[0] * (img.height // self.gridsize[1])) // 2
        crop_y = (img.height - self.gridsize[1] * (img.width // self.gridsize[0])) // 2
        img = img.crop((crop_x, crop_y, img.width - crop_x, img.height - crop_y))
        # Resize image to have equal cells
        new_image_width = self.desired_cell_size * self.gridsize[0]
        new_image_height = self.desired_cell_size * self.gridsize[1]
        img = img.resize((new_image_width, new_image_height))
        # Create figure with annotations
        if not self.gridsize == [17, 17]:
            fig, ax = plt.subplots(figsize=(8, 8))
        else:
            fig, ax = plt.subplots(figsize=(15, 15))
        ax.imshow(img)
        # Draw grid lines in white
        ax.set_xticks([i * self.desired_cell_size for i in range(self.gridsize[0] + 1)], minor=False)
        ax.set_yticks([i * self.desired_cell_size for i in range(self.gridsize[1] + 1)], minor=False)
        ax.grid(which="both", color="white", linestyle='-', linewidth=2)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        # Add annotations for each cell
        for row in range(self.gridsize[1]):
            for col in range(self.gridsize[0]):
                cell_coord = (row, col)
                # Check for special states and annotate accordingly
                if cell_coord in self.startState:
                    self.annotate_cell(ax, row, col, 'START', color='coral')  # Start
                elif cell_coord in self.victimStates:
                    self.annotate_cell(ax, row, col, 'VIC', color='cyan')  # Terminal
                elif cell_coord in self.ditches:
                    self.annotate_cell(ax, row, col, 'D', color='red')  # Ditch
                elif cell_coord in self.fires:
                    self.annotate_cell(ax, row, col, 'F', color='orange')  # Fire
                elif cell_coord in self.POIs:
                    self.annotate_cell(ax, row, col, 'P', color='purple')  # POI
                elif cell_coord in self.infoLocation:
                    self.annotate_cell(ax, row, col, 'INFO', color='yellow')  # Info
                else:
                    self.annotate_cell(ax, row, col, str(cell_coord))  # Regular cell coordinate
        # Save figure to buffer and load as PIL image
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        self.annotated_image = Image.open(buf)


    ### Visualizes the current state of the environment.
    ### Visualizes the current state of the environment.
    def render(self):
        clear_output(wait=True)
        fig, ax = plt.subplots(figsize=(8, 8))#plt.subplots()
        ax.set_xlim(0, self.gridsize[0])
        ax.set_ylim(0, self.gridsize[1])
        ax.set_xticks(range(self.gridsize[0]))
        ax.set_yticks(range(self.gridsize[1]))
        ax.grid(which='both')
        # Plotting the agent
        agent_pos, _, _, _ = self.currentState
        ax.plot(agent_pos[1] + 0.5, agent_pos[0] + 0.5, 'o', color='blue', markersize=10)  # Agent as a blue dot
        # Plotting the ditches
        for ditch in self.ditches:
            ax.plot(ditch[1] + 0.5, ditch[0] + 0.5, 'x', color='red', markersize=10)
        # Plotting the terminal state (victim's location)
        for terminal in self.victimStates:
            ax.plot(terminal[1] + 0.5, terminal[0] + 0.5, 'P', color='green', markersize=10)
        for poi in self.POIs:
            ax.plot(poi[1] + 0.5, poi[0] + 0.5, 'P', color='pink', markersize=10)
        for hazard in self.fires:
            ax.plot(hazard[1] + 0.5, hazard[0] + 0.5, 'x', color='orange', markersize=10)
        # Plotting the info location
        for info in self.infoLocation:
            ax.plot(info[1] + 0.5, info[0] + 0.5, '*', color='yellow', markersize=10)
        plt.gca().invert_yaxis()
        plt.show()
        
        


class HRLSARenvLLM(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self):
        super(HRLSARenvLLM, self).__init__()
        self.action_space = spaces.Discrete(4)  # Up, Down, Left, Right for EXPLORE
        self.observation_space = spaces.MultiDiscrete([7, 7, 2, 2, 2, 2])  # x, y, Info collected X, info collected Y, info_collectedZ, Victim saved
        self.state = [4, 1, 0, 0, 0, 0]  # (x, y, info_collectedX, info_collectedY, info_collectedZ, victim_saved)
        self.info_location = [(4, 4), (6, 2), (5, 5)]
        self.final_location = (0, 3)
        self.current_option = 0  # Starting option will be decided by the manager
        self.grid_size = 7
        self.max_steps = self.grid_size * self.grid_size
        self.total_turns = 0

        self.winReward = 118
        self.askingReward = 20
        self.exceedStepPenalty = -5
        self.turnPenalty = -1
        self.wrongAskPenalty = -5
        self.ditchPenalty = -30
        self.savePenalty = -5

        self.POIs, self.fires, self.hazards, self.pois = [], [], [], []
        self.visited_information_state = False
        self.ask_action_counter = 0
        document_path = "/Users/dimipan/Documents/HRL-LLM/data/sar_data.json"
        document_type = get_file_type(document_path)
        self.assistant = DisasterResponseAssistant(document_path, document_type)


        self.ditches = [(1, 6), (2, 2), (2, 4), (3, 2), (3, 3), (3, 4), (4, 5), \
                        (5, 0), (5, 1), (5, 2), (6, 0), (0, 2), (0, 4)]
        
        self.sensor_readings = {}
        
        self.keywords_for_POIs = ["victim", "trail", "potential sighting", "Screams", "shelter", "high ground", "water source",
                                  "access route", "last known position", "high probability area", "safe"]
        self.keywords_for_danger = ["fire", "heat", "smoke", "restricted", "no access allowed", "flames", "dangerous", "steep terrain",
                                    "dense vegetation", "unstable structures", "unstable buildings", "hazardous material", "unsafe"]

        self.locationsDict = {
            'hospital': (0, 3),
            'train station': (5, 6), ## (6, 5)
            'school': (3, 0),
            'mall': (4, 1),  # (1, 1)
            'bank': (6, 5),
            'restaurant': (2, 0),
            'shop': (1, 2),
            'bakery': (3, 6),
            'petrol station': (2, 5)
        }
    

    ### Handles the 'ASK' action, providing the agent with environmental information
    def ask_action(self, state):
        self.ask_action_counter += 1
        x, y, info_collectedX, info_collectedY, info_collectedZ, victim_saved = state
        verbal_inputs = []
        if not info_collectedZ:
            VERBAL_INPUT1 = "Hey, there's a victim at the hospital and fire was reported at the train station and the bank. A safe area is the mall and make sure you keep an eye on the access route in the school, restaurant, and shop. There are also reports of significant instances of heat at the bakery. Police told us that no access allowed around the petrol station."
            verbal_inputs.append(VERBAL_INPUT1)
            
            if self.ask_action_counter <= 1:
                print(f"real LLM is about to start handling the input {VERBAL_INPUT1}")
                for input_text in verbal_inputs:
                    response = self.assistant.generate_response(input_text)
                    if response:
                        self.visited_information_state = True
                    self.hazards, self.pois = self.assistant.refine_response(response)
                    print(f"real LLM is about to end handling the input {VERBAL_INPUT1}")
                    self.update_environment_REAL(self.hazards, self.pois)
            else:
                # #print(f"input will be handled hereby by pseudoLLM")
                # print(self.hazards, self.pois)
                self.visited_information_state = True
                self.update_environment_REAL(self.hazards, self.pois)
    
    def update_environment_REAL(self, haz, poi):
        for hazardous_location in haz:
            self.sensor_readings[(hazardous_location[0], hazardous_location[1], 1, 1, 1, 0)] = -10.0
            self.fires.append(hazardous_location)
        for safe_location in poi:
            self.sensor_readings[(safe_location[0], safe_location[1], 1, 1, 1, 0)] = 10.0
            self.POIs.append(safe_location)
    
    def next_state_vision(self, target, action):
        x, y, _, _, _, _ = target
        if action == 0:  # Up
            x = max(x - 1, 0)
        elif action == 1:  # Down
            x = min(x + 1, self.grid_size - 1)
        elif action == 2: # Left
            y = max(y - 1, 0)
        elif action == 3:  # Right
            y = min(y + 1, self.grid_size - 1)
        next_position_x, next_position_y = x, y
        return (next_position_x, next_position_y)

    def step(self, action):
        x, y, info_collectedX, info_collectedY, info_collectedZ, victim_saved = self.state
        reward = self.turnPenalty  # Small penalty for each step
        done = False
        self.total_turns += 1
        if self.total_turns >= self.max_steps:
            reward += self.exceedStepPenalty  ## (-5)
            done = True
            self.current_option = 0

        if self.current_option == 0: ## EXPLORE
            if (tuple([x, y]) != self.info_location[0] and not info_collectedX) or (tuple([x, y]) != self.info_location[1] and not info_collectedY and info_collectedX) or (tuple([x, y]) != self.info_location[2] and not info_collectedZ and info_collectedX and info_collectedY):  # EXPLORE (heading to info location X)
                if action == 0:  # Up
                    x = max(x - 1, 0)
                elif action == 1:  # Down
                    x = min(x + 1, self.grid_size - 1)
                elif action == 2: # Left
                    y = max(y - 1, 0)
                elif action == 3:  # Right
                    y = min(y + 1, self.grid_size - 1)
            if (tuple([x, y]) == self.info_location[0] and not info_collectedX) or (tuple([x, y]) == self.info_location[1] and not info_collectedY and info_collectedX) or (tuple([x, y]) == self.info_location[2] and not info_collectedZ and info_collectedX and info_collectedY): ## switch to COLLECT
                self.current_option = 1
                
            if tuple([x, y]) != self.final_location and info_collectedX and info_collectedY and info_collectedZ: ## EXPLORE (heading to final location)
                if action == 0:  # Up
                    x = max(x - 1, 0)
                elif action == 1:  # Down
                    x = min(x + 1, self.grid_size - 1)
                elif action == 2: # Left
                    y = max(y - 1, 0)
                elif action == 3:  # Right
                    y = min(y + 1, self.grid_size - 1)
            if tuple([x, y]) == self.final_location and info_collectedX and info_collectedY and info_collectedZ: ## switch to SAVE
                self.current_option = 2
                
            if tuple([x, y]) in self.ditches:
                reward += self.ditchPenalty  ## (-20)
                done = True
                self.current_option = 0
            
        
        elif self.current_option == 1:
            if tuple([x, y]) == self.info_location[0]:
                if action == 0:  # Assume action 0 is "getX"
                    if not info_collectedX:
                        info_collectedX = 1
                        # reward += self.askingReward  # Reward for collecting the info for the first time
                    else:
                        pass
                        # reward += self.wrongAskPenalty  # Penalty for attempting to collect again
                    self.current_option = 0  # Switch back to EXPLORE after collection attempt

            if tuple([x, y]) == self.info_location[1]:
                if action == 1:  # Assume action 1 is "getY"
                    if not info_collectedY and info_collectedX:
                        # self.ask_action(self.state)
                        info_collectedY = 1
                        # reward += self.askingReward  # Reward for collecting the info for the first time
                    else:
                        pass
                        # reward += self.wrongAskPenalty  # Penalty for attempting to collect again
                    self.current_option = 0  # Switch back to EXPLORE after collection attempt
            
            if tuple([x, y]) == self.info_location[2]:
                if action == 2:  # Assume action 2 is "getZ"
                    if not info_collectedZ and info_collectedX and info_collectedY:
                        self.ask_action(self.state)
                        info_collectedZ = 1
                        # reward += self.askingReward  # Reward for collecting the info for the first time
                    else:
                        pass
                        # reward += self.wrongAskPenalty  # Penalty for attempting to collect again
                    self.current_option = 0  # Switch back to EXPLORE after collection attempt
            

        elif self.current_option == 2:
            if tuple([x, y]) == self.final_location:
                if action == 0 and info_collectedX and info_collectedY and info_collectedZ:  # Assume action 0 is "save"
                    if not victim_saved:
                        victim_saved = 1
                        reward += self.winReward  # High reward for saving the victim
                        done = True
                    else:
                        reward += self.savePenalty # Penalty for redundant saving
                else:
                    reward += self.savePenalty  # Penalty if save action is taken without collected info or at wrong location


        self.state = [x, y, info_collectedX, info_collectedY,info_collectedZ, victim_saved]
        return self.state, reward, done, {}

    def reset(self):
        self.total_turns = 0
        self.POIs = []
        self.fires = []
        self.visited_information_state = False
        self.state = [4, 1, 0, 0, 0, 0]
        self.current_option = 0  # Start with EXPLORE
        return self.state

    def render(self, mode='human'):
        grid = np.zeros((7, 7), dtype=int)
        x, y, _, _ = self.state
        grid[y, x] = 1
        print("Current Grid:\n", grid)