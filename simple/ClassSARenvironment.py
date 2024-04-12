"""Initializes the SAR environment. It sets up the grid, start state, victim locations, hazards (ditches and fires),
points of interest (POIs), information location, and various rewards and penalties. It also prepares visualization tools.
"""


import matplotlib.pyplot as plt
from IPython.display import clear_output
from PIL import Image
from io import BytesIO
from ClassDisasterResponseAssistant import DisasterResponseAssistant
import os

class SARenv:
    def __init__(self, gridsize, startState, victimStates, ditches, fires, POIs, infoLocation, image_path, document_path, mode='prod'):
        self.gridsize = gridsize              # size of the grid environment
        self.startState = startState          # The starting state of the agent
        self.victimStates = victimStates      # The locations where victims are present.
        self.ditches = ditches                # The locations representing obstacles (ditches) -- can't go through
        self.fires = fires                    # The locations representing fires.
        self.POIs = POIs                      #  Points of interest in the environment.
        self.infoLocation = infoLocation  # Location where the agent needs to ask for information
        self.maxSteps = 2 * (self.gridsize[0] * self.gridsize[1])  # Example step limit
        self.ditchPenalty = -20
        self.savePenalty = -10
        self.turnPenalty = -1
        self.askingReward = 5
        self.wrongAskPenalty = -1
        self.winReward = 100
        self.mode = mode
        
        self.document_type = self.get_file_type(document_path)
        self.assistant = DisasterResponseAssistant(document_path, self.document_type)
        self.ask_action_counter = 0
        self.hazards = []
        self.pois = []

        self.desired_cell_size = 50  # The size of cells in the grid for visualization
        self.generate_annotated_image(image_path)

        self.create_statespace()
        self.create_statespace_VISUALISATION()
        self.stateCount = self.get_statespace_len()
        self.stateDict = {k: v for k, v in zip(self.statespace, range(self.stateCount))}
        self.currentState = (self.startState[0], False)  # State includes position and info status
        self.actionspace = [0, 1, 2, 3, 4, 5]  # Actions: 'UP', 'DOWN', 'LEFT', 'RIGHT', 'ASK', 'SAVE'
        self.actionDict = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT', 4: 'ASK', 5: 'SAVE'}
        self.actionCount = self.get_actionspace_len()
        
        self.last_reward_state = None    # Attribute to remember the last reward state
        
        self.victim_saved = False
        self.isGameEnd = False
        self.visited_information_state = False
        self.response = None
        self.totalTurns = 0
        self.sensor_readings = {state: 1.0 for state in self.get_statespace()}

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
        self.statespace = [((row, col), info_status) for row in range(self.gridsize[0]) for col in range(self.gridsize[1]) for info_status in [False, True]]
    def get_statespace(self): return self.statespace
    def get_actionspace(self): return self.actionspace
    def get_statespace_len(self): return len(self.statespace)
    def get_actionspace_len(self): return len(self.actionspace)
    def get_actiondict(self): return self.actionDict
    def create_statespace_VISUALISATION(self):
        self.statespace_vis = [((row, col)) for row in range(self.gridsize[0]) for col in range(self.gridsize[1])]
    def get_statespace_VISUALISATION(self): return self.statespace_vis

    
    def get_file_type(self, docs_path):
        # Split the path and get the extension
        _, file_extension = os.path.splitext(docs_path)
        # Return the file extension without the period
        return file_extension[1:] if file_extension else None


    ### Handles the 'ASK' action, providing the agent with environmental information
    def ask_action(self, state):
        self.ask_action_counter += 1
        position, info_status = state
        verbal_inputs = []
        if not info_status:
            self.last_reward_state = (position, True) # Update the last_reward_state here upon successful information retrieval
            #print(f"last reward state {self.last_reward_state}")
            VERBAL_INPUT1 = "Hey, there's a victim at the hospital and I think I also saw fire in the train station and the bank. Hey, you wait! Someone told me about screams heard at the school and close to the mall. Hurry!"
            #VERBAL_INPUT2 = "Watch out, fire was reported next to the mall and opposite to the school. There's a shelter through the bank and the train station."
            #VERBAL_INPUT3 = "Screams were heard at the shop and close to the restaurant"
            verbal_inputs.append(VERBAL_INPUT1)
            #verbal_inputs.append(VERBAL_INPUT2)
            #verbal_inputs.append(VERBAL_INPUT3)
            
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
                #print(f"input will be handled hereby by pseudoLLM")
                #print(self.hazards, self.pois)
                self.visited_information_state = True
                self.update_environment_REAL(self.hazards, self.pois)
                # for input_text in verbal_inputs:
                #     self.simulate_LLM_process_alternative(input_text)
                
    
    def update_environment_REAL(self, haz, poi):
        for hazardous_location in haz:
            self.sensor_readings[(hazardous_location, True)] = -10.0
            self.fires.append(hazardous_location)
        for safe_location in poi:
            self.sensor_readings[(safe_location, True)] = 10.0
            self.POIs.append(safe_location)
            
            

    ### Determines the next state of the agent based on the current state and action. It handles different actions
    ### like movement, asking for information, and saving victims
    def next_state_vision(self, current_state, action):
        position, info_status = current_state
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
        if action in [4, 5]:
            next_row, next_col = position
        next_position = (next_row, next_col)
        # # Update info_status on ASK action
        # if action == 4 and position == self.infoLocation[0]:
        #     self.ask_action(current_state)
        #     info_status = True
        return (next_position, info_status)


    ### Determines the next state of the agent based on the current state and action. It handles different actions
    ### like movement, asking for information, and saving victims
    def next_state(self, current_state, action):
        position, info_status = current_state
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
        if action in [4, 5]:
            next_row, next_col = position
        next_position = (next_row, next_col)
        next_info_status = info_status
        # Update info_status on ASK action
        if action == 4 and next_position == self.infoLocation[0]:
            self.ask_action(current_state)
            #info_status = True
            next_info_status = True
            #self.last_reward_state = (position, next_info_status)
            #print(f"last reward state {self.last_reward_state}")
        # Check for game-ending conditions
        if action == 5 and position == self.victimStates[0] and next_position == self.victimStates[0] and info_status:
            self.victim_saved = True
            self.isGameEnd = True
        if next_position in self.ditches: #or (action == 5 and next_position == self.victimStates[0] and info_status):
            self.isGameEnd = True
        if self.totalTurns >= self.maxSteps:
            self.isGameEnd = True
        return (next_position, next_info_status)


    ### Calculates the reward based on the agent's actions and the current state.
    def compute_reward(self, state, next_state, action):
        _, info_status = state
        pos_next, _ = next_state
        reward = self.turnPenalty
        if action == 5:
            if info_status and pos_next == self.victimStates[0]:
                reward += self.winReward  # winReward (+100)
            else:
                reward += self.savePenalty  # savePenalty (-10)
        elif action == 4:
            if pos_next == self.infoLocation[0]:
                if not info_status:
                    reward += self.askingReward 
                else:
                    reward += self.wrongAskPenalty  # askingReward (+5) or wrongAskPenalty (-1)
            else:
                reward += self.wrongAskPenalty  # wrongAskPenalty (-1)
        elif pos_next in self.ditches:
            reward += self.ditchPenalty  # ditchPenalty (-20)
        # elif pos_next in self.fires and info_status:
        #     reward -= 10
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
        #self.currentState = (self.startState[0], False)
        self.currentState = start_state if start_state else (self.startState[0], False)
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


    def generate_annotated_image(self, image_path):
        # Load and process the image
        img = Image.open(image_path)
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
        agent_pos, _ = self.currentState
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

# # Example initialization and training
# gridsize = [7, 7]
# startState = [(4, 1)]
# victimStates = [(0, 3)]
# ditches = [(1, 6), (2, 1), (2, 2), (2, 4), (3, 2), (3, 3), (3, 4), (4, 5), \
#     (5, 0), (5, 1), (5, 2), (6, 0), (0, 2), (0, 4), (5, 5)]
# fires = []
# POIs = []  # Victim locations
# infoLocation = [(6, 1)]  # Location to ask for information was (6, 1)
# #image_path = "/home/research100/Desktop/disaster_area.jpg"
# #image_path = "/home/dimiubuntu/Documents/enhanced_RL/disaster_area.jpg"
# image_path = "/home/research100/Documents/sample/enhanced_RL/enhanced_RL/images/disaster_area.jpg"
# #image_path = "/content/disaster_area.jpg"

# env = SARenv(gridsize, startState, victimStates, ditches, fires, POIs, infoLocation, image_path, mode='debug')