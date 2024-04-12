import matplotlib.pyplot as plt
from IPython.display import clear_output
from PIL import Image
from io import BytesIO
from ClassDisasterResponseAssistant import DisasterResponseAssistant
import os
import torch

class SARenvHRL:
    def __init__(self, gridsize, startState, finalState, ditches, fires, POIs, infoLocation, image_path, document_path, mode='prod'):
        self.gridsize = gridsize
        self.startState = startState
        self.finalState = finalState
        self.ditches = ditches
        self.infoLocation = infoLocation
        self.fires = fires
        self.POIs = POIs
        
        self.maxSteps = 2 * (self.gridsize[0] * self.gridsize[1])  # max steps needed for NAVIGATION -- EXPLORE
        self.ditchPenalty = -10
        self.turnPenalty = -1
        #self.savePenalty = -10
        self.exceedStepPenalty = -1
        self.wrongCollectPenalty = -5
        self.winReward = 100
    
        self.mode = mode
        
        self.desired_cell_size = 50  # The size of cells in the grid for visualization
        self.generate_annotated_image(image_path)
        
        self.actionspace = {'EXPLORE': ['up', 'down', 'left', 'right'], 
                     'COLLECT': ['get X', 'get Y', 'get Z'],
                     'SAVE': ['save']}
        self.optionspace = ['EXPLORE', 'COLLECT', 'SAVE']

        
        self.visited_information_state = False
        self.isGameEnd = False
        self.info_collectedX = False
        self.info_collectedY = False
        self.info_collectedZ = False
        self.totalTurns = 0
        self.totalAsks = 0
        self.totalSaves = 0
        self.victim_saved = False
        self.correct_sequence = ['X']
        
        self.collected_info = []
        self.ask_action_counter = 0
        
        self.document_type = self.get_file_type(document_path)
        self.assistant = DisasterResponseAssistant(document_path, self.document_type)
        self.ask_action_counter = 0
        self.hazards = []
        self.pois = []
        self.sensor_readings = {}
        
    
        self.current_state = (self.startState[0], self.info_collectedX, self.info_collectedY, self.info_collectedZ, self.victim_saved)
        if self.mode == 'debug':
            print("Initialization complete.")
        
        
        self.keywords_for_POIs = ["victim", "trail", "potential sighting", "Screams", "shelter", "high ground", "water source",
                                  "access route", "last known position", "high probability area", "safe"]
        self.keywords_for_danger = ["fire", "heat", "smoke", "restricted", "no access allowed", "flames", "dangerous", "steep terrain",
                                    "dense vegetation", "unstable structures", "unstable buildings", "hazardous material", "unsafe"]

        self.locationsDict = {
            'hospital': (0, 3),
            'train station': (5, 6),
            'school': (2, 0),
            'mall': (3, 0),
            'bank': (2, 5),
            'restaurant': (6, 5),
            'shop': (1, 1)
        }
    
    def reset(self):
        self.correct_sequence = ['X']
        self.visited_information_state = False
        self.collected_info = []
        self.info_collectedX, self.info_collectedY, self.info_collectedZ = False, False, False
        self.victim_saved = False
        self.isGameEnd = False
        self.totalTurns = 0
        self.totalAsks = 0
        self.totalSaves = 0
        self.position = self.startState[0]
        self.current_state = (self.position, self.info_collectedX, self.info_collectedY, self.info_collectedZ, self.victim_saved)
        return self.current_state
    
    def get_file_type(self, docs_path):
        # Split the path and get the extension
        _, file_extension = os.path.splitext(docs_path)
        # Return the file extension without the period
        return file_extension[1:] if file_extension else None

    ##### ----------------------------
    
    def next_state_vision(self, target_state, action):  ### here just movement, thus just EXPLORE option
        position = target_state
        next_position = position
        # Movement actions update
        if action == self.actionspace[self.optionspace[0]][0] and position[0] > 0:  # up
            next_position = (position[0] - 1, position[1])
        elif action == self.actionspace[self.optionspace[0]][1] and position[0] < self.gridsize[0] - 1:  # down
            next_position = (position[0] + 1, position[1])
        elif action == self.actionspace[self.optionspace[0]][2] and position[1] > 0:  # left
            next_position = (position[0], position[1] - 1)
        elif action == self.actionspace[self.optionspace[0]][3] and position[1] < self.gridsize[1] - 1:  # right
            next_position = (position[0], position[1] + 1)
        return (next_position)
    
    
    def ask_action(self, state):
        self.ask_action_counter += 1
        position, info_collectedX, info_collectedY, info_collectedZ, victim_saved = state
        verbal_inputs = []
        if not info_collectedX:
            
            # # VERBAL_INPUT1 = "Hey, there's a victim at the hospital."
            # # VERBAL_INPUT2 = "Watch out, fire was reported next to the mall."#. There's a shelter through the bank and the train station."
            # # #VERBAL_INPUT3 = "Someone told me about Screams heard at the shop and close to restaurant."
            # # verbal_inputs.append(VERBAL_INPUT1)
            # # verbal_inputs.append(VERBAL_INPUT2)
            # # #verbal_inputs.append(VERBAL_INPUT3)
            
            # VERBAL_INPUT1 = "Hey, there's a victim at the hospital."
            # VERBAL_INPUT2 = "Watch out, there is a shelter though the mall. Also, fire was reported at the restaurant."
            # VERBAL_INPUT3 = "Someone told me about Screams heard at the shop."
            # verbal_inputs.append(VERBAL_INPUT1)
            # verbal_inputs.append(VERBAL_INPUT2)
            # verbal_inputs.append(VERBAL_INPUT3)
            # for input_text in verbal_inputs:
            #         self.simulate_LLM_process_alternative(input_text)
    
    
    
            
            
            VERBAL_INPUT1 = "Hey, there's a victim at the hospital and I think I also saw fire in the train station." #and the bank. Hey, you wait! Someone told me about screams heard at the school and close to the mall. Hurry!"
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
                #print(f"input will be handled hereby by pseudoLLM")
                #print(self.hazards, self.pois)
                self.visited_information_state = True
                self.update_environment_REAL(self.hazards, self.pois)
                # for input_text in verbal_inputs:
                #     self.simulate_LLM_process_alternative(input_text)
    
    def update_environment_REAL(self, haz, poi):
        for hazardous_location in haz:
            self.sensor_readings[(hazardous_location, True, False, False, False)] = -10.0
            #self.sensor_readings[(hazardous_location, True)] = -10.0
            self.fires.append(hazardous_location)
        for safe_location in poi:
            self.sensor_readings[(safe_location, True, False, False, False)] = 10.0
            #self.sensor_readings[(safe_location, True)] = 10.0
            self.POIs.append(safe_location)
        
    
    # ### Simulates the process of obtaining information from a language model (multiple locations)
    # ### problem when two locations associated to different class are present in the smae sentence 
    # def simulate_LLM_process_alternative(self, input):
    #     sum_embedding = torch.tensor([0, 0], dtype=torch.float32)
    #     locations_in_input = []
    #     for location in self.locationsDict:
    #         # Check if the location keyword is in the information string
    #         if location in input:
    #             location_embedding = torch.tensor(self.locationsDict[location], dtype=torch.float32)
    #             sum_embedding += location_embedding
    #             locations_in_input.append(tuple(int(x) for x in location_embedding.tolist()))
    #     #return locations_in_input
    #     if locations_in_input:
    #         #print(f"response is {locations_in_input} -- when input is: {input}")
    #         self.visited_information_state = True
    #         #print(f"response is {response}")
    #         sentences = input.split(". ") if ". " in input else [input]
    #         #print(f"the sentences are: {sentences}")
    #         for sentence in sentences:
    #             is_poi = any(keyword in sentence for keyword in self.keywords_for_POIs)
    #             is_fire = any(keyword in sentence for keyword in self.keywords_for_danger)
    #             #print(f"In sentence '{sentence}' we have POI: {is_poi} and fire: {is_fire}")
    #             for location, location_coords in self.locationsDict.items():
    #                 if location in sentence:
    #                     info = tuple(int(x) for x in torch.tensor(location_coords, dtype=torch.float32).tolist())
    #                     #print(f"info now is {info} and poi {is_poi} and fire {is_fire}")
    #                     # If the location is already categorized, skip it
    #                     if info in self.POIs or info in self.fires:
    #                         continue
    #                     # Add location to POIs or fires based on the context of the sentence
    #                     self.update_environment(info, is_poi, is_fire)
    
    # def update_environment(self, info, is_poi, is_fire):
    #     if is_poi and not is_fire:
    #         self.sensor_readings[(info, True, False, False, False)] = 10.0
    #         self.POIs.append(info)
    #     elif is_fire and not is_poi:
    #         self.sensor_readings[(info, True, False, False, False)] = -10.0
    #         self.fires.append(info)
                   
    
    ##### does not take into account the order of the info collection (WORKS)
    def step(self, action, option):
        position, info_collectedX, info_collectedY, info_collectedZ, victim_saved = self.current_state  # Unpack the current state        
        option_terminated = False
        fell_into_ditch = False
        next_position = position
    
        if option == self.optionspace[0]:  ### EXPLORE
            if action in self.actionspace[self.optionspace[0]]:   ### ['up', 'down', 'left', 'right']
                self.totalTurns += 1
                # Movement actions update
                if action == self.actionspace[self.optionspace[0]][0] and position[0] > 0:  # up
                    next_position = (position[0] - 1, position[1])
                elif action == self.actionspace[self.optionspace[0]][1] and position[0] < self.gridsize[0] - 1:  # down
                    next_position = (position[0] + 1, position[1])
                elif action == self.actionspace[self.optionspace[0]][2] and position[1] > 0:  # left
                    next_position = (position[0], position[1] - 1)
                elif action == self.actionspace[self.optionspace[0]][3] and position[1] < self.gridsize[1] - 1:  # right
                    next_position = (position[0], position[1] + 1)
            # Check if information is collected or if the agent fell into a ditch
            if (next_position == self.infoLocation[0] and not info_collectedX) or \
                (next_position == self.infoLocation[1] and not info_collectedY) or \
                (next_position == self.infoLocation[2] and not info_collectedZ):
                    option_terminated = True
            
            if next_position == self.finalState[0] and info_collectedX and not victim_saved:
                option_terminated = True
            
            if self.totalTurns > self.maxSteps or next_position in self.ditches:
                option_terminated = True
                fell_into_ditch = next_position in self.ditches  # Agent fell into a ditch if true

        elif option == self.optionspace[1]:  ### COLLECT
            # Ask actions update
            if action in self.actionspace[self.optionspace[1]]:  ### ['get X', 'get Y', 'get Z']
                # Ask actions update
                self.totalAsks += 1
                info_type = action.split(' ')[1]  # Extract the type of information being collected ('X', 'Y', 'Z')
                if position == self.infoLocation[0] and info_type == 'X':
                    self.ask_action(self.current_state)
                    info_collectedX = True
                    self.collected_info.append(info_type)
                    option_terminated = True
                elif position == self.infoLocation[1] and info_type == 'Y':
                    info_collectedY = True
                    self.collected_info.append(info_type)
                    option_terminated = True
                elif position == self.infoLocation[2] and info_type == 'Z':
                    info_collectedZ = True
                    self.collected_info.append(info_type)
                    option_terminated = True
                else:
                    # If the agent is not at the correct location or the info is out of sequence,
                    # the option is terminated.
                    option_terminated = True
                    
        elif option == self.optionspace[2]:   ### SAVE 
            if action in self.actionspace[self.optionspace[2]]: ### ['save']
                if position == self.finalState[0] and info_collectedX:
                    # Assuming saving is only contingent on collecting information X for simplicity
                    victim_saved = True
                    option_terminated = True
                else:
                    #reward = self.wrongCollectPenalty  # Or another penalty as appropriate
                    option_terminated = True
        
        
        # Check if all information has been collected in the correct sequence
        all_info_collected_in_correct_order = self.collected_info == self.correct_sequence
        #done = next_position == self.finalState[0] and all_info_collected_in_correct_order
        done = position == self.finalState[0] and victim_saved and all_info_collected_in_correct_order
        reward = self.calculate_reward(position, next_position, done, option, fell_into_ditch)
        ditch_event = fell_into_ditch  # Set ditch_event based on whether the agent fell into a ditch
        # Reset the agent's position to the start state if it fell into a ditch
        if fell_into_ditch:
            next_position = self.startState[0]  ## an xtupisw empodio to option kanei terminate alla oxi to epeisodio
        
        self.current_state = (next_position, info_collectedX, info_collectedY, info_collectedZ, victim_saved)
        return self.current_state, reward, done, option_terminated, ditch_event

    def calculate_reward(self, position, next_position, done, current_option, fell_into_ditch):
        if fell_into_ditch:
            return self.ditchPenalty
        elif done:
            return self.winReward
        elif self.totalTurns > self.maxSteps:
            return self.exceedStepPenalty
        elif current_option == self.optionspace[1] and not self.is_sequence_correct():
            return self.wrongCollectPenalty  # Assume you have defined this penalty value in the initializer
        elif current_option == self.optionspace[2] and not self.is_sequence_correct():
            return self.wrongCollectPenalty
        else:
            return self.turnPenalty
    
    def is_sequence_correct(self):
        return self.collected_info == self.correct_sequence[:len(self.collected_info)]

    
    # def calculate_discrepancy(self, collected_info):
    #     # Initialize the discrepancy score
    #     discrepancy_score = 0
    #     # Create a mapping of element to its index in the optimal_sequence for order comparison
    #     optimal_indexes = {element: index for index, element in enumerate(list(self.ListB.keys()))}
    #     # Check for missing elements and add their weights to the discrepancy score
    #     for element in list(self.ListB.keys()):
    #         if element not in collected_info:
    #             discrepancy_score += self.ListB[element]
    #     # Check the order for elements in the estimated_sequence
    #     last_index = -1  # Initialize with an index that's before the first element's index
    #     for element in collected_info:
    #         if element in optimal_indexes:
    #             current_index = optimal_indexes[element]
    #             # If the current element is out of order, add its weight to the discrepancy score
    #             if current_index < last_index:
    #                 discrepancy_score += self.ListB[element]
    #             else:
    #                 # Update last_index to the current element's index if it's in correct order
    #                 last_index = current_index
    #     #print(discrepancy_score)
    #     return discrepancy_score
    
    

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
        fig, ax = plt.subplots(figsize=(8, 8))
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
                elif cell_coord in self.finalState:
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
        #agent_pos, _, _ = self.currentState
        agent_pos = self.position
        ax.plot(agent_pos[1] + 0.5, agent_pos[0] + 0.5, 'o', color='blue', markersize=10)  # Agent as a blue dot
        # Plotting the ditches
        for ditch in self.ditches:
            ax.plot(ditch[1] + 0.5, ditch[0] + 0.5, 'x', color='red', markersize=10)
        # Plotting the terminal state (victim's location)
        for terminal in self.finalState:
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
#          (5, 0), (5, 1), (5, 2), (6, 0), (0, 2), (0, 4), (5, 5)]

# #ditches = []
# fires = []
# POIs = []  # Victim locations
# #infoLocation = [(6, 1), (4, 4), (2, 6)]  # Location to ask for information was (6, 1)
# #infoLocation = {(6, 1): 'X', (4, 4): 'Y', (2, 6): 'Z'}
# infoLocation = [(6, 1), (4, 4), (2, 6)]
# fires = []
# POIs = []

# image_path = "/home/dimiubuntu/Desktop/local_code_scripts/enhanced_RL/enhanced_RL/images/disaster_area.jpg"
# document_path = "/home/dimiubuntu/Desktop/local_code_scripts/enhanced_RL/enhanced_RL/data/sar_data.json"


# env = SARenvHRL(gridsize, startState, victimStates, ditches, fires, POIs, infoLocation, image_path, document_path, mode='debug')