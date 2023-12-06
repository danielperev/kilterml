import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import random

class Placements:
    def __init__(self, filepath:str):
        self.file_path = filepath
        self.all_data = {}
        self.all_placements = {}
        self.all_matrix_climbs = {}
        self.all_impossible_placements = {}
        self.DEFAULT_MATRIX_SIZE=(36,35)
        self.DEFAULT_MATRIX={}
        self.setup_placements()
        self.setup_default_matrix()
        self.setup_matrices()
        
        self._default_board_cmap = ListedColormap(["#2e2d2e", "#2e2d2e", "#00dd00", "#00ffff", "#ffa600", "#ff00ff"])
        self._empty_board_cmap = ListedColormap(["#ff0000", "#26ff00"])

    def clean_hold(self, hold: dict) -> dict:
        if "ledPosition" in hold:
                del hold["ledPosition"]

        match hold["type"]:
            case "START": #green
                hold["value"] = 1
            case "MIDDLE": #blue
                hold["value"] = 2
            case "FEET-ONLY": #orange
                hold["value"] = 3
            case "FINISH": #pink
                hold["value"] = 4
        del hold["type"]

        hold['x'] = max(hold['x']-1,0)
        return hold
    
    def setup_placements(self):
        with open(self.file_path / "all_data.json", "r", encoding="utf8") as all_data_file:
            self.all_data = json.load(all_data_file)

        with open(self.file_path / "impossible_holds.json", "r", encoding="utf8") as impossible_holds_file:
            self.all_impossible_placements = json.load(impossible_holds_file)["placements"]

        for climb in self.all_data:
            self.all_placements[climb["name"]] = [self.clean_hold(hold) for hold in climb["placements"]]

    def setup_default_matrix(self):
        matrix = np.zeros(self.DEFAULT_MATRIX_SIZE, dtype=np.uint8)
        matrix = np.hstack((matrix,np.full((matrix.shape[0],1), -1)))
        for coords in self.all_impossible_placements:
            matrix[coords["y"]][coords["x"]] = -1
        self.DEFAULT_MATRIX=matrix
        
    def setup_matrices(self):
        for name,placements in self.all_placements.items():
            self.all_matrix_climbs[name] = self.placements_to_matrix(placements)

    def placements_to_matrix(self, placements: list[dict]) -> list[list]:
        climb_matrix = self.DEFAULT_MATRIX.copy()

        for hold in placements:
            climb_matrix[hold['y']][hold['x']] = hold['value']
        
        return climb_matrix

    def all_placements_one_board(self, placements_dict: dict[str, list[dict]], matrix: list[list]) -> list[list]:
        for _,placements in placements_dict.items():
            for hold in placements:
                if matrix[hold['y']][hold['x']] == 0:
                    matrix[hold['y']][hold['x']] = hold['value']
        
        return matrix

    def sample_random_climbs(all_climbs: dict[str, list[list]], sample_number = 9):
        return {k:v for k,v in random.sample(list(all_climbs.items()), sample_number)}

    def plot_matrix_placements(self, matrix_placements: dict[str, list[list]], plot_size = 3):
        fig, axes = plt.subplots(plot_size, plot_size, figsize=(36,36))
        if plot_size > 1:
            for i,ax in enumerate(axes.flat):
                ax.tick_params(axis='both', which='major', labelsize=7)
                ax.set_title([key for index,key in enumerate(matrix_placements.keys()) if index == i][0], fontsize=9)
                ax.matshow([value for index,value in enumerate(matrix_placements.values()) if index == i][0], cmap=self._default_board_cmap)
        else:
            axes.set_title(list(matrix_placements.keys())[0], fontsize=16)
            axes.matshow(list(matrix_placements.values())[0], cmap=self._default_board_cmap)
        fig.set_size_inches((10, 10))
        plt.subplots_adjust(hspace=0.25, wspace=0.0)
        plt.show()
#-----

# plot_matrix_placements(sample_random_climbs(all_matrix_climbs))
# plot_matrix_placements(all_placements_one_board({"all": all_placements}, DEFAULT_MATRIX.copy()), plot_size=1)
# plot_matrix_placements({"all": DEFAULT_MATRIX}, plot_size=1, color_map=empty_board_cmap)

