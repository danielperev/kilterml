import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import torch
import random
import math

class Placements:
    def __init__(self, filepath:str):
        self.file_path = filepath
        self.all_data = {}
        self.all_placements = {}
        self.all_matrix_climbs = {}
        self.all_impossible_placements = {}
        self.DEFAULT_MATRIX_SIZE=(36,35)
        self.DEFAULT_MATRIX={}
        self.read_data()
        self.setup_placements()
        self.setup_default_matrix()
        self.setup_matrices()
        
        self._default_board_cmap = ListedColormap(["#2e2d2e", "#00dd00", "#00ffff", "#ffa600", "#ff00ff"])
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
    
    def diffavg2vdiff(self,diff_avg: float) -> int:
        vdiff_dict = {0:0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 3, 7: 3, 8: 4, 9: 4,10: 5,11: 5,12: 6,13: 7,14: 8,15: 8,16: 9,17: 10,18: 11,19: 12,20: 13,21: 14,22: 15,23: 16}
        return vdiff_dict[math.floor(diff_avg - 10)]
    
    def clean_stats(self, stats: dict) -> dict:
        stats["difficulty"] = self.diffavg2vdiff(stats["difficulty_average"])
        del stats["difficulty_average"]
        return stats
    
    def read_data(self):
        with open(self.file_path / "all_data.json", "r", encoding="utf8") as all_data_file:
            self.all_data = json.load(all_data_file)

        with open(self.file_path / "impossible_holds.json", "r", encoding="utf8") as impossible_holds_file:
            self.all_impossible_placements = json.load(impossible_holds_file)["placements"]

    def setup_placements(self):
        for climb in self.all_data:
            self.all_placements[climb["uuid"]] = [climb["name"],
                                                  [self.clean_stats(stats) for stats in climb["climb_stats"]],
                                                  [self.clean_hold(hold) for hold in climb["placements"]]]

    def setup_default_matrix(self, with_impossible=False):
        matrix = np.zeros(self.DEFAULT_MATRIX_SIZE)
        if with_impossible:
            matrix = np.hstack((matrix,np.full((matrix.shape[0],1), -1)))
            for coords in self.all_impossible_placements:
                matrix[coords["y"]][coords["x"]] = 1
        self.DEFAULT_MATRIX=matrix
        
    def setup_matrices(self):
        for uuid,climb_elements in self.all_placements.items():
            climb_elements[-1] = self.placements_to_matrix(climb_elements[-1])
            self.all_matrix_climbs[uuid] = climb_elements

    def placements_to_matrix(self, placements: list[dict]) -> list[list]:
        climb_matrix = self.DEFAULT_MATRIX.copy()

        for hold in placements:
            climb_matrix[hold['y']][hold['x']] = hold['value']
        
        return climb_matrix
    
    def simple_climb_targets(self):
        matrix_climb_targets = []
        for values in self.all_matrix_climbs.values():
            stats_sorted_desc = sorted(values[1], key=lambda stat: stat["ascensionist_count"], reverse=True)
            climb_difficulty = stats_sorted_desc[0]["difficulty"]
            if (climb_difficulty < 10):
                matrix_climb_targets.append((torch.tensor(values[-1]), climb_difficulty))
        return matrix_climb_targets

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
# if __name__ == "__main__":
#     from pathlib import Path
#     root_dir = Path(__file__).resolve().parent.parent.parent
#     data_folder = root_dir / "data"

#     placements = Placements(data_folder)

    # print(len([climb['uuid'] for climb in placements.all_data]))
    # print(len(placements.all_placements))
    # print(placements.all_matrix_climbs["f01419e1-2672-4593-96ca-62e3655abc46"])
    # print(placements.simple_climb_targets()[0][1])
    # print(placements)
    # print(placements.simple_climb_targets()[0][0].shape)
