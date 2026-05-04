"""
Color management utilities for L-Py tree simulation system.

This module provides utilities for assigning unique colors to tree segments
for visualization and labeling purposes.
"""

import itertools
import json


class ColorManager:
    """Manages assignment of unique colors to named entities."""
    _component_colors={"trunk":(50, 50, 255), "branch":((50, 255, 50), (30, 200, 30), (25, 155, 25)), "spur":(255, 50, 50)}

    def __init__(self):
        self.color_to_name = {}
        self.name_to_color = {}
        # Permute all possible colors to make a list
        self.all_colors = list(itertools.product(range(256), repeat=3))  # 0-255 inclusive
        self.color_pointer = 0

    @staticmethod
    def component_color(name):
        if "trunk" in name:
            return ColorManager._component_colors["trunk"]
        elif "branch" in name:
            if "L" in name:
                indx = max(int(name[7]), 2)
                return ColorManager._component_colors["branch"][indx]
            else:
                return ColorManager._component_colors["branch"][0]
        elif "spur" in name:
            return ColorManager._component_colors["spur"]
        return (255, 255, 255)


    def get_unique_color(self, name, if_exists=True):
        """Get a unique RGB color tuple for the given name."""
        if name in self.name_to_color and if_exists:
            return self.name_to_color[name]

        if self.color_pointer >= len(self.all_colors):
            raise ValueError("Ran out of unique colors!")

        unique_color = self.all_colors[self.color_pointer]
        self.color_pointer += 1

        # Assign and save mapping
        self.name_to_color[name] = unique_color
        self.color_to_name[unique_color] = name

        return unique_color
    
    def remove_name(self, name):
        """Remove a name and its associated color from the manager."""
        if name in self.name_to_color:
            print(f"Removing color mapping for name: {name}")
            color = self.name_to_color[name]
            del self.name_to_color[name]
            if color in self.color_to_name:
                del self.color_to_name[color]

    def export_mapping_dict(self) -> str:
        """Export color -> name mapping as JSON string."""
        export_dict = {}
        for color, name in self.color_to_name.items():
            export_dict[str(color)] = {"part_name": name.lower().strip()}
        return export_dict

    def export_mapping_json(self, filename: str) -> None:
        """Export color -> name mapping to JSON"""
        export_dict = {}
        for color, name in self.color_to_name.items():
            export_dict[str(color)] = {"part_name": name.lower().strip()}

        with open(filename, "w") as f:
            json.dump(export_dict, f, indent=4)

        print(f"Exported color mappings to {filename}")
        return

    def set_unique_color(self, color, name):
        """Set a specific color for a given name."""
        self.name_to_color[name] = color
        self.color_to_name[color] = name
        return
