import pandas as pd
from pathlib import Path

# Function to mark to mark cuts, location should be outputed by functions such as prune_dist()
    # This should be called after the ltr huristics cuts are marked becuase this fucntion will not
    # clear the csv file. It is called within the pruning function that will marking things.

def mark_cuts(location, type, radius, normal_vector, year, name):
    if location != None:    
        df = pd.read_csv(str(Path(__file__).parent.resolve() / "marked_locations.csv"))
        for location, type, radius, normal_vector, year, name in zip(location, type, radius, normal_vector, year, name):
            mask = (
                    (df['Loc_x'] == location[0]) &
                    (df['Loc_y'] == location[1]) &
                    (df['Loc_z'] == location[2]) &
                    (df['Type'] == type) 
                    )
            row_exists = mask.any()

            if not row_exists:
                new_row = pd.DataFrame([{
                                            'Loc_x':location[0], 
                                            'Loc_y':location[1], 
                                            'Loc_z':location[2],
                                            'Type':type,
                                            'Radius':radius,
                                            'Norm_x1':normal_vector[0][0],
                                            'Norm_y1':normal_vector[0][1],
                                            'Norm_z1':normal_vector[0][2],
                                            'Norm_x2':normal_vector[1][0],
                                            'Norm_y2':normal_vector[1][1],
                                            'Norm_z2':normal_vector[1][2],
                                            'Year':year,
                                            'Name':name

                                                }])
                
                df = pd.concat([new_row], ignore_index=False)
            else:
                print(f'[ERROR] Duplicate mark location')

            df.to_csv(Path(__file__).parent.resolve()/"marked_locations.csv", 
                    mode='a',
                    header=False,
                    index=False)
