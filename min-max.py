

import pandas as pd

def min_max_normalization(input_file, output_file):
    """
    This function reads a CSV file containing input data, applies min-max normalization to the data, 
    and writes the normalized data to a new CSV file.
    """
    # Read the input file into a pandas DataFrame
    data = pd.read_csv("input.csv")
    
    # Find the minimum and maximum values of the data
    min_val = data.min()
    max_val = data.max()
    
    # Apply the min-max normalization formula to each column in the data
    normalized_data = (data - min_val) / (max_val - min_val)
    
    # Write the normalized data to the output file
    normalized_data.to_csv("output.csv", index=False)
    
    return normalized_data

# Example usage
#input_file = "input.csv"
#output_file = "output.csv"

normalized_data = min_max_normalization("input.csv", "output.csv")