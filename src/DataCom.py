import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
import matlab
import matlab.engine
import numpy as np


def grain_compare(file1, file2):
    # Read data from CSV files
    data1 = pd.read_csv(file1)
    data2 = pd.read_csv(file2)

    # Assuming the data columns are named 'Value'. Adjust this as per your CSV file's column names.
    data_type = ['grain_diameter', 'Ellipse_LongAxis', 'Aspect_ratio', 'Ellipse_Angle']
    wd = []
    for i in data_type:
        values1 = data1[i]
        values2 = data2[i]

        # Calculate the Wasserstein distance between the two datasets
        distance = wasserstein_distance(values1, values2)
        wd = [wd, distance]
        print(f"Wasserstein Distance of {i}: {distance}")

        # Determine the range for the histograms
        combined_values = pd.concat([values1, values2])
        min_val = combined_values.min()
        max_val = combined_values.max()
        bins = np.linspace(min_val, max_val, 11)  # Create 20 bins between the min and max

        # Plotting the distribution of the data in a bar chart
        plt.figure(figsize=(12, 6))

        # Create histograms for both datasets
        plt.hist(values1, bins=bins, alpha=0.5, label=file1[10:14], edgecolor='black')
        plt.hist(values2, bins=bins, alpha=0.5, label=file2[10:14], edgecolor='black')

        # Adding titles and labels
        plt.title(i)
        plt.xlabel(i)
        plt.ylabel('Counts')
        plt.legend(loc='upper right')

        plt.savefig(file1[:-4] + i + '_com.jpg', format='jpg')

    return wd


def misorientation(file1, file2):
    # analyze the .ang file including EBSD-IPF map, grain analysis
    try:
        # Start MATLAB engine
        eng = matlab.engine.start_matlab()

        # Add path
        eng.addpath(r'MisOri.m')

        # Call the MATLAB function
        # pls making sure that the data has been de-normalized
        print("Comparing " + file1 + " and " + file2 + "...")
        eng.MisOri(file1, file2, nargout=0)
        print("Done.")

    except matlab.engine.MatlabExecutionError as e:
        print("Error in MATLAB execution:", e)

    finally:
        # Stop the MATLAB engine, ensuring it closes properly
        if 'eng' in locals():
            eng.quit()


if __name__ == "__main__":
    file_path1 = 'MicAnaTmp/true_z24_grain.csv'
    file_path2 = 'MicAnaTmp/true_z23_grain.csv'
    grain_compare(file_path1, file_path2)
    misorientation(file_path1[:-10] + '.ang', file_path2[:-10] + '.ang')
