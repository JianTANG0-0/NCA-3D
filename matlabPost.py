import matlab
import matlab.engine
import numpy as np
import os

def gen_ang_file(data, f_name, note):
    # generate ang file for microstructure analysis in MTEX

    try:
        # Start MATLAB engine
        eng = matlab.engine.start_matlab()

        # Add path
        eng.addpath(r'ang_gen.m')

        # Call the MATLAB function
        # pls making sure that the data has been de-normalized
        print("generating " + f_name + "...")
        data= matlab.double(data.tolist())
        eng.ang_gen(data, f_name, nargout=0)
        print(f_name + " is generated.")

    except matlab.engine.MatlabExecutionError as e:
        print("Error in MATLAB execution:", e)

    finally:
        # Stop the MATLAB engine, ensuring it closes properly
        if 'eng' in locals():
            eng.quit()

    return None


def ana_ang_file(f_name):
    # analyze the .ang file including EBSD-IPF map, grain analysis
    try:
        # Start MATLAB engine
        eng = matlab.engine.start_matlab()

        # Add path
        eng.addpath(r'MTEX_ana.m')

        # Call the MATLAB function
        # pls making sure that the data has been de-normalized
        print("Analyzing " + f_name + "...")
        eng.MTEX_ana(f_name, nargout=0)
        print("Analysis of "+f_name+" is done.")

    except matlab.engine.MatlabExecutionError as e:
        print("Error in MATLAB execution:", e)

    finally:
        # Stop the MATLAB engine, ensuring it closes properly
        if 'eng' in locals():
            eng.quit()


def pos_encode(data, l):
    # Encode the position of the pixels
    height = data.shape[0]
    width = data.shape[1]

    # Generate x and y coordinate grids
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')

    # Normalize coordinates to be between 0 and 1 for consistency
    x_normalized = x * l
    y_normalized = y * l

    # Concatenate these grids to the last dimension
    # Reshape x and y to add a channel dimension for concatenation
    # Reshape x and y to add a channel dimension for concatenation
    x_channel = x_normalized.reshape(height, width, 1)
    y_channel = y_normalized.reshape(height, width, 1)

    # Concatenate along the last dimension
    data = np.concatenate((data, x_channel, y_channel), axis=-1)
    return data


def def_cross_section(data, l, d, s, note='true_'):
    # taking cross-section data and save it into .ang file for microstructure analysis
    # l: mesh sie; d: cross section dimensions; s: places to cut; note: 'true_' or 'pred_'

    directory_name = "MicAnaTmp"

    try:
        # Check if the directory exists
        if not os.path.exists(directory_name):
            # Create the directory
            os.makedirs(directory_name)
    except PermissionError:
        print(f"Permission denied: unable to create the directory '{directory_name}'.")
    except Exception as e:
        print(f"An error occurred: {e}")


    for i, (di, si) in enumerate(zip(d, s)):
        if di == 0:
            cs_data = data[si, ...]
            f_name = directory_name + '/' + note + 'x' + str(si) + '.ang'
        elif di == 1:
            cs_data = data[:, si, ...]
            f_name = directory_name + '/' + note + 'y' + str(si) + '.ang'
        elif di == 2:
            cs_data = data[:, :, si, ...]
            f_name = directory_name + '/' + note + 'z' + str(si) + '.ang'
        else:
            raise ValueError("Non-defined dimension for cross section")
        cs_data = pos_encode(cs_data, l)
        gen_ang_file(cs_data.reshape((-1, cs_data.shape[-1])), f_name, note)
        ana_ang_file(f_name)

if __name__ == "__main__":
    data = np.load('./LPBF_data_2.npy', allow_pickle=True)[1, -1, ...]
    data = np.concatenate([data[..., 2:5]/180.*np.pi, data[..., 1:2]],axis=-1)
    def_cross_section(data, 4, [2], [23])
