import os
import numpy as np
from tqdm import tqdm

def read_dat_files(data_folder):
    all_data = []
    filenames = []
    for filename in os.listdir(data_folder):
        if filename.endswith('.dat'):
            file_path = os.path.join(data_folder, filename)
            data = np.loadtxt(file_path, delimiter=',')
            all_data.append(data)
            filenames.append(filename)
    return all_data, filenames


def process_data(data, time_steps, x_dim, y_dim, z_dim):
    Fai = [      0, 2.7266,    3.8942,    4.8190,    5.6251,    6.3612,    7.0529,    7.7160,
                 8.3621,    9.0000,    9.6379,   10.2840,   10.9471,   11.6388,   12.3749,   13.1810,   14.1058,   15.2734]
    value_to_class = {value: i for i, value in enumerate(Fai)}

    num_rows = data.shape[0]
    current_time_steps = num_rows // (x_dim * y_dim * z_dim)

    if current_time_steps < time_steps:
        reshaped_data = np.zeros((time_steps, x_dim, y_dim, z_dim, 6))
        for t in range(current_time_steps):
            start_index = t * x_dim * y_dim * z_dim
            end_index = (t + 1) * x_dim * y_dim * z_dim
            timestep_data = data[start_index:end_index, :]

            for row in timestep_data:
                x, y, z = int(row[0])-1, int(row[1])-1, int(row[2])-1
                row[4] = value_to_class[round(row[4],4)]
                reshaped_data[t, x, y, z, :] = row[3:9]

        # Repeat the last time step to extend to 22 time steps
        for t in range(current_time_steps, time_steps):
            reshaped_data[t] = reshaped_data[current_time_steps - 1]

    elif current_time_steps == time_steps:
        reshaped_data = np.zeros((time_steps, x_dim, y_dim, z_dim, 6))
        for t in range(current_time_steps):
            start_index = t * x_dim * y_dim * z_dim
            end_index = (t + 1) * x_dim * y_dim * z_dim
            timestep_data = data[start_index:end_index, :]

            for row in timestep_data:
                x, y, z = int(row[0])-1, int(row[1])-1, int(row[2])-1
                row[4] = value_to_class[round(row[4], 4)]
                reshaped_data[t, x, y, z, :] = row[3:9]

    else:
        return None  # Skip this data if time steps > 22

    return reshaped_data


def concatenate_and_save(processed_data_list, file_count):
    big_array = np.concatenate(processed_data_list, axis=0)
    big_array_c_order = np.ascontiguousarray(big_array)
    filename = f'dirsoild_{file_count}.npy'
    np.save(filename, big_array_c_order)


def main():
    data_folder = 'data'
    time_steps = 32
    x_dim, y_dim, z_dim = 32, 32, 64
    processed_data_list = []
    skipped_files = []
    file_counter = 0
    save_counter = 1

    all_files_data, filenames = read_dat_files(data_folder)

    for idx, data in enumerate(tqdm(all_files_data, desc="Processing data")):
        processed_data = process_data(data, time_steps, x_dim, y_dim, z_dim)
        if processed_data is None:
            skipped_files.append(filenames[idx])
        else:
            processed_data_list.append(processed_data[None, ...])
            file_counter += 1

        if file_counter == 200:
            save_counter += 1
            concatenate_and_save(processed_data_list, save_counter)
            processed_data_list = []
            file_counter = 0

    # Save any remaining data
    if processed_data_list:
        save_counter += 1
        concatenate_and_save(processed_data_list, save_counter)

    # Print skipped files
    if skipped_files:
        print("Skipped files due to more than 22 time steps:")
        for filename in skipped_files:
            print(filename)


if __name__ == "__main__":
    main()
