import numpy as np
from sklearn.decomposition import PCA
import csv


def mark_grain(ori: np.array, phase: np.array):
    nx, ny, nz, _ = ori.shape
    visited = np.zeros((nx, ny, nz), dtype=int)
    grains = []
    grain_no = np.zeros((nx, ny, nz), dtype=int)
    grain_count = 0

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if visited[i, j, k] == 0 and phase[i, j, k]:
                    grains.append([])
                    queue = [(i, j, k)]
                    visited[i, j, k] = 1
                    grain_count += 1
                    grain_no[i, j, k] = grain_count
                    while queue:
                        s = queue.pop(0)
                        cur_x, cur_y, cur_z = s
                        grains[-1].append([cur_x, cur_y, cur_z])
                        for xi in range(-1, 2):
                            for yi in range(-1, 2):
                                for zi in range(-1, 2):
                                    if (cur_x + xi == nx) or (cur_y + yi == ny) or (cur_z + zi == nz) or \
                                            (cur_x + xi == -1) or (cur_y + yi == -1) or (cur_z + zi == -1):
                                        continue
                                    if (visited[cur_x + xi, cur_y + yi, cur_z + zi] == 0 and
                                            (cal_misorientation(ori[cur_x + xi, cur_y + yi, cur_z + zi, :],
                                                                ori[cur_x, cur_y, cur_z, :]) <= 15.) and
                                            phase[cur_x + xi, cur_y + yi, cur_z + zi, 0] and (
                                                    abs(xi) + abs(yi) + abs(zi) != 0)):
                                        queue.append((cur_x + xi, cur_y + yi, cur_z + zi))
                                        visited[cur_x + xi, cur_y + yi, cur_z + zi] = 1
                                        grain_no[cur_x + xi, cur_y + yi, cur_z + zi] = grain_count

    return grain_no, grains


def cal_misorientation(ori1: int, ori2: int) -> float:
    pred = np.array(ori1)[None, None, :]
    true = np.array(ori2)[None, None, :]

    p1 = pred[:, :, 0]
    p = pred[:, :, 1]
    p2 = pred[:, :, 2]
    q1 = true[:, :, 0]
    q = true[:, :, 1]
    q2 = true[:, :, 2]

    nx = p.shape[0]
    ny = p.shape[1]

    t1 = np.zeros((nx, ny, 24))
    t2 = np.zeros((nx, ny, 24))
    t3 = np.zeros((nx, ny, 24))
    theta = np.zeros((nx, ny, 24))
    g1 = np.zeros((nx, ny, 3, 3))
    g2 = np.zeros((nx, ny, 3, 3))
    gp = np.zeros((nx, ny, 3, 3))
    gp1 = np.zeros((nx, ny, 3, 3))
    gp2 = np.zeros((nx, ny, 3, 3))
    gq = np.zeros((nx, ny, 3, 3))
    gq1 = np.zeros((nx, ny, 3, 3))
    gq2 = np.zeros((nx, ny, 3, 3))
    m = np.zeros((nx, ny, 24, 3, 3))

    # converting in the form of matrices for both grains
    gp1[:, :, 0, 0] = np.cos(p1)
    gp1[:, :, 1, 0] = -np.sin(p1)
    gp1[:, :, 0, 1] = np.sin(p1)
    gp1[:, :, 1, 1] = np.cos(p1)
    gp1[:, :, 2, 2] = 1
    gp2[:, :, 0, 0] = np.cos(p2)
    gp2[:, :, 1, 0] = -np.sin(p2)
    gp2[:, :, 0, 1] = np.sin(p2)
    gp2[:, :, 1, 1] = np.cos(p2)
    gp2[:, :, 2, 2] = 1
    gp[:, :, 0, 0] = 1
    gp[:, :, 1, 1] = np.cos(p)
    gp[:, :, 1, 2] = np.sin(p)
    gp[:, :, 2, 1] = -np.sin(p)
    gp[:, :, 2, 2] = np.cos(p)
    gq1[:, :, 0, 0] = np.cos(q1)
    gq1[:, :, 1, 0] = -np.sin(q1)
    gq1[:, :, 0, 1] = np.sin(q1)
    gq1[:, :, 1, 1] = np.cos(q1)
    gq1[:, :, 2, 2] = 1
    gq2[:, :, 0, 0] = np.cos(q2)
    gq2[:, :, 1, 0] = -np.sin(q2)
    gq2[:, :, 0, 1] = np.sin(q2)
    gq2[:, :, 1, 1] = np.cos(q2)
    gq2[:, :, 2, 2] = 1
    gq[:, :, 0, 0] = 1
    gq[:, :, 1, 1] = np.cos(q)
    gq[:, :, 1, 2] = np.sin(q)
    gq[:, :, 2, 1] = -np.sin(q)
    gq[:, :, 2, 2] = np.cos(q)
    g1 = np.matmul(np.matmul(gp2, gp), gp1)
    g2 = np.matmul(np.matmul(gq2, gq), gq1)

    # symmetry matrices considering the 24 symmetries for cubic system
    T = np.zeros((24, 3, 3))
    T[0, :, :] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    T[1, :, :] = [[0, 0, -1], [0, -1, 0], [-1, 0, 0]]
    T[2, :, :] = [[0, 0, -1], [0, 1, 0], [1, 0, 0]]
    T[3, :, :] = [[-1, 0, 0], [0, 1, 0], [0, 0, -1]]
    T[4, :, :] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
    T[5, :, :] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
    T[6, :, :] = [[1, 0, 0], [0, -1, 0], [0, 0, -1]]
    T[7, :, :] = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
    T[8, :, :] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    T[9, :, :] = [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]
    T[10, :, :] = [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]
    T[11, :, :] = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
    T[12, :, :] = [[0, 1, 0], [0, 0, 1], [1, 0, 0]]
    T[13, :, :] = [[0, 0, -1], [-1, 0, 0], [0, 1, 0]]
    T[14, :, :] = [[0, -1, 0], [0, 0, 1], [-1, 0, 0]]
    T[15, :, :] = [[0, 1, 0], [0, 0, -1], [-1, 0, 0]]
    T[16, :, :] = [[0, 0, -1], [1, 0, 0], [0, -1, 0]]
    T[17, :, :] = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]
    T[18, :, :] = [[0, -1, 0], [0, 0, -1], [1, 0, 0]]
    T[19, :, :] = [[0, 1, 0], [1, 0, 0], [0, 0, -1]]
    T[20, :, :] = [[-1, 0, 0], [0, 0, 1], [0, 1, 0]]
    T[21, :, :] = [[0, 0, 1], [0, -1, 0], [1, 0, 0]]
    T[22, :, :] = [[0, -1, 0], [-1, 0, 0], [0, 0, -1]]
    T[23, :, :] = [[-1, 0, 0], [0, 0, -1], [0, -1, 0]]

    T = np.array(T[None, None, ...])

    # finding the 24 misorientation matrices(also can be calculated for 576 matrices)
    for i in range(24):
        m[:, :, i, :, :] = np.matmul(np.linalg.inv(np.matmul(T[:, :, i, :, :], g1)), g2)
        t1[:, :, i] = m[:, :, i, 0, 0]
        t2[:, :, i] = m[:, :, i, 1, 1]
        t3[:, :, i] = m[:, :, i, 2, 2]
        theta[:, :, i] = np.arccos(0.5 * (t1[:, :, i] + t2[:, :, i] + t3[:, :, i] - 1))

    # minimum of 24 angles is taken as misorientation angle
    ansRad = np.nanmin(theta, axis=-1)
    ansTheta = ansRad * 180.0 / np.pi
    return ansTheta


def grain_analysis(pix_in_grains: list, cell_len: float) -> list:
    pca = PCA(n_components=3)

    grain_sum_vols = []
    grain_sum_aspect_ratios = []
    grain_center_cor = []
    grain_spatial_orientation = []
    grain_ellipsoid_axis = []

    for i in range(len(pix_in_grains)):
        pix_grain_i = pix_in_grains[i]
        pix_num = len(pix_grain_i)
        sum_vol = pix_num * cell_len ** 3

        if pix_num < 4:
            continue  # grain_sum_aspect_ratios.append(1.)
        else:
            scaled_pix_cor = np.array(pix_grain_i) * cell_len
            center_cor = np.mean(scaled_pix_cor, axis=0)[None, :]
            grain_center_cor.append(center_cor)
            pca.fit(scaled_pix_cor - center_cor)
            grain_spatial_orientation.append(pca.components_.T)
            ev = pca.explained_variance_
            lengths = np.sqrt(ev)
            aspect_ratio = 2 * lengths[0] / (lengths[1] + lengths[2])
            grain_sum_aspect_ratios.append(aspect_ratio)
            grain_ellipsoid_axis.append([lengths[0], lengths[1], lengths[2]])
        grain_sum_vols.append(sum_vol)

    return np.array(grain_sum_vols), np.array(grain_sum_aspect_ratios), \
           np.array(grain_center_cor), np.array(grain_spatial_orientation), \
           np.array(grain_ellipsoid_axis)

def grain_ana_sub(ori: np.array, phase: np.array, cell_len: float, note: str):
    grain_no, grains = mark_grain(ori, phase)
    grain_sum_vols, grain_sum_aspect_ratios, grain_center_cor, \
    grain_spatial_orientation, grain_ellipsoid_axis = grain_analysis(grains, cell_len)
    f_name = 'MicAnaTmp/' + note + '.csv'
    data = np.concatenate([grain_sum_vols[:, None], grain_sum_aspect_ratios[:, None], grain_ellipsoid_axis,
                     np.squeeze(grain_center_cor),grain_spatial_orientation.reshape(-1, 9)], axis=1)
    title =["grain_vol", "grain_aspect_ratio", "grain_ellip_axis1",  "grain_ellip_axis2", \
           "grain_ellip_axis3", "grain_cen1", "grain_cen2", "grain_cen3", "grain_dir1", \
           "grain_dir2", "grain_dir3", "grain_dir4", "grain_dir5", "grain_dir6", \
           "grain_dir7", "grain_dir8", "grain_dir9"]

    with open(f_name, mode="w", newline="") as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(title)
        # Write the data
        writer.writerows(data)


