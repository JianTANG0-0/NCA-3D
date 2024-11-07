import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torchvision.transforms as T
import numpy as np
from LoadFunc_train import *
from Loadmodel_T_train import *
from mgpu_ppre_t_grad import red_t, cal_acc, prepare, cal_misori
import time
import grain_ana

def add_frame(x, y, vid, vid_hc, ti):
    x = x.permute(0, 3, 4, 2, 1).cpu().detach().numpy()
    x = np.concatenate([np.clip(np.argmax(x[..., :36], axis=-1, keepdims=True) / 36., 0.0, 1.0),
                              np.clip(np.argmax(x[..., 36:54], axis=-1, keepdims=True) / 18., 0.0, 1.0) / 2.0,
                              np.clip(np.argmax(x[..., 54:90], axis=-1, keepdims=True) / 36., 0.0, 1.0),
                              x[..., 90:]], axis=-1)

    y = y.permute(0, 1, 4, 5, 3, 2).cpu().detach().numpy()
    ea_i = 0
    x_dim = x.shape[1]
    cmap = plt.get_cmap("turbo")  # define a colormap

    for xi in [x_dim//4, x_dim//2, x_dim*3//4]:

        sam_i = x[ea_i, :, xi, ..., :3]
        sam_i = np.concatenate([sam_i, sam_i[...,:1, :,:]*0.0], axis=-3)
        target_i = y[ea_i, ti, :, xi, ...,:3]/36.
        target_i = np.concatenate([target_i, target_i[..., :1, :, :] * 0.0], axis=-3)
        zmis_ori = cal_misori(np.clip(sam_i, 0.0, 1.0) * np.pi * 2.0,
                             target_i * np.pi * 2.0)  # misorientation angle
        zfil = (zmis_ori > 15.0) & (zmis_ori < 75.0)
        zdif_img = cmap(zmis_ori * zfil / 90.0)[..., :3]

        hid_cha = x[ea_i, :, xi, ..., -3:]
        hid_cha = np.concatenate([hid_cha, hid_cha[..., :1, :, :] * 0.0], axis=-3)
        hid_cha = (hid_cha-np.min(hid_cha))/(np.max(hid_cha)-np.min(hid_cha))
        if xi == x_dim//4:
            show_img_hid = np.rot90(hid_cha, 1, (0,1))
            #np.vstack((cmap(hid_cha[..., 0])[..., :3],cmap(hid_cha[..., 1])[..., :3], cmap(hid_cha[..., 2])[..., :3])), 1, (0,1))#
            show_img = np.rot90(np.vstack((sam_i, target_i, zdif_img)), 1, (0,1))
        else:
            show_img = np.vstack((show_img, np.rot90(np.vstack((sam_i, target_i, zdif_img)), 1, (0,1))))
            show_img_hid = np.hstack((show_img_hid, np.rot90(hid_cha, 1, (0,1))))
            #np.rot90(np.vstack((cmap(hid_cha[..., 0])[..., :3],cmap(hid_cha[..., 1])[..., :3], cmap(hid_cha[..., 2])[..., :3])), 1, (0,1))))#

    plt.imshow(zoom(show_img))
    plt.savefig(f'com{ti+1}.jpg')
    vid.add(zoom(show_img))
    plt.imshow(zoom(show_img_hid))
    plt.savefig(f'hid{ti + 1}.jpg')
    vid_hc.add(zoom(show_img_hid))

def cal_vol_acc(x, y, data_no):
    x = x.permute(0, 3, 4, 2, 1).cpu().detach().numpy()
    x = np.concatenate([np.argmax(x[..., :36], axis=-1, keepdims=True),
                        np.argmax(x[..., 36:54], axis=-1, keepdims=True),
                        np.argmax(x[..., 54:90], axis=-1, keepdims=True),
                        x[..., 90:]], axis=-1)

    y = y.permute(0, 1, 4, 5, 3, 2).cpu().detach().numpy()
    acc = np.mean(np.sum(x[..., :3] != y[:, -1, ..., :3], axis=-1) == 0) * 100
    d = [2]#[0, 0, 0, 2, 2, 2, 1, 1, 1]
    s = [28]#[8, 16, 24, 4, 16, 28, 8, 16, 24]
    test_compare(y, x, d, s, data_no)

    return acc

def para_ang(data, di, si, note='true_'):
    if di == 0:
        f_name = 'MicAnaTmp/' + note + 'x' + str(si) + '.ang'
    elif di == 1:
        f_name = 'MicAnaTmp/' + note + 'y' + str(si) + '.ang'
    elif di == 2:
        f_name = 'MicAnaTmp/' + note + 'z' + str(si) + '.ang'
    else:
        raise ValueError("Non-defined dimension for cross section")

    from matlabPost import def_cross_section
    def_cross_section(data, 5, [di], [si], note)
    return f_name


def test_compare(data_true, data_pred, d, s, data_no):
    data_true = np.concatenate([data_true[0, -1, ..., :1]/36.*np.pi*2.0,
                                data_true[0, -1, ..., 1:2]/18.*np.pi,
                                data_true[0, -1, ..., 2:3]/36.*np.pi*2.,
                                data_true[0, -1, ..., 3:4]],axis=-1)
    data_pred = np.concatenate([data_pred[0, ..., :1] / 36. * np.pi * 2.0,
                                data_pred[0, ..., 1:2] / 18. * np.pi,
                                data_pred[0, ..., 2:3] / 36. * np.pi * 2.,
                                data_pred[0, ..., 3:4]], axis=-1)

    #grain_ana.grain_ana_sub(data_true[..., :3], data_true[..., 3:4], 5, 'ttrue_'+ str(data_no))

    #grain_ana.grain_ana_sub(data_pred[..., :3], data_pred[..., 3:4], 5, 'ppred_'+ str(data_no))
    from DataCom import grain_compare, misorientation

    wd_list = []

    for i, (di, si) in enumerate(zip(d, s)):
        fname1 = para_ang(data_true, di, si, note='true_'+str(data_no))
        fname2 = para_ang(data_pred, di, si, note='pred_'+str(data_no))
        with open("MicAnaTmp/wd.txt", "w") as outfile:
            wd = grain_compare(fname1[:-4]+'_grain.csv', fname2[:-4]+'_grain.csv')
            wd = [di, si, wd]
            #wd_list = wd_list.append(wd)
            #outfile.write("\n".join(wd_list))
            #misorientation(fname1, fname2)

# function for run individual test run and  calculate average accuracy and rsme
def evaluate_model(ca, test_set, batch_size, num_t, device):
    NCA_comp_speed = []
    vol_acc = []
    acc = []
    l_time_sum = 0.0
    data_no = 0
    for j, (x_batch, xt_batch) in enumerate(test_set):
        x_batch, xt_batch = x_batch.to(device), xt_batch.to(device)
        if j == -1:
            with VideoWriter('ca_nca_compare.mp4') as vid:
                with VideoWriter('nca_hid_channel.mp4') as vid_hc:
                    add_frame(x_batch, x_batch[:,None,...], vid, vid_hc, 0)
                    for nca_step in range(num_t):
                        try:
                            start = time.time()
                            x_batch = ca(x_batch)
                            end = time.time()
                            NCA_comp_speed.append(end - start)
                        except RuntimeError as exception:
                            if "out of memory" in str(exception):
                                if hasattr(torch.cuda, 'empty_cache'):
                                    torch.cuda.empty_cache()
                            else:
                                raise exception
                        add_frame(x_batch, xt_batch, vid, vid_hc, nca_step)
                        l_time_sum += weigt_loss(x_batch, xt_batch[:, nca_step]).item()/batch_size
                        if nca_step < num_t:
                            x_batch[:, 91:92, ...] = xt_batch[:, nca_step, 4:5].type(torch.FloatTensor)
                        if nca_step < num_t - 1:
                            x_batch[:, 92:93, ...] = xt_batch[:, nca_step + 1, 4:5].type(torch.FloatTensor)
        else:
            for nca_step in range(num_t):
                try:
                    start = time.time()
                    x_batch = ca(x_batch)
                    end = time.time()
                    NCA_comp_speed.append(end - start)
                except RuntimeError as exception:
                    if "out of memory" in str(exception):
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                    else:
                        raise exception
                l_time_sum += weigt_loss(x_batch, xt_batch[:, nca_step]).item() / batch_size
                if nca_step < num_t:
                    x_batch[:, 91:92, ...] = xt_batch[:, nca_step, 4:5].type(torch.FloatTensor)
                if nca_step < num_t - 1:
                    x_batch[:, 92:93, ...] = xt_batch[:, nca_step + 1, 4:5].type(torch.FloatTensor)
        data_no += 1
        acc.append(cal_acc(x_batch, xt_batch[:, nca_step]))
        vol_acc.append(cal_vol_acc(x_batch, xt_batch, data_no))

    l_time_sum /= 200.
    test_acc = np.mean(np.array(acc))
    test_vol_acc = np.mean(np.array(vol_acc))
    print("test loss: ", l_time_sum,
          "test acc: ", test_acc, "%",
          "test acc: ", test_vol_acc, "%")
    np.savetxt('./runtime.csv', np.array(NCA_comp_speed), delimiter=',')


def view_test(x,y):
    x = x.permute(0, 3, 4, 2, 1).cpu().detach().numpy()
    x = np.concatenate([np.argmax(x[..., :36], axis=-1, keepdims=True),
                        np.argmax(x[..., 36:54], axis=-1, keepdims=True),
                        np.argmax(x[..., 54:90], axis=-1, keepdims=True),
                        x[..., 90:]], axis=-1)

    y = y.permute(0, 3, 4, 2, 1).cpu().detach().numpy()
    plt.imshow(x[0, ..., 16, :, :3] / 36.)
    plt.show()
    plt.imshow(y[0, ..., 16, :, :3] / 36.)
    plt.show()
    acc = np.mean(np.sum(x[..., :3] != y[:, ..., :3], axis=-1) == 0) * 100
    print(f'test{acc}')

if __name__ == '__main__':
    rand_seed = 3024
    random.seed(rand_seed)
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed_all(rand_seed)

    # load the CRNN Model from file
    model_file = './step1/'
    ca = load_model(model_file)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ca = ca.to(device)
    ca.eval()

    nca_train_data = np.load('../dirsoild_4.npy', allow_pickle=True)[:, ::2, ..., :-1]
    nca_train_data = torch.from_numpy(nca_train_data)
    nca_train_data = nca_train_data.permute([0, 1, 5, 4, 2, 3]).type(torch.FloatTensor)
    batch_size = 1
    num_t = 16
    ini_t = [0]
    rank = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
    world_size = 1
    sp_rate = [1]
    CHANNEL_N = 10
    _, _, x_train, y_train= red_t(ini_t, num_t, sp_rate, CHANNEL_N, 0, nca_train_data)
    dataset = torch.utils.data.TensorDataset(x_train, y_train)
    test_set = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    evaluate_model(ca, test_set, batch_size, num_t, device)

