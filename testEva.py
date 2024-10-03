#from DataCom import *
#from matlabPost import *
import matplotlib.pyplot as plt
import matplotlib
cmap = matplotlib.cm.get_cmap("turbo")  # define a colormap
from mgpu_ppre_t_grad import red_t
from mgpu_ppre_t_grad import cal_misori
from Loadmodel_T_train import *
from LoadFunc_train import *
import torch
import numpy as np
import time

def nca_pred(data, ini_t, parameterization):
    num_t = parameterization.get("tot_t", 16)
    channel = parameterization.get("in_dim", 10)
    sp_rate = parameterization.get("sp_rate", [1])

    data_pred = data_norm(data)
    _, _, x, y = red_t(ini_t, num_t, sp_rate, channel, 1, data_pred)

    ca = NCA(parameterization)
    ca.initialize_weights()

    ca = load_model('step1/')

    rank = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")

    ca = ca.to(rank)
    x, y = x.to(rank), y.to(rank)

    for nca_step in range(num_t):
        try:
            start_time = time.time()
            x = ca(x)
            #plot_pred(x, y[:, nca_step], nca_step, 18)
            # End the timer
            end_time = time.time()

            # Calculate the duration
            duration = end_time - start_time
            print(f"The code took {duration} seconds to execute.")
        except RuntimeError as exception:
            if "out of memory" in str(exception):
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            else:
                raise exception

        if nca_step < num_t:
            x[:, 91:92, ...] = y[:, nca_step, 4:5].type(torch.FloatTensor)
        if nca_step < num_t - 1:
            x[:, 92:93, ...] = y[:, nca_step + 1, 4:5].type(torch.FloatTensor)

    x = (x.permute(0, 3, 4, 2, 1)).detach().cpu().numpy()
    x = np.concatenate([np.argmax(x[..., :36], axis=-1, keepdims=True)/36.*2.0*np.pi,
                        np.argmax(x[..., 36:54], axis=-1, keepdims=True)/18.*np.pi,
                        np.argmax(x[..., 54:90], axis=-1, keepdims=True)/36.*2.0*np.pi,
                        x[..., 90:]],axis=-1)

    return x

def plot_pred(x, t, i, si):
    x = (x.permute(0, 3, 4, 2, 1)).detach().cpu().numpy()
    x = np.concatenate([np.argmax(x[..., :36], axis=-1, keepdims=True),
                        np.argmax(x[..., 36:54], axis=-1, keepdims=True),
                        np.argmax(x[..., 54:90], axis=-1, keepdims=True),
                        x[..., 90:]], axis=-1)

    x_true = (t.permute(0, 3, 4, 2, 1)).detach().cpu().numpy()
    x_true = np.concatenate([x_true[..., :1]/ 36.,x_true[..., 1:2]/ 18.,x_true[..., 2:3]/ 36.,
                        x_true[..., 3:]], axis=-1)

    # calculate the difference between NCA and CA
    ea_x = x[0, ..., si, :3]* np.pi
    ea_x[..., 0] = ea_x[..., 0] *2.
    ea_x[..., 2] = ea_x[..., 2] *2.
    ea_t = x_true[0, ..., si, :3]  * np.pi
    ea_t[..., 0] = ea_t[..., 0] *2.
    ea_t[..., 2] = ea_t[..., 2] *2.
    mis_ori = cal_misori(ea_x,ea_t)  # misorientation angle
    filter = (mis_ori > 15.0) & (mis_ori < 75.0)  # diff>10.0

    show_img = np.hstack((x[0,..., si, :3], x_true[0,..., si, :3],
                                  cmap(mis_ori * filter / 90.0)[..., :3]))
    plt.imshow(show_img)
    plt.savefig('./ea_com+t'+str(i)+'_si'+str(si)+'.jpg')

    show_img = (x[0, ..., si, -3:]-np.min(x[0, ..., si, -3:]))/\
               (np.max(x[0, ..., si, -3:])-np.min(x[0, ..., si, -3:])-1e-12)
    plt.imshow(show_img[...,-3:])
    plt.savefig('./hs_+t' + str(i) +'_si'+str(si)+'.jpg')

    cs = np.where(x[0, ..., si, 3]>0.99,1.,0.)
    show_img = (-cs + x_true[0, ..., si, 3]+1.0)/2.0
    plt.imshow(cmap(show_img)[..., :3])
    plt.savefig('./ps_+t' + str(i) +'_si'+str(si)+'.jpg')

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


def test_compare(data, d, s, ini_t, parameterization):
    data_true = np.concatenate([data[0, -1, ..., :1]/36.*np.pi*2.0, data[0, -1, ..., 1:2]/18.*np.pi, data[0, -1, ..., 2:3]/36.*np.pi*2., data[0, -1, ..., 3:4]],axis=-1)
    data_pred = nca_pred(data, ini_t, parameterization)[0, ..., :4]
    print(data_pred.shape)
    from DataCom import grain_compare, misorientation

    wd_list = []

    for i, (di, si) in enumerate(zip(d, s)):
        fname1 = para_ang(data_true, di, si, note='true_')
        fname2 = para_ang(data_pred, di, si, note='pred_')
        with open("MicAnaTmp/wd.txt", "w") as outfile:
            wd = grain_compare(fname1[:-4]+'_grain.csv', fname2[:-4]+'_grain.csv')
            wd = [di, si, wd]
            #wd_list = wd_list.append(wd)
            #outfile.write("\n".join(wd_list))
            misorientation(fname1, fname2)



def data_norm(nca_train_data):
    nca_train_data = torch.from_numpy(nca_train_data)
    nca_train_data = nca_train_data.permute([0, 1, 5, 4, 2, 3]).type(torch.FloatTensor)
    return nca_train_data


def load_model(model_file='./Setup_0'):
    model_para = np.load(model_file+'/model_setting.npy',allow_pickle=True).item()
    model_file = model_file + '/model.pkl'
    ca = NCA(model_para)
    ca.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
    print(model_file)
    #ca = torch.load(model_file, map_location='cpu')
    return ca


if __name__ == "__main__":
    parameters = {
        "hid_lay_num": [5],
        "kernel_size": [3],
        "neu_num": [256],
        "dic_neu_num": [64],
        "dic_lay_num": [0],
        "in_dim": [10],
        "tot_t": [16],
    }

    settings = list(itertools.product(*parameters.values()))
    i = 0
    folder_name = str(os.getcwd())
    for setup in settings:
        print("###################################")
        print('setup:  No.', i + 1)
        folder_path = folder_name + "/Setup_" + str(i)
        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)
        setup_properties = parameters
        j = 0
        for key in parameters:
            setup_properties[key] = setup[j]
            j = j + 1

    d = [0, 0, 0, 2,2,2, 1, 1, 1]
    s = [8, 16, 24, 4,16,28, 8, 16, 24]
    # read the data file
    for i in range(1,2):
        data = np.load('./dirsoild_4.npy', allow_pickle=True)[i:i+1, ::2, ..., :-1]
        ini_t = [0]
        test_compare(data, d, s, ini_t, setup_properties)
