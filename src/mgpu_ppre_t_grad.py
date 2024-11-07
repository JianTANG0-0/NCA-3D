from .LoadFunc_train import *


def train_onestep(logger, num_t, training_set, ca, rank, optimizer, echo=False):
    try:
        ca.train()
        size = len(training_set.dataset)
        echo_size = size//10
        acc_train = []
        for j, (x_batch, xt_batch) in enumerate(training_set):
            l_time_sum = 0.0
            l_ea1, l_ea2, l_ea3, l_s = 0.0, 0.0, 0.0, 0.0
            x_batch, xt_batch = x_batch.to(rank), xt_batch.to(rank)

            for nca_step in range(num_t):
                try:
                    x_batch = ca(x_batch)
                except RuntimeError as exception:
                    if "out of memory" in str(exception):
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                    else:
                        raise exception

                l_time_sum += ca.cweigt_loss(x_batch, xt_batch[:, nca_step])

                if nca_step < num_t:
                    x_batch[:, ca.tot_cla+1:ca.tot_cla+2, ...] = xt_batch[:, nca_step, 4:5].type(torch.FloatTensor)
                if nca_step < num_t - 1:
                    x_batch[:, ca.tot_cla+2:ca.tot_cla+3, ...] = xt_batch[:, nca_step + 1, 4:5].type(torch.FloatTensor)
            # l_time_sum += regularization_param * regularization(ca,regularization_exp)

            l_time_sum.backward()
            for p in ca.parameters():
                p.grad /= (p.grad.norm() + 1e-8)
            optimizer.step()
            optimizer.zero_grad()

            if echo:
                if ((j+1)*len(x_batch) % echo_size < len(x_batch)) & ((rank == 0) | (rank == torch.device('cuda'))):
                    ca.eval()
                    train_acc = cal_acc(x_batch, xt_batch[:, -1], ca)
                    acc_train.append(train_acc)
                    logger.info(f"loss: {l_time_sum.item():>7f}  Acc: {train_acc:>0.3f}  [{(j+1)*len(x_batch):>5d}/{size:>5d}]\n")
                    torch.cuda.empty_cache()


        if echo:
            if ((rank == 0) | (rank == torch.device('cuda'))):
                logger.info(f"Average Acc: {np.mean(np.array(acc_train))}\n")
            return np.mean(np.array(acc_train))
        else:
            return None, logger
    except Exception as e:
        logger.error(f"Error occurred during training: {e}", exc_info=True)
        close_logging_handlers()



def test(logger, num_t, validation_set, ca, rank, epoch, folder_name, note='valid'):
    size = len(validation_set.dataset)
    num_batches = len(validation_set)
    ca.eval()
    with torch.no_grad():
        acc_val = []
        l_valid = 0.0
        for j, (x_valid, target_valid) in enumerate(validation_set):
            x_valid, target_valid = x_valid.to(rank), target_valid.to(rank)
            for nca_step in range(num_t):
                try:
                    x_valid = ca(x_valid)  # _input)
                except RuntimeError as exception:
                    if "out of memory" in str(exception):
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                    else:
                        raise exception
                l_valid += ca.cweigt_loss(x_valid, target_valid[:, nca_step]).item()
                if nca_step < num_t:
                    x_valid[:, ca.tot_cla+1:ca.tot_cla+2, ...] = target_valid[:, nca_step, 4:5].type(torch.FloatTensor)
                if nca_step < num_t - 1:
                    x_valid[:, ca.tot_cla+2:ca.tot_cla+3, ...] = target_valid[:, nca_step + 1, 4:5].type(torch.FloatTensor)
            valid_acc = cal_acc(x_valid, target_valid[:, -1], ca)
            acc_val.append(valid_acc)
    if ((rank == 0) | (rank == torch.device('cuda'))):
        logger.info(f"Accuracy: {np.mean(np.array(acc_val)):>0.3f}%, Avg loss: {l_valid/num_batches:>8f} \n")
        valid_acc1, valid_acc2, valid_acc3 = cal_ori_acc(x_valid, target_valid[:, -1], ca)
        logger.info(f"Accuracy1: {valid_acc1*100:>0.3f}%, Accuracy2: {valid_acc2*100:>0.3f}%, Accuracy3: {valid_acc3*100:>0.3f}% \n")
        predict_fig(x_valid, target_valid, ca, epoch, folder_name, note)
    return l_valid, np.mean(np.array(acc_val))


def train(rank, world_size, logging_file, parameterization, fold_path, training_data, validation_data, testing_data):

    num_t = parameterization.get("tot_t", 18)
    rand_seed = parameterization.get("rand_seed", -1)
    if rand_seed != -1:
        random.seed(rand_seed)
        np.random.seed(rand_seed)
        torch.manual_seed(rand_seed)
        torch.cuda.manual_seed_all(rand_seed)

    path = fold_path + '/model.pkl'
    # setup the process groups
    if world_size > 1:
        setup(rank, world_size)
    else:
        rank = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
    logger = setup_logging(logging_file)
    ca = NCA(parameterization)
    ca.initialize_weights()
    retrain = parameterization.get("retrain", False)
    if retrain:
        ca = load_model()
    # move model to rank
    # process_group = torch.distributed.new_group()
    # ca = torch.nn.SyncBatchNorm.convert_sync_batchnorm(ca, process_group) # normal norm layer cannot work with DDP
    ca = ca.to(rank)
    if world_size > 1:
        ca = DDP(ca, device_ids=[rank], output_device=rank)  # ,find_unused_parameters=True
    model = ca.module if isinstance(ca, torch.nn.parallel.DistributedDataParallel) else ca

    epoch_num = parameterization.get("epoch", 4000)
    echo_step = int(parameterization.get("echo_step", 20))
    early_stopper = EarlyStopping(patience=70, verbose=True, delta=0.1)

    optimizer = torch.optim.Adam(ca.parameters(), lr=parameterization.get("lr", 0.001),
                                 weight_decay=parameterization.get("l2_reg", 0.0))
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(parameterization.get("step_size", 3000)),
        gamma=parameterization.get("gamma", 1.0),  # default is no learning rate decay
    )

    batch_size = int(parameterization.get("batch_size", 100))

    training_set = prepare(training_data, rank, world_size, batch_size)
    validation_set = prepare(validation_data, rank, world_size, batch_size)
    testing_set = prepare(testing_data, rank, world_size, batch_size)

    for epoch in range(epoch_num):
        if world_size > 1:
            training_set.sampler.set_epoch(epoch)
        if ((epoch % echo_step//5 == 0) or (epoch == (epoch_num - 1))) & ((rank == 0) | (world_size == 1)):
            if world_size>1:
                dist.barrier()
            torch.save(model.state_dict(), path)
        if ((epoch % echo_step == 0) or (epoch == (epoch_num - 1))) & ((rank == 0) | (world_size == 1)):
            logger.info(f"Curent Epoch {epoch} :\n")
            acc_train = train_onestep(logger, num_t, training_set, model, rank, optimizer, True)
            logger.info(f"Validation result :\n")
            l_valid, acc_val = test(logger, num_t, validation_set, model, rank, epoch, fold_path)
            early_stopper(acc_val)
        else:
            acc_train = train_onestep(logger, num_t, training_set, model, rank, optimizer, False)
            l_valid = 0.0
            acc_train = 0

        if np.isnan(l_valid) or (acc_train > 96) or early_stopper.early_stop:
            if ((rank == 0) | (world_size == 1)):
                logger.info("######### training end ##########")
            break
        scheduler.step()
    if ((rank == 0) | (world_size == 1)):
        logger.info(f"Testing result :\n")
    test(logger, num_t, testing_set, model, rank, epoch, fold_path, 'test')
    close_logging_handlers()
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()



# load the CRNN Model from file
def load_model(model_file='../Setup_0'):
    model_para = np.load(model_file + '/model_setting.npy', allow_pickle=True).item()
    model_file = model_file + '/model.pkl'
    ca = NCA(model_para)
    ca.load_state_dict(torch.load(model_file, map_location='cpu'))
    return ca


def ensemble_runs(parameters, nca_train_data, valid_ratio, test_ratio, train_dataset, world_size):
    settings = list(itertools.product(*parameters.values()))
    i = 0
    folder_name = str(os.getcwd())
    for setup in settings:
        print("###################################")
        print('setup:  No.', i + 1)
        folder_path = folder_name + "/Setup_" + str(i)
        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)
        logging_file = folder_path+"/train_log.log"
        # Use the custom logger setup function
        logger = setup_logging(logging_file)

        setup_properties = parameters
        j = 0
        for key in parameters:
            setup_properties[key] = setup[j]
            j = j + 1
        logger.info(setup_properties)
        logger.info('data stored at: ', folder_path)
        logger.info("###################################")
        setup_path = folder_path + '/model_setting.npy'
        np.save(setup_path, setup_properties)

        if not train_dataset:
            train_dataset = DefDataset(setup_properties, nca_train_data[:-int(np.floor(nca_train_data.shape[0] * (valid_ratio+test_ratio)))])
            valid_dataset = DefDataset(setup_properties,
                                       nca_train_data[-int(np.floor(nca_train_data.shape[0] * (valid_ratio+test_ratio))):-int(np.floor(nca_train_data.shape[0] * test_ratio))])
            test_dataset = DefDataset(setup_properties,
                                      nca_train_data[-int(np.floor(nca_train_data.shape[0] * test_ratio)):])
        else:
            if len(parameters['speedup_rate']) > 1:# or len(parameters['tot_t']) > 1:
                train_dataset = DefDataset(setup_properties, nca_train_data[:-int(
                    np.floor(nca_train_data.shape[0] * (valid_ratio + test_ratio)))])
                valid_dataset = DefDataset(setup_properties,
                                           nca_train_data[
                                           -int(np.floor(nca_train_data.shape[0] * (valid_ratio + test_ratio))):-int(
                                               np.floor(nca_train_data.shape[0] * test_ratio))])
                test_dataset = DefDataset(setup_properties,
                                          nca_train_data[-int(np.floor(nca_train_data.shape[0] * test_ratio)):])

        logger.info(f'Training data size:{len(train_dataset)};  Validation '
              f'data size:{len(valid_dataset)};  Testing data size:{len(test_dataset)};  ')
        # Debugging the logger before passing it to test()
        print("Logger Handlers Before test():", logger.handlers)
        print("Logger Level Before test():", logger.level)
        mp.spawn(train, args=(world_size, logging_file, setup_properties, folder_path, train_dataset, valid_dataset, test_dataset),
                 nprocs=world_size)
        i = i + 1
    logger.info("ending training")

if __name__ == '__main__':

    nca_train_data = np.load('../dirsoild_3.npy', allow_pickle=True)[:, ..., :28, :-1]
    #'''
    for i in range(6,7):
        nca_train_data2 = np.load('dirsoild_' + str(i) + '.npy', allow_pickle=True)[:, ...,:28, :-1]
        nca_train_data = np.concatenate([nca_train_data, nca_train_data2], axis=0)
        del nca_train_data2
    #'''
    nca_train_data = torch.from_numpy(nca_train_data)
    nca_train_data = nca_train_data.permute([0, 1, 5, 4, 2, 3]).type(torch.FloatTensor)
    print(f'Input data shape: {nca_train_data.shape};')

    train_dataset = None

    parameters = {
        "lr": [1e-3],
        "step_size": [700],
        "gamma": [0.5],
        "hid_lay_num": [3],
        "kernel_size": [3],
        "neu_num": [384],
        "epoch": [1600],
        "echo_step": [40],
        "rand_seed": [3024],
        "speedup_rate": [[1.0]],
        "batch_size": [4],
        "drop": [0.9],
        "tot_t": [10],
        "EA_cla": [[10, 5, 10]],
        "loss_weight": [[0.1, 0.1, 0.1]],
         "retrain": [True],
         "reg_para": [1e-10],
         "reg_exp": [2.0],
    }

    # define GPU number
    world_size = 3

    # define validation and test datasize
    valid_ratio = 3./10
    test_ratio = 2./10


    # run foreach setup
    ensemble_runs(parameters, nca_train_data, valid_ratio, test_ratio, train_dataset, world_size)