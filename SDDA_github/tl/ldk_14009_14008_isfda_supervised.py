import sys
# sys.path.append('E:\Pycharm_BCI\Self_Study\T-TIME')
sys.path.append('/data1/ldk/T-TIME')

import numpy as np
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import csv
from utils.network import backbone_net
from utils.LogRecord import LogRecord
from utils.dataloader import read_mi_combine_tar, read_mi_combine_tar_diff, read_mi_combine_tar_diff_supervised_3
from utils.utils_1 import fix_random_seed, cal_acc_comb, data_loader, cal_auc_comb, cal_score_online
from utils.alg_utils import EA, EA_online
from scipy.linalg import fractional_matrix_power
from utils.loss import Entropy
from sklearn.metrics import accuracy_score, auc, roc_auc_score, f1_score, precision_score, recall_score
from tl.utils.utils_1 import data_alignment, data_alignment_session
from utils.dataloader import data_process

import gc
import sys
import time


def ISFDA(loader, model, args, balanced=True):
    # ISFDA
    # online-TTA version

    y_true = []
    y_pred = []

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # initialize test reference matrix for Incremental EA
    if args.align:
        R = 0

    iter_test = iter(loader)

    # loop through test data stream one by one
    for i in range(len(loader)):
        #################### Phase 1: target label prediction ####################
        model.eval()
        data = next(iter_test)
        inputs = data[0]
        labels = data[1]
        inputs = inputs.reshape(inputs.size(0), 1, inputs.shape[-2], inputs.shape[-1]).cpu()

        # accumulate test data
        if i == 0:
            data_cum = inputs.float().cpu()
        else:
            data_cum = torch.cat((data_cum, inputs.float().cpu()), 0)

        # Incremental EA
        # if args.align:
        #     sample_test = EA(data_cum)
            # start_time = time.time()
            #
            # if i == 0:
            #     sample_test = data_cum.reshape(args.chn, args.time_sample_num)
            # else:
            #     sample_test = data_cum[i].reshape(args.chn, args.time_sample_num)
            # # update reference matrix
            # R = EA_online(sample_test, R, i)   # ????????????????????????????????????
            #
            # sqrtRefEA = fractional_matrix_power(R, -0.5)
            # # transform current test sample
            # sample_test = np.dot(sqrtRefEA, sample_test)
            #
            # EA_time = time.time()
            # if args.calc_time:
            #     print('sample ', str(i), ', pre-inference IEA finished time in ms:', np.round((EA_time - start_time) * 1000, 3))
            # sample_test = sample_test.reshape(1, 1, args.chn, args.time_sample_num)
        # else:
            # sample_test = data_cum[i].numpy()
        sample_test = data_cum.numpy()
            # sample_test = sample_test.reshape(96, 1, sample_test.shape[1], sample_test.shape[2])

        if args.data_env != 'local':
            sample_test = torch.from_numpy(sample_test).to(torch.float32).cuda()
        else:
            sample_test = torch.from_numpy(sample_test).to(torch.float32)

        features, outputs = model(sample_test)

        softmax_out = nn.Softmax(dim=1)(outputs)

        outputs = outputs.float().cpu()
        labels = labels.float().cpu()
        _, predict = torch.max(outputs, 1)


        # y_pred.append(softmax_out.detach().cpu().numpy())
        # y_true.append(labels.item())
        y_pred = softmax_out.detach().cpu().numpy()
        y_true.extend(labels.tolist())

        #################### Phase 2: target model update ####################
        model.train()
        # sliding batch
        # if (i + 1) >= args.test_batch and (i + 1) % args.stride == 0:


            # if args.align:
            #     batch_test = np.copy(data_cum[i - args.test_batch + 1:i + 1])
            #     # transform test batch
            #     batch_test = np.dot(sqrtRefEA, batch_test)
            #     batch_test = np.transpose(batch_test, (1, 2, 0, 3))
            # else:


            # batch_test = data_cum[i - args.test_batch + 1:i + 1].numpy()
            # batch_test = batch_test.reshape(args.test_batch, 1, batch_test.shape[2], batch_test.shape[3])
            #
            # if args.data_env != 'local':
            #     batch_test = torch.from_numpy(batch_test).to(torch.float32).cuda()
            # else:
            #     batch_test = torch.from_numpy(batch_test).to(torch.float32)
            #
            # start_time = time.time()
            # for step in range(args.steps):

        # features, outputs = model(batch_test)
        outputs = outputs.float().cpu()
        args.epsilon = 1e-5
        softmax_out = nn.Softmax(dim=1)(outputs / args.t)

        # IM
        CEM_loss = torch.mean(Entropy(softmax_out))
        msoftmax = softmax_out.mean(dim=0)
        MDR_loss = torch.sum(msoftmax * torch.log(msoftmax + args.epsilon))
        im_loss = CEM_loss + MDR_loss

        # ISFDA
        # Intra-class Tightening and Inter-class Separation
        # Class Center Distances based on PL
        pl = torch.max(softmax_out, 1)[1]

        # print(softmax_out)

        class_0_ids = torch.where(pl == 0)[0]
        class_1_ids = torch.where(pl == 1)[0]

        for l in range(len(softmax_out)):
            if softmax_out[l][0] >= 0.5 and softmax_out[l][0] < 0.6:
                class_1_ids = torch.cat([class_1_ids, torch.tensor([l])])
            elif softmax_out[l][1] >= 0.5 and softmax_out[l][1] < 0.6:
                class_0_ids = torch.cat([class_0_ids, torch.tensor([l])])

        # 分类讨论找中心点(如果类里没东西, dist_loss不用算了，就是0)
        dist_loss = None
        if len(class_0_ids) == 1:
            class_0_center = features[class_0_ids]
        elif len(class_0_ids) == 0:
            dist_loss = 0
        else:
            class_0_center = torch.mean(features[class_0_ids], dim=0)
        if len(class_1_ids) == 1:
            class_1_center = features[class_1_ids]
        elif len(class_1_ids) == 0:
            dist_loss = 0
        else:
            class_1_center = torch.mean(features[class_1_ids], dim=0)

        if dist_loss is None:
            cos = nn.CosineSimilarity(dim=1)
            inter_loss = torch.sum(torch.tensor(1) - cos(features[class_0_ids].cpu(), class_1_center.cpu().reshape(1, -1)))
            inter_loss += torch.sum(torch.tensor(1) - cos(features[class_1_ids].cpu(), class_0_center.cpu().reshape(1, -1)))
            inter_loss = inter_loss / args.test_batch
            intra_loss = torch.sum(torch.tensor(1) - cos(features[class_0_ids].cpu(), class_0_center.cpu().reshape(1, -1)))
            intra_loss += torch.sum(torch.tensor(1) - cos(features[class_1_ids].cpu(), class_1_center.cpu().reshape(1, -1)))
            intra_loss = intra_loss / args.test_batch
            dist_loss = intra_loss - inter_loss

        loss = im_loss + dist_loss
        # print("___________++++++++++_+__+__+")
        # print(im_loss)
        # print(dist_loss)
        # print("___________++++++++++_+__+__+")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()

    if balanced:
        _, predict = torch.max(torch.from_numpy(np.array(y_pred)).to(torch.float32).reshape(-1, args.class_num), 1)
        pred = torch.squeeze(predict).float()


        auc_scores = roc_auc_score(y_true, pred)

        # acc_scores = accuracy_score(y_true, pred)
        # pre_scores = precision_score(y_true, pred)
        # rec_scores = recall_score(y_true, pred)
        # f1_scores = f1_score(y_true, pred)
        if args.data_name == 'BNCI2014001-4':
            y_pred = np.array(y_pred).reshape(-1, )  # multiclass
        else:
            y_pred = np.array(y_pred).reshape(-1, args.class_num)[:, 1]  # binary
    else:
        predict = torch.from_numpy(np.array(y_pred)).to(torch.float32).reshape(-1, args.class_num)
        y_pred = np.array(predict).reshape(-1, args.class_num)[:, 1]  # binary
        auc_scores = roc_auc_score(y_true, y_pred)
    return auc_scores * 100, y_pred
    # return acc_scores * 100, pre_scores * 100, rec_scores * 100, f1_scores * 100, y_pred


def train_target(args):
    if not args.align:
        extra_string = '_noEA'
    else:
        extra_string = ''
    X_src, y_src, X_tar, y_tar, X_tar_unlabel, y_tar_unlabel = read_mi_combine_tar_diff_supervised_3(args)
    print('X_src, y_src, X_tar, y_tar, X_tar_unlabel, y_tar_unlabel:', X_src.shape, y_src.shape, X_tar.shape, y_tar.shape, X_tar_unlabel.shape, y_tar_unlabel.shape)
    dset_loaders = data_loader(X_src, y_src, X_tar, y_tar, args)
    dset_loaders2 = data_loader(X_src, y_src, X_tar_unlabel, y_tar_unlabel, args)

    print('======================subject {}============================='.format(args.idt))

    args.chn = 8

    netF, netC = backbone_net(args, return_type='xy')
    if args.data_env != 'local':
        netF, netC = netF.cuda(), netC.cuda()
    base_network = nn.Sequential(netF, netC)

    if args.max_epoch == 0:
        if args.align:
            if args.data_env != 'local':
                base_network.load_state_dict(torch.load('./runs/' + str(args.data_name1) + '/' + str(args.backbone) +
                    '_S' + str(args.idt) + '_seed' + str(args.SEED) + extra_string + '.ckpt'))
            else:
                base_network.load_state_dict(torch.load('./runs/' + str(args.data_name1) + '/' + str(args.backbone) +
                    '_S' + str(args.idt) + '_seed' + str(args.SEED) + extra_string + '.ckpt', map_location=torch.device('cpu')))
    else:
        # criterion = nn.CrossEntropyLoss()
        '''
            Imbalanced数据集的交叉熵要加权重
        '''
        num_class_1 = 700
        num_class_0 = 3500
        total_samples = num_class_0 + num_class_1
        weight_class_0 = num_class_1 / total_samples
        weight_class_1 = num_class_0 / total_samples
        weight = torch.tensor([weight_class_0, weight_class_1], dtype=torch.float32)
        weight = weight.cuda()
        criterion = nn.CrossEntropyLoss(weight=weight)
        optimizer_f = optim.Adam(netF.parameters(), lr=args.lr)
        optimizer_c = optim.Adam(netC.parameters(), lr=args.lr)

        max_iter = args.max_epoch * len(dset_loaders["source"])
        interval_iter = max_iter // args.max_epoch
        args.max_iter = max_iter
        iter_num = 0
        base_network.train()

        while iter_num < max_iter:
            try:
                inputs_source, labels_source = next(iter_source)
            except:
                iter_source = iter(dset_loaders["source"])
                inputs_source, labels_source = next(iter_source)

            if inputs_source.size(0) == 1:
                continue

            iter_num += 1

            features_source, outputs_source = base_network(inputs_source)

            classifier_loss = criterion(outputs_source, labels_source)

            optimizer_f.zero_grad()
            optimizer_c.zero_grad()
            classifier_loss.backward()
            optimizer_f.step()
            optimizer_c.step()

            if iter_num % interval_iter == 0 or iter_num == max_iter:
                base_network.eval()

                if args.balanced:
                    acc_t_te, pre_t_te, rec_t_te, f1_t_te, _ = cal_acc_comb(dset_loaders2["Target"], base_network, args=args)
                    log_str = 'Task: {}, Iter:{}/{}; Offline-EA Acc = {:.2f}%'.format(args.task_str, int(iter_num // len(dset_loaders["source"])), int(max_iter // len(dset_loaders["source"])), acc_t_te)
                else:
                    acc_t_te, _ = cal_auc_comb(dset_loaders2["Target-Imbalanced"], base_network, args=args)
                    log_str = 'Task: {}, Iter:{}/{}; Offline-EA AUC = {:.2f}%'.format(args.task_str, int(iter_num // len(dset_loaders["source"])), int(max_iter // len(dset_loaders["source"])), acc_t_te)
                args.log.record(log_str)
                print(log_str)

                base_network.train()

        print('saving model...')
        torch.save(base_network.state_dict(),
                   './runs/' + str(args.data_name) + '/' + str(args.backbone) + '_S' + str(
                       args.idt) + '_seed' + str(args.SEED) + extra_string + '.ckpt')

    base_network.eval()

    print('executing TTA...')

    if args.balanced:
        # acc_t_te, pre_t_te, rec_t_te, f1_t_te, y_pred = ISFDA(dset_loaders["Target-Online"], base_network, args=args, balanced=True)
        acc_t_te, pre_t_te, rec_t_te, f1_t_te, y_pred = ISFDA(dset_loaders2["Target"], base_network, args=args, balanced=True)
        log_str = 'Task: {}, TTA Acc = {:.2f}%'.format(args.task_str, acc_t_te)
    else:
        auc_t_te, y_pred = ISFDA(dset_loaders2["Target"], base_network, args=args, balanced=False)
        log_str = 'Task: {}, TTA AUC = {:.2f}%'.format(args.task_str, auc_t_te)
    args.log.record(log_str)
    print(log_str)

    if args.balanced:
        print('Test Acc = {:.2f}%'.format(acc_t_te))
        print('Test Pre = {:.2f}%'.format(pre_t_te))
        print('Test Rec = {:.2f}%'.format(rec_t_te))
        print('Test F1 = {:.2f}%'.format(f1_t_te))
    else:
        print('Test AUC = {:.2f}%'.format(auc_t_te))

    torch.save(base_network.state_dict(), './runs/' + str(args.data_name) + '/' + str(args.backbone) + '_S' + str(args.idt) + '_seed' + str(
        args.SEED) + extra_string + '_adapted' + '.ckpt')

    # save the predictions for ensemble
    with open('./logs/' + str(args.data_name) + '_' + str(args.method) + '_seed_' + str(args.SEED) +"_pred.csv", 'a') as f:
        writer = csv.writer(f)
        writer.writerow(y_pred)

    gc.collect()
    if args.data_env != 'local':
        torch.cuda.empty_cache()

    return auc_t_te


if __name__ == '__main__':

    dataset1 = 'BNCI2014009'
    dataset2 = 'BNCI2014008'

    dct = pd.DataFrame(
        columns=['dataset', 'avg', 'std', 's0', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11',
                 's12', 's13'])

    paradigm1, N1, chn1, class_num1, time_sample_num1, sample_rate1 = 'MI', 10, 16, 2, 206, 256
    paradigm2, N2, chn2, class_num2, time_sample_num2, sample_rate2 = 'MI', 8, 8, 2, 257, 256

    use_pretrained_model = False
    if use_pretrained_model:
        # no training
        max_epoch = 0
    else:
        # training epochs
        max_epoch = 100

    lr = 0.001
    test_batch = 8
    steps = 1
    stride = 1
    align = True
    t = 2
    balanced = False
    calc_time = False

    source_str = str(dataset1)
    target_str = str(dataset2)
    data_name = source_str + '_2_' + target_str
    args = argparse.Namespace(N1=N1, data_name1=dataset1,
                              N2=N2, data_name2=dataset2,
                              align=align, lr=lr, t=t, max_epoch=max_epoch, stride=stride, steps=steps, calc_time=calc_time,
                              class_num=2, chn=8, time_sample_num=206, layer='wn',
                              sample_rate=256, feature_deep_dim=48,
                              data_name=data_name, test_batch=test_batch, balanced=balanced)

    args.method = 'ISFDA-TTA'
    args.backbone = 'EEGNet'

    # train batch size
    args.batch_size = 32

    # GPU device id
    try:
        device_id = str(sys.argv[1])
        os.environ["CUDA_VISIBLE_DEVICES"] = device_id
        args.data_env = 'gpu' if torch.cuda.device_count() != 0 else 'local'
    except:
        args.data_env = 'local'

    total_auc = []

    # update multiple models, independently, from the source models
    for s in [1, 2, 3, 4, 5]:
        args.SEED = s

        fix_random_seed(args.SEED)
        torch.backends.cudnn.deterministic = True

        args.data1 = dataset1
        args.data2 = dataset2
        print(args.data1)
        print(args.data2)
        print(args.method)
        print(args.SEED)
        print(args)
        print("--------------  start running:   --------------")

        args.local_dir = './data/' + str(dataset1) + '2' + str(dataset1) + '/'
        args.result_dir = './logs/'
        my_log = LogRecord(args)
        my_log.log_init()
        my_log.record('=' * 50 + '\n' + os.path.basename(__file__) + '\n' + '=' * 50)

        sub_auc_all = np.zeros(N2)
        for idt in range(N2):  # 留一法, idt作为test, 非idt作为train
            args.idt = idt
            source_str = str(dataset1)
            target_str = str(dataset2) + '    subject ' + str(idt)
            args.task_str = source_str + '_2_' + target_str
            info_str = '\n========================== Transfer to ' + target_str + ' =========================='
            print(info_str)
            my_log.record(info_str)
            args.log = my_log

            # sub_acc_all[idt], sub_pre_all[idt], sub_rec_all[idt], sub_f1_all[idt] = train_target(args)
            sub_auc_all[idt] = train_target(args)
        print('Sub auc: ', np.round(sub_auc_all, 3))
        print('Avg auc: ', np.round(np.mean(sub_auc_all), 3))

        total_auc.append(sub_auc_all)

        auc_sub_str = str(np.round(sub_auc_all, 3).tolist())
        auc_mean_str = str(np.round(np.mean(sub_auc_all), 3).tolist())

        args.log.record("\n==========================================")
        args.log.record(auc_sub_str)
        args.log.record(auc_mean_str)

    args.log.record('\n' + '#' * 20 + 'final results' + '#' * 20)

    print('total_auc is : ', str(total_auc))

    args.log.record(str(total_auc))

    subject_auc_mean = np.round(np.average(total_auc, axis=0), 5)
    total_auc_mean = np.round(np.average(np.average(total_auc)), 5)
    total_auc_std = np.round(np.std(np.average(total_auc, axis=1)), 5)

    print("+-+-+-+-+_+_+_+__+_+_+++_++_+_+_+_++_++_+_+_+_+_+_+_+_+_+_+_+_+")

    print('subject_auc_mean is : ', subject_auc_mean)
    print('total_auc_mean is : ', total_auc_mean)
    print('total_auc_std is : ', total_auc_std)

    args.log.record(str(subject_auc_mean))
    args.log.record(str(total_auc_mean))
    args.log.record(str(total_auc_std))

    source_str = str(dataset1)
    target_str = str(dataset2)
    dataset = source_str + ' 2 ' + target_str
    result_dct = {'dataset': dataset, 'auc_avg': total_auc_mean, 'auc_std': total_auc_std}

    for i in range(len(subject_auc_mean)):
        result_dct['s' + str(i)] = subject_auc_mean[i]

    dct = dct.append(result_dct, ignore_index=True)

    # save results to csv
    dct.to_csv('./logs/' + str(args.method) + ".csv")