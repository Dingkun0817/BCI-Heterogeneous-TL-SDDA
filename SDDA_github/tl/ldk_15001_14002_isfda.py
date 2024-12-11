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
from utils.dataloader import read_mi_combine_tar, read_mi_combine_tar_diff
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
        acc_scores = accuracy_score(y_true, pred)
        pre_scores = precision_score(y_true, pred)
        rec_scores = recall_score(y_true, pred)
        f1_scores = f1_score(y_true, pred)
        if args.data_name == 'BNCI2014001-4':
            y_pred = np.array(y_pred).reshape(-1, )  # multiclass
        else:
            y_pred = np.array(y_pred).reshape(-1, args.class_num)[:, 1]  # binary
    else:
        predict = torch.from_numpy(np.array(y_pred)).to(torch.float32).reshape(-1, args.class_num)
        y_pred = np.array(predict).reshape(-1, args.class_num)[:, 1]  # binary
        score = roc_auc_score(y_true, y_pred)
    return acc_scores * 100, pre_scores * 100, rec_scores * 100, f1_scores * 100, y_pred


def train_target(args):
    if not args.align:
        extra_string = '_noEA'
    else:
        extra_string = ''
    X_1, y_1, num_subjects_1, paradigm_1, sample_rate_1, ch_num_1 = data_process(args.data1)
    X_2, y_2, num_subjects_2, paradigm_2, sample_rate_2, ch_num_2 = data_process(args.data2)
    # if args.align:
    #     print("-------   START EA: -------")
    #     X_1 = data_alignment(X_1, args.N1, args.data1)
    #     # X_2 = data_alignment(X_2, args.N2, args.data2)
    #     X_2 = data_alignment_session(X_2, args.N2, args.data2)
    #     print("-------   FINISH EA: -------")
    X_1 = X_1[:, [4, 6, 8], :]
    X_2 = X_2[:, [4, 7, 10], :]

    print('======================subject {}============================='.format(args.idt))
    train_x = X_1
    train_y = y_1
    data_subjects = np.split(X_2, indices_or_sections=14, axis=0)
    labels_subjects = np.split(y_2, indices_or_sections=14, axis=0)
    test_x = data_subjects.pop(args.idt)
    test_y = labels_subjects.pop(args.idt)
    print('train_x, train_y, test_x, test_y.shape', train_x.shape, train_y.shape, test_x.shape, test_y.shape)
    dset_loaders = data_loader(train_x, train_y, test_x, test_y, args)

    args.chn = 3

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
        criterion = nn.CrossEntropyLoss()
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
                    acc_t_te, pre_t_te, rec_t_te, f1_t_te, _ = cal_acc_comb(dset_loaders["Target"], base_network, args=args)
                    log_str = 'Task: {}, Iter:{}/{}; Offline-EA Acc = {:.2f}%'.format(args.task_str, int(iter_num // len(dset_loaders["source"])), int(max_iter // len(dset_loaders["source"])), acc_t_te)
                else:
                    acc_t_te, _ = cal_auc_comb(dset_loaders["Target-Imbalanced"], base_network, args=args)
                    log_str = 'Task: {}, Iter:{}/{}; Offline-EA AUC = {:.2f}%'.format(args.task_str, int(iter_num // len(dset_loaders["source"])), int(max_iter // len(dset_loaders["source"])), acc_t_te)
                args.log.record(log_str)
                print(log_str)

                base_network.train()

        print('saving model...')
        torch.save(base_network.state_dict(),
                   './runs/' + str(args.data_name) + '/' + str(args.backbone) + '_S' + str(
                       args.idt) + '_seed' + str(args.SEED) + extra_string + '.ckpt')

    base_network.eval()

    # acc_scores = cal_score_online(dset_loaders["Target-Online"], base_network, args=args)
    # if args.balanced:
    #     log_str = 'Task: {}, Online IEA Acc = {:.2f}%'.format(args.task_str, acc_scores)
    #     # log_str = 'Task: {}, Online IEA Pre = {:.2f}%'.format(args.task_str, pre_scores)
    #     # log_str = 'Task: {}, Online IEA Rec = {:.2f}%'.format(args.task_str, rec_scores)
    #     # log_str = 'Task: {}, Online IEA F1 = {:.2f}%'.format(args.task_str, f1_scores)
    # else:
    #     log_str = 'Task: {}, Online IEA AUC = {:.2f}%'.format(args.task_str, acc_scores)
    # args.log.record(log_str)
    # print(log_str)

    print('executing TTA...')

    if args.balanced:
        # acc_t_te, pre_t_te, rec_t_te, f1_t_te, y_pred = ISFDA(dset_loaders["Target-Online"], base_network, args=args, balanced=True)
        acc_t_te, pre_t_te, rec_t_te, f1_t_te, y_pred = ISFDA(dset_loaders["Target"], base_network, args=args, balanced=True)
        log_str = 'Task: {}, TTA Acc = {:.2f}%'.format(args.task_str, acc_t_te)
    else:
        acc_t_te, pre_t_te, rec_t_te, f1_t_te, y_pred = ISFDA(dset_loaders["Target-Online-Imbalanced"], base_network, args=args, balanced=False)
        log_str = 'Task: {}, TTA AUC = {:.2f}%'.format(args.task_str, acc_t_te)
    args.log.record(log_str)
    print(log_str)

    if args.balanced:
        print('Test Acc = {:.2f}%'.format(acc_t_te))
        print('Test Pre = {:.2f}%'.format(pre_t_te))
        print('Test Rec = {:.2f}%'.format(rec_t_te))
        print('Test F1 = {:.2f}%'.format(f1_t_te))
    else:
        print('Test AUC = {:.2f}%'.format(acc_t_te))

    torch.save(base_network.state_dict(), './runs/' + str(args.data_name) + '/' + str(args.backbone) + '_S' + str(args.idt) + '_seed' + str(
        args.SEED) + extra_string + '_adapted' + '.ckpt')

    # save the predictions for ensemble
    with open('./logs/' + str(args.data_name) + '_' + str(args.method) + '_seed_' + str(args.SEED) +"_pred.csv", 'a') as f:
        writer = csv.writer(f)
        writer.writerow(y_pred)

    gc.collect()
    if args.data_env != 'local':
        torch.cuda.empty_cache()

    return acc_t_te, pre_t_te, rec_t_te, f1_t_te


if __name__ == '__main__':

    dataset1 = 'BNCI2015001'
    dataset2 = 'BNCI2014002'

    dct = pd.DataFrame(
        columns=['dataset', 'avg', 'std', 's0', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11',
                 's12', 's13'])

    paradigm1, N1, chn1, class_num1, time_sample_num1, sample_rate1 = 'MI', 12, 13, 2, 2561, 512
    paradigm2, N2, chn2, class_num2, time_sample_num2, sample_rate2 = 'MI', 14, 15, 2, 2561, 512

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
    balanced = True
    calc_time = False

    source_str = str(dataset1)
    target_str = str(dataset2)
    data_name = source_str + '_2_' + target_str
    args = argparse.Namespace(N1=N1, data_name1=dataset1,
                              N2=N2, data_name2=dataset2,
                              align=align, lr=lr, t=t, max_epoch=max_epoch, stride=stride, steps=steps, calc_time=calc_time,
                              class_num=2, chn=3, time_sample_num=2561, layer='wn',
                              sample_rate=512, feature_deep_dim=640, trial_num=100,
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

    total_acc = []
    total_pre = []
    total_rec = []
    total_f1 = []

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

        sub_acc_all = np.zeros(N2)
        sub_pre_all = np.zeros(N2)
        sub_rec_all = np.zeros(N2)
        sub_f1_all = np.zeros(N2)

        for idt in range(N2):
            args.idt = idt
            source_str = 'Except_S' + str(idt)
            target_str = 'S' + str(idt)
            args.task_str = source_str + '_2_' + target_str
            info_str = '\n========================== Transfer to ' + target_str + ' =========================='
            print(info_str)
            my_log.record(info_str)
            args.log = my_log

            sub_acc_all[idt], sub_pre_all[idt], sub_rec_all[idt], sub_f1_all[idt] = train_target(args)
            print("_++_+_+_+_+_+_+_++__+++_+_+_++++++++++_+++++____+_+_++_+_+_+_+_")
            print('Sub acc: ', np.round(sub_acc_all, 3))
            print('Avg acc: ', np.round(np.mean(sub_acc_all), 3))
            print('Sub pre: ', np.round(sub_pre_all, 3))
            print('Avg pre: ', np.round(np.mean(sub_pre_all), 3))
            print('Sub rec: ', np.round(sub_rec_all, 3))
            print('Avg rec: ', np.round(np.mean(sub_rec_all), 3))
            print('Sub f1: ', np.round(sub_f1_all, 3))
            print('Avg f1: ', np.round(np.mean(sub_f1_all), 3))
            total_acc.append(sub_acc_all)
            total_pre.append(sub_pre_all)
            total_rec.append(sub_rec_all)
            total_f1.append(sub_f1_all)

            acc_sub_str = str(np.round(sub_acc_all, 3).tolist())
            acc_mean_str = str(np.round(np.mean(sub_acc_all), 3).tolist())
            pre_sub_str = str(np.round(sub_pre_all, 3).tolist())
            pre_mean_str = str(np.round(np.mean(sub_pre_all), 3).tolist())
            rec_sub_str = str(np.round(sub_rec_all, 3).tolist())
            rec_mean_str = str(np.round(np.mean(sub_rec_all), 3).tolist())
            f1_sub_str = str(np.round(sub_f1_all, 3).tolist())
            f1_mean_str = str(np.round(np.mean(sub_f1_all), 3).tolist())
            args.log.record("\n==========================================")
            args.log.record(acc_sub_str)
            args.log.record(acc_mean_str)
            args.log.record(pre_sub_str)
            args.log.record(pre_mean_str)
            args.log.record(rec_sub_str)
            args.log.record(rec_mean_str)
            args.log.record(f1_sub_str)
            args.log.record(f1_mean_str)

        args.log.record('\n' + '#' * 20 + 'final results' + '#' * 20)

        print('total_acc is : ', str(total_acc))
        print('total_pre is : ', str(total_pre))
        print('total_rec is : ', str(total_rec))
        print('total_f1 is : ', str(total_f1))

        args.log.record(str(total_acc))
        args.log.record(str(total_pre))
        args.log.record(str(total_rec))
        args.log.record(str(total_f1))

        subject_acc_mean = np.round(np.average(total_acc, axis=0), 5)
        total_acc_mean = np.round(np.average(np.average(total_acc)), 5)
        total_acc_std = np.round(np.std(np.average(total_acc, axis=1)), 5)

        subject_pre_mean = np.round(np.average(total_pre, axis=0), 5)
        total_pre_mean = np.round(np.average(np.average(total_pre)), 5)
        total_pre_std = np.round(np.std(np.average(total_pre, axis=1)), 5)

        subject_rec_mean = np.round(np.average(total_rec, axis=0), 5)
        total_rec_mean = np.round(np.average(np.average(total_rec)), 5)
        total_rec_std = np.round(np.std(np.average(total_rec, axis=1)), 5)

        subject_f1_mean = np.round(np.average(total_f1, axis=0), 5)
        total_f1_mean = np.round(np.average(np.average(total_f1)), 5)
        total_f1_std = np.round(np.std(np.average(total_f1, axis=1)), 5)
        print("+-+-+-+-+_+_+_+__+_+_+++_++_+_+_+_++_++_+_+_+_+_+_+_+_+_+_+_+_+")

        print('subject_acc_mean is : ', subject_acc_mean)
        print('total_acc_mean is : ', total_acc_mean)
        print('total_acc_std is : ', total_acc_std)

        print('subject_pre_mean is : ', subject_pre_mean)
        print('total_pre_mean is : ', total_pre_mean)
        print('total_pre_std is : ', total_pre_std)

        print('subject_rec_mean is : ', subject_rec_mean)
        print('total_rec_mean is : ', total_rec_mean)
        print('total_rec_std is : ', total_rec_std)

        print('subject_f1_mean is : ', subject_f1_mean)
        print('total_f1_mean is : ', total_f1_mean)
        print('total_f1_std is : ', total_f1_std)

        args.log.record(str(subject_acc_mean))
        args.log.record(str(total_acc_mean))
        args.log.record(str(total_acc_std))

        args.log.record(str(subject_pre_mean))
        args.log.record(str(total_pre_mean))
        args.log.record(str(total_pre_std))

        args.log.record(str(subject_rec_mean))
        args.log.record(str(total_rec_mean))
        args.log.record(str(total_rec_std))

        args.log.record(str(subject_f1_mean))
        args.log.record(str(total_f1_mean))
        args.log.record(str(total_f1_std))

        # source_str = str(dataset1)
        # target_str = str(dataset2)
        # data_name = source_str + ' 2 ' + target_str
        result_dct = {'dataset': data_name, 'acc_avg': total_acc_mean, 'acc_std': total_acc_std,
                      'pre_avg': total_pre_mean, 'pre_std': total_pre_std,
                      'rec_avg': total_rec_mean, 'rec_std': total_rec_std,
                      'f1_avg': total_f1_mean, 'f1_std': total_f1_std}
        for i in range(len(subject_acc_mean)):
            result_dct['s' + str(i)] = subject_acc_mean[i]

        dct = dct.append(result_dct, ignore_index=True)

    # save results to csv
    dct.to_csv('./logs/' + str(args.method) + ".csv")