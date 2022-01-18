import torch
import os
from extracting_local_graph import LocalGraph
from torch.utils.data import DataLoader
import pickle
import argparse
import numpy as np
from maml import MAML
import time
import psutil
from memory_profiler import memory_usage

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def collate(samples):
    graphs_spt, labels_spt, graph_qry, labels_qry, center_spt, center_qry, nodeidx_spt, nodeidx_qry, support_graph_idx, query_graph_idx = map(
        list, zip(*samples))
    return graphs_spt, labels_spt, graph_qry, labels_qry, center_spt, center_qry, nodeidx_spt, nodeidx_qry, support_graph_idx, query_graph_idx


def main():
    mem_usage = memory_usage(-1, interval=.5, timeout=1)
    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    root = args.data_dir + 'data/'

    print(args)
    with open(root + 'results.txt', 'a') as f:
        f.write(str(args) + '\n')

    feat = np.load(root + 'features.npy', allow_pickle=True)
    print('feature dimension: ', len(feat[0][0]))

    with open(root + 'graph_dgl.pkl', 'rb') as f:
        dgl_graph = pickle.load(f)

    with open(root + 'label.pkl', 'rb') as f:
        info = pickle.load(f)

    labels_num = len(np.unique(np.array(list(info.values()))))
    print('There are {} classes '.format(labels_num))

    if len(feat.shape) == 2:
        feat = [feat]

    config = [('GraphConv', [feat[0].shape[1], args.hidden_dim])]

    if args.h > 1:
        config = config + [('GraphConv', [args.hidden_dim, args.hidden_dim])] * (args.h - 1)

    config = config + [('Linear', [args.hidden_dim, labels_num])]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    maml = MAML(args, config).to(device)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)

    max_acc = 0

    db_train = []
    db_val = []
    db_test = []

    for k_fold in range(10):
        db_train.append(
            LocalGraph(root, 'train_' + str(k_fold), info, n_way=args.n_way, k_shot=args.k_spt, k_query=args.k_qry,
                       batchsz=args.batchsz, args=args, adjs=dgl_graph, h=args.h))
        db_val.append(
            LocalGraph(root, 'val_' + str(k_fold), info, n_way=args.n_way, k_shot=args.k_spt, k_query=args.k_qry,
                       batchsz=100, args=args, adjs=dgl_graph, h=args.h))
        db_test.append(
            LocalGraph(root, 'test_' + str(k_fold), info, n_way=args.n_way, k_shot=args.k_spt, k_query=args.k_qry,
                       batchsz=100, args=args, adjs=dgl_graph, h=args.h))
    print('------ Start Training ------')

    acc_total = []
    F1_total_sum = []
    Recall_total_sum = []
    Precision_total_sum = []
    Sensitivity_total_sum = []
    Specificity_total_sum = []
    MCC_total_sum = []

    total_start = time.time()

    for k_fold in range(10):
        print('------ ', k_fold + 1, 'fold ------')
        if k_fold != 0:
            maml = MAML(args, config).to(device)
        s_start = time.time()
        max_memory = 0
        for epoch in range(args.epoch):
            db = DataLoader(db_train[k_fold], args.task_num, shuffle=True, num_workers=args.num_workers,
                            pin_memory=True, collate_fn=collate)
            s_f = time.time()
            for step, (x_spt, y_spt, x_qry, y_qry, c_spt, c_qry, n_spt, n_qry, g_spt, g_qry) in enumerate(db):
                nodes_len = 0
                if step >= 1:
                    data_loading_time = time.time() - s_r
                else:
                    data_loading_time = time.time() - s_f
                s = time.time()
                # x_spt: a list of #task_num tasks, where each task is a mini-batch of k-shot * n_way localgraphs
                # y_spt: a list of #task_num lists of labels. Each list is of length k-shot * n_way int.
                nodes_len += sum([sum([len(j) for j in i]) for i in n_spt])
                accs = maml(x_spt, y_spt, x_qry, y_qry, c_spt, c_qry, n_spt, n_qry, g_spt, g_qry, feat)
                max_memory = max(max_memory, float(psutil.virtual_memory().used / (1024 ** 3)))
                if step % args.train_result_report_steps == 0:
                    print('Epoch:', epoch + 1, ' Step:', step, ' training acc:', str(accs[-1].tolist())[:5],
                          ' time elapsed:', str(time.time() - s)[:5], ' data loading takes:',
                          str(data_loading_time)[:5], ' Memory usage:',
                          str(float(psutil.virtual_memory().used / (1024 ** 3)))[:5])
                s_r = time.time()

            # validation per epoch
            db_v = DataLoader(db_val[k_fold], 1, shuffle=True, num_workers=args.num_workers, pin_memory=True,
                              collate_fn=collate)
            accs_all_test = []

            for x_spt, y_spt, x_qry, y_qry, c_spt, c_qry, n_spt, n_qry, g_spt, g_qry in db_v:
                accs, _, _, _ = maml.finetunning(x_spt, y_spt, x_qry, y_qry, c_spt, c_qry, n_spt, n_qry, g_spt, g_qry,
                                                 feat)
                accs_all_test.append(accs)

            accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
            print('Epoch:', epoch + 1, ' Val acc:', str(accs[-1])[:5])
            if accs[-1] > max_acc:
                max_acc = accs[-1]

        db_t = DataLoader(db_test[k_fold], 1, shuffle=True, num_workers=args.num_workers, pin_memory=True,
                          collate_fn=collate)

        accs_all_test = []
        perform_para_all = []
        precision_all = []
        recall_all = []
        f1 = []
        specificity = []
        mcc = []

        test_start = time.time()

        for x_spt, y_spt, x_qry, y_qry, c_spt, c_qry, n_spt, n_qry, g_spt, g_qry in db_t:
            accs, per_para_i, taget_label, pre_label = maml.finetunning(x_spt, y_spt, x_qry, y_qry, c_spt, c_qry, n_spt,
                                                                        n_qry, g_spt, g_qry, feat)
            accs_all_test.append(accs)
            perform_para_all.append(per_para_i)

        accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
        perform_para = np.array(perform_para_all).mean(axis=0).astype(np.float32)

        test_end = str(time.time() - test_start)[:5]

        for i in range(labels_num):
            # precision
            if perform_para[0][i] + perform_para[1][i] == 0:
                precision_all.append(0)
            else:
                precision_all.append(perform_para[0][i] / (perform_para[0][i] + perform_para[1][i]))
            # recall(sensitivity)
            if perform_para[0][i] + perform_para[2][i] == 0:
                recall_all.append(0)
            else:
                recall_all.append(perform_para[0][i] / (perform_para[0][i] + perform_para[2][i]))
            # f1
            if precision_all[i] + recall_all[i] == 0:
                f1.append(0)
            else:
                f1.append((2 * precision_all[i] * recall_all[i]) / (precision_all[i] + recall_all[i]))
            # specificity
            if perform_para[3][i] + perform_para[1][i] == 0:
                specificity.append(0)
            else:
                specificity.append(perform_para[3][i] / (perform_para[3][i] + perform_para[1][i]))
            # MCC
            if (perform_para[0][i] + perform_para[1][i]) == 0 or (perform_para[0][i] + perform_para[2][i]) == 0 \
                    or (perform_para[3][i] + perform_para[1][i]) == 0 or (perform_para[3][i] + perform_para[2][i]) == 0:
                mcc.append(0)
            else:
                mcc.append((perform_para[0][i] * perform_para[3][i] - perform_para[1][i] * perform_para[2][i]) /
                           ((perform_para[0][i] + perform_para[1][i]) * (perform_para[0][i] + perform_para[2][i]) * (
                                   perform_para[3][i] + perform_para[1][i]) * (
                                    perform_para[3][i] + perform_para[2][i])) ** 0.5)

        precision = np.array(precision_all).mean()
        recall = np.array(recall_all).mean()
        f1 = np.array(f1).mean()
        sensitivity = recall_all

        print('Test acc:', str(accs[1])[:5])
        print('Test Precision:', str(precision)[:5])
        print('Test Recall:', str(recall)[:5])
        print('Test F1:', str(f1)[:5])
        print('Test Sensitivity:', sensitivity)
        print('Test Specificity:', specificity)
        print('Test MCC:', mcc)
        print('Test Time:', test_end)

        with open(root + 'results.txt', 'a') as f:
            f.write('------' + str(k_fold + 1) + 'fold ------\n')
            f.write('Test acc:' + str(accs[1])[:5] + '\n')
            f.write('Test Precision:' + str(precision)[:5] + '\n')
            f.write('Test Recall:' + str(recall)[:5] + '\n')
            f.write('Test F1:' + str(f1)[:5] + '\n')
            f.write('Test Sensitivity:' + str(sensitivity) + '\n')
            f.write('Test Specificity:' + str(specificity) + '\n')
            f.write('Test MCC:' + str(mcc) + '\n')
            f.write('Test Max Momory:' + str(max_memory)[:5] + '\n')
            f.write('Test Time:' + test_end + '\n')
            f.write('One Epoch Time:' + str(time.time() - s_start)[:5] + '\n')

        acc_total.append(float(str(accs[1])[:5]))
        F1_total_sum.append(float(str(f1)[:5]))
        Recall_total_sum.append(float(str(recall)[:5]))
        Precision_total_sum.append(float(str(precision)[:5]))
        Sensitivity_total_sum.append(sensitivity)
        Specificity_total_sum.append(specificity)
        MCC_total_sum.append(mcc)

        print('One Epoch Time:', str(time.time() - s_start)[:5])
        print('Max Momory:', str(max_memory)[:5])

    print('Total Acc:', str(np.array(acc_total).mean())[:5])
    print('Total Precision:', str(np.array(Precision_total_sum).mean())[:5])
    print('Total Recall:', str(np.array(Recall_total_sum).mean())[:5])
    print('Total F1:', str(np.array(F1_total_sum).mean())[:5])
    print('Total Sensitivity:', str(np.array(Sensitivity_total_sum).mean(axis=0).astype(np.float32)))
    print('Total Specificity:', str(np.array(Specificity_total_sum).mean(axis=0).astype(np.float32)))
    print('Total MCC:', str(np.array(MCC_total_sum).mean(axis=0).astype(np.float32)))
    print('Total Time:', str(time.time() - total_start)[:5])

    with open(root + 'results.txt', 'a') as f:
        f.write('------ Total Result ------\n')
        f.write('Total Acc:' + str(np.array(acc_total).mean())[:5] + '\n')
        f.write('Total Precision:' + str(np.array(Precision_total_sum).mean())[:5] + '\n')
        f.write('Total Recall:' + str(np.array(Recall_total_sum).mean())[:5] + '\n')
        f.write('Total F1:' + str(np.array(F1_total_sum).mean())[:5] + '\n')
        f.write('Total Sensitivity:' + str(np.array(Sensitivity_total_sum).mean(axis=0).astype(np.float32)) + '\n')
        f.write('Total Specificity:' + str(np.array(Specificity_total_sum).mean(axis=0).astype(np.float32)) + '\n')
        f.write('Total MCC:' + str(np.array(MCC_total_sum).mean(axis=0).astype(np.float32)) + '\n')
        f.write('Total Time:' + str(time.time() - total_start)[:5] + '\n')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=15)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=5)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=10)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=4)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=1e-3)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
    argparser.add_argument('--input_dim', type=int, help='input feature dim', default=1)
    argparser.add_argument('--hidden_dim', type=int, help='hidden dim', default=256)
    argparser.add_argument("--data_dir", default='DATA_PATH/', type=str, required=False, help="The input data dir.")
    argparser.add_argument("--val_result_report_steps", default=100, type=int, required=False, help="validation report")
    argparser.add_argument("--train_result_report_steps", default=100, type=int, required=False, help="training report")
    argparser.add_argument("--num_workers", default=0, type=int, required=False, help="num of workers")
    argparser.add_argument("--batchsz", default=500, type=int, required=False, help="batch size")
    argparser.add_argument("--h", default=1, type=int, required=False, help="neighborhood size")
    argparser.add_argument('--sample_nodes', type=int, help='sample nodes if above this number of nodes', default=1000)

    args = argparser.parse_args()

    main()
