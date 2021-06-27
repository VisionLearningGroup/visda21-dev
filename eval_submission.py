import sys
import numpy as np
from sklearn.metrics import roc_auc_score


def return_label_list(file):
    tmp = open(file, "r")
    lines = tmp.readlines()
    labels = [int(line.strip().split(" ")[1]) for line in lines]
    file_name = [line.strip().split(" ")[0] for line in lines]
    return labels, file_name


def return_result_list(file):
    tmp = open(file, "r")
    lines = tmp.readlines()
    labels = [int(line.strip().split(" ")[1]) for line in lines]
    file_name = [line.strip().split(" ")[0] for line in lines]
    anomal_lists = [float(line.strip().split(" ")[2]) for line in lines]
    return labels, file_name, anomal_lists


def check_file_name(file1, file2):
    for i, fil_1 in enumerate(file1):
        fil_2 = file2[i]
        if fil_2 not in fil_1:
            print(fil_1, fil_2)
            return False
    return True


def acc_perclass(gt, submit, output="result.txt", class_known=1000):
    ## gt file has the same format as files under val_filelists.
    ## submission file needs to have the following format,
    ## [filename] [prediction of closed class index] [anomaly score]
    ## Note that higher anomaly score means the sample is more likely an outlier.

    gt_labels, file_gt = return_label_list(gt)
    submit_labels, file_sb, anomal_sb = return_result_list(submit)
    try:
        assert len(gt_labels) == len(submit_labels)
    except:
        raise Exception('Number of submitted files and GT is different!')
    try:
        assert check_file_name(file_gt, file_sb)
    except:
        raise Exception('Submitted files do not correpond to GT files!')

    gt_labels = np.array(gt_labels)
    submit_labels = np.array(submit_labels)
    anomal_sb = np.array(anomal_sb)
    ind_known = np.where(gt_labels<class_known)[0]
    ind_unknown = np.where(gt_labels>=class_known)[0]
    close_pred = submit_labels[ind_known]
    close_gt = gt_labels[ind_known]
    acc = 100 * float((close_pred == close_gt).sum() / len(ind_known))
    anomal_known = anomal_sb[ind_known]
    anomal_unknown = anomal_sb[ind_unknown]
    all_score = np.r_[anomal_known, anomal_unknown]
    label_roc = np.zeros(len(all_score))
    label_roc[len(anomal_known):] = 1
    auroc = roc_auc_score(label_roc, all_score)

    print("ACC All %f AUROC %f" % (acc, auroc))
    with open(output, "w") as out_line:
        out_line.write("ACC All %f AUROC %f" % (acc, auroc))

gt_file = sys.argv[1]
submit_file = sys.argv[2]
acc_perclass(gt_file, submit_file)
