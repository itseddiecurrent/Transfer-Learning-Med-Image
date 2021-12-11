import numpy as np
from skimage.metrics import adapted_rand_error
import skimage.color
import sklearn.metrics
import skimage.io
import os
import matplotlib.pyplot as plt


def compute_metrics(real_images, fake_images, dataset='binary'):
    metrics = dict()
    prec_list, rec_list, f1_list, iou_list, acc_list = [],[],[],[],[]
    for real, fake in zip(real_images, fake_images):
        if dataset == 'binary':
            truth = np.ceil(skimage.color.rgb2gray(skimage.io.imread(real))).astype('int')
            comp = np.ceil(skimage.color.rgb2gray(skimage.io.imread(fake))).astype('int')
        elif dataset == 'multiclass': 
            truth = skimage.io.imread(real)
            comp = skimage.io.imread(fake)
        plt.subplot(1,3,1)
        plt.imshow(truth[:,:,0])
        plt.subplot(1,3,2)
        plt.imshow(comp[:,:,0])
        plt.show()
        prec_list.append(sklearn.metrics.precision_score(truth.flatten(), comp.flatten(), pos_label=1, average='binary'))
        rec_list.append(sklearn.metrics.recall_score(truth.flatten(), comp.flatten(), pos_label=1, average='binary'))
        f1_list.append(sklearn.metrics.f1_score(truth.flatten(), comp.flatten()))
        iou_list.append(sklearn.metrics.jaccard_score(truth.flatten(), comp.flatten()))
        acc_list.append(sklearn.metrics.accuracy_score(truth.flatten(), comp.flatten()))
        #arr, prec, rec = skimage.metrics.adapted_rand_error(truth, comp)
    metrics['precision'] = np.mean(prec_list)
    metrics['recall'] = np.mean(rec_list)
    metrics['F1 score'] = np.mean(f1_list)
    metrics['iou'] = np.mean(iou_list)
    metrics['acc_list'] = np.mean(acc_list)
    with open("results_file.csv", mode='w') as res_file:
        for key, val in zip(metrics.keys(), metrics.values()):
            print(key, " ---> ", val)
            res_file.write(str(key)+','+str(val)+'\n')

def load_images(folder_path):
    real, fake = [], []
    for root, dir, files in os.walk(folder_path):
        for file in files:
            if "_fake_B" in file:
                fake.append(os.path.join(folder_path, file))
            if "_real_B" in file:
                real.append(os.path.join(folder_path, file))
    ''''sanity check
    for a, b in zip(real, fake):
        print("a --->", a, 'b-->', b)'''
    return real, fake

real, fake = load_images(folder_path='results/facades_before_finetune')
compute_metrics(real, fake, dataset='multiclass')


