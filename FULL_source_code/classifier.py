import numpy as np
import pandas as pd
import os
import pywt
import sys
import warnings
from IPython.display import Image
from glob import glob
from scipy.ndimage import filters
from multiprocessing import cpu_count, Pool
from PIL import Image
from numpy.fft import fft2, ifft2
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.insert(0, '/content/drive/MyDrive/WIP/stylegan3_prnu-based_detection/src')
import prnu
warnings.simplefilter(action='ignore', category=FutureWarning)

def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.gray_r):
    plt.clf()
    plt.matshow(df_confusion, cmap=cmap) # imshow
    #plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    #plt.tight_layout()
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)
    #print(df_confusion.stats()['overall']['Accuracy'])
    plt.show()


if __name__ == "__main__":
    ###
    # ff_dirlist = GAN Images
    # nat_dirlist = Natural Face Images
    ###

    # Set number of images to load
    num_of_imgs = 490
    num_of_test = 20
    #490

    ans = input('Type "s" to compute data from images and save it, anything to retrieve data from previously saved file: ')
    
    if ans == "s":
        print('Loading sample set')
        ff_dirlist = np.array(sorted(glob(r'/content/drive/MyDrive/WIP/images/stylegan3_generated/*.png')))[:num_of_imgs]
        print(ff_dirlist)

        print('Loading reference image')
        nat_dirlist = np.array(sorted(glob(r'/content/drive/MyDrive/WIP/images/original/*.png')))[:num_of_imgs]
        print(nat_dirlist)

        print('Loading reference image')
        test_dirlist = np.array(sorted(glob(r'/content/drive/MyDrive/WIP/images/classificatore/real/*.png')))[:num_of_test]
        print(test_dirlist)

        print('Loading reference image')
        testf_dirlist = np.array(sorted(glob(r'/content/drive/MyDrive/WIP/images/classificatore/fake/*.png')))[:num_of_test]
        print(testf_dirlist)

        print('Computing fingerprints for ff')
        ff = []
        imgs = []
        for img_path in ff_dirlist:
            im = Image.open(img_path)
            im_arr = np.asarray(im)
            if im_arr.dtype != np.uint8:
                print('Error while reading image: {}'.format(img_path))
                continue
            if im_arr.ndim != 3:
                print('Image is not RGB: {}'.format(img_path))
                continue

            im_cut2 = prnu.cut_ctr(im_arr, (512, 512, 3))
            imgs += [im_cut2]
        ff += [prnu.extract_multiple_aligned(imgs, processes=1)]

        print(len(ff))
    
        ff = np.stack(ff, 0)
        print(ff.shape)

        print('Computing fingerprints for nat')
        nat = []
        imgss = []
        for img_path in nat_dirlist:
            im = Image.open(img_path)
            im_arr = np.asarray(im)
            if im_arr.dtype != np.uint8:
                print('Error while reading image: {}'.format(img_path))
                continue
            if im_arr.ndim != 3:
                print('Image is not RGB: {}'.format(img_path))
                continue

            im_cut1 = prnu.cut_ctr(im_arr, (512, 512, 3))
            imgss += [im_cut1]
        nat += [prnu.extract_multiple_aligned(imgss, processes=1)]

        nat = np.stack(nat, 0)
        print(nat.shape)

    
        print('Computing fingerprints for reference')
        test = []
        imgss = []
        for img_path in test_dirlist:
            im = Image.open(img_path)
            im_arr = np.asarray(im)
            if im_arr.dtype != np.uint8:
                print('Error while reading image: {}'.format(img_path))
                continue
            if im_arr.ndim != 3:
                print('Image is not RGB: {}'.format(img_path))
                continue

            im_cut3 = prnu.cut_ctr(im_arr, (512, 512, 3))
            imgss += [im_cut3]
        test += [prnu.extract_multiple_aligned(imgss, processes=1)]

        test = np.stack(test, 0)
        print(test.shape)

        print('Computing fingerprints for fake reference')
        testf = []
        imgss = []
        for img_path in testf_dirlist:
            im = Image.open(img_path)
            im_arr = np.asarray(im)
            if im_arr.dtype != np.uint8:
                print('Error while reading image: {}'.format(img_path))
                continue
            if im_arr.ndim != 3:
                print('Image is not RGB: {}'.format(img_path))
                continue

            im_cut4 = prnu.cut_ctr(im_arr, (512, 512, 3))
            imgss += [im_cut4]
        testf += [prnu.extract_multiple_aligned(imgss, processes=1)]

        testf = np.stack(testf, 0)
        print(testf.shape)

        print('\nComputing residuals nat')

        imgs = []
        for img_path in nat_dirlist:
            imgs += [prnu.cut_ctr(np.asarray(Image.open(img_path)), (512, 512, 3))]

        pool = Pool(cpu_count())
        nat_w = pool.map(prnu.extract_single, imgs)
        pool.close()

        nat_w = np.stack(nat_w, 0)
        print(nat_w.shape)

        print('Computing residuals ff')

        imgs = []
        for img_path in ff_dirlist:
            imgs += [prnu.cut_ctr(np.asarray(Image.open(img_path)), (512, 512, 3))]

        pool = Pool(cpu_count())
        ff_w = pool.map(prnu.extract_single, imgs)
        pool.close()

        ff_w = np.stack(ff_w, 0)
        print(ff_w.shape)

        print('Computing residuals test')

        imgs = []
        for img_path in test_dirlist:
            imgs += [prnu.cut_ctr(np.asarray(Image.open(img_path)), (512, 512, 3))]

        pool = Pool(cpu_count())
        test_w = pool.map(prnu.extract_single, imgs)
        pool.close()

        test_w = np.stack(test_w, 0)
        print(test_w.shape)


        print('Computing residuals testf')

        imgs = []
        for img_path in testf_dirlist:
            imgs += [prnu.cut_ctr(np.asarray(Image.open(img_path)), (512, 512, 3))]

        pool = Pool(cpu_count())
        testf_w = pool.map(prnu.extract_single, imgs)
        pool.close()

        testf_w = np.stack(testf_w, 0)
        print(testf_w.shape)


        # Creating correlations
        test_ff_corr = pd.DataFrame()
        test_nat_corr = pd.DataFrame()
        testf_ff_corr = pd.DataFrame()
        testf_nat_corr = pd.DataFrame()

        # Correlation test_ff
        print('\nComputing correlation')
        test_ff_corr = pd.DataFrame(prnu.aligned_cc(ff, test_w[[0]])['ncc'])
        test_ff_corr.columns = ["Corr Value"]
        for i in range(num_of_test):
            b = pd.DataFrame(prnu.aligned_cc(ff, test_w[[i]])['ncc'])
            test_ff_corr = test_ff_corr.append(b)

        test_ff_corr = test_ff_corr.iloc[1:]
        test_ff_corr.columns = ["Corr_Value", "Sıl"]
        test_ff_corr.drop("Sıl", inplace=True, axis=1)
        test_ff_corr.Corr_Value.hist()

        # Correlation test_nat
        print('\nComputing correlation')
        test_nat_corr = pd.DataFrame(prnu.aligned_cc(nat, test_w[[0]])['ncc'])
        test_nat_corr.columns = ["Corr Value"]
        for i in range(num_of_test):
            b = pd.DataFrame(prnu.aligned_cc(nat, test_w[[i]])['ncc'])
            test_nat_corr = test_nat_corr.append(b)

        test_nat_corr = test_nat_corr.iloc[1:]
        test_nat_corr.columns = ["Corr_Value", "Sıl"]
        test_nat_corr.drop("Sıl", inplace=True, axis=1)
        test_nat_corr.Corr_Value.hist()

        # Correlation testf_ff
        print('\nComputing correlation')
        testf_ff_corr = pd.DataFrame(prnu.aligned_cc(ff, testf_w[[0]])['ncc'])
        testf_ff_corr.columns = ["Corr Value"]
        for i in range(num_of_test):
            b = pd.DataFrame(prnu.aligned_cc(ff, testf_w[[i]])['ncc'])
            testf_ff_corr = testf_ff_corr.append(b)

        testf_ff_corr = testf_ff_corr.iloc[1:]
        testf_ff_corr.columns = ["Corr_Value", "Sıl"]
        testf_ff_corr.drop("Sıl", inplace=True, axis=1)
        testf_ff_corr.Corr_Value.hist()

        # Correlation testf_nat
        print('\nComputing correlation')
        testf_nat_corr = pd.DataFrame(prnu.aligned_cc(nat, testf_w[[0]])['ncc'])
        testf_nat_corr.columns = ["Corr Value"]
        for i in range(num_of_test):
            b = pd.DataFrame(prnu.aligned_cc(nat, testf_w[[i]])['ncc'])
            testf_nat_corr = testf_nat_corr.append(b)

        testf_nat_corr = testf_nat_corr.iloc[1:]
        testf_nat_corr.columns = ["Corr_Value", "Sıl"]
        testf_nat_corr.drop("Sıl", inplace=True, axis=1)
        testf_nat_corr.Corr_Value.hist()

        print("Saving Data")

        print("-------------------------------------------------")
        print(test_ff_corr)
        print("-------------------------------------------------")
        print(test_nat_corr)
        print("-------------------------------------------------")
        print(testf_ff_corr)
        print("-------------------------------------------------")
        print(testf_nat_corr)

        np.savetxt('test_ff_corr_out.txt',test_ff_corr)
        np.savetxt('test_nat_corr_out.txt',test_nat_corr)
        np.savetxt('testf_ff_corr_out.txt',testf_ff_corr)
        np.savetxt('testf_nat_corr_out.txt',testf_nat_corr)
    else:
        test_ff_corr_data = '/content/drive/MyDrive/WIP/stylegan3_prnu-based_detection/test_ff_corr_out.txt'
        test_nat_corr_data = '/content/drive/MyDrive/WIP/stylegan3_prnu-based_detection/test_nat_corr_out.txt'
        testf_ff_corr_data = '/content/drive/MyDrive/WIP/stylegan3_prnu-based_detection/testf_ff_corr_out.txt'
        testf_nat_corr_data = '/content/drive/MyDrive/WIP/stylegan3_prnu-based_detection/testf_nat_corr_out.txt'
        
        test_ff_corr = pd.DataFrame(data = np.loadtxt(test_ff_corr_data))
        test_nat_corr = pd.DataFrame(data = np.loadtxt(test_nat_corr_data))
        testf_ff_corr = pd.DataFrame(data = np.loadtxt(testf_ff_corr_data))
        testf_nat_corr = pd.DataFrame(data = np.loadtxt(testf_nat_corr_data))

    zeros = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ones = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    actual = []
    predicted = []

    THRESHOLD = 0.0025

    print("TEST_FF: ")
    pred_ff = []
    print(test_ff_corr)
    for index, row in test_ff_corr.iterrows():
      if float(row["Corr_Value"]) < THRESHOLD:
        pred_ff.append(0)
      else: 
        pred_ff.append(1) 
    print(pred_ff)

    actual += zeros
    predicted += pred_ff

    print("TEST_NAT: ")
    pred_nat = []
    print(test_nat_corr)
    for index, row in test_nat_corr.iterrows():
      if float(row["Corr_Value"]) < THRESHOLD:
        pred_nat.append(0)
      else: 
        pred_nat.append(1) 
    print(pred_nat)

    actual += ones
    predicted += pred_nat


    print("TESTF_FF: ")
    pred_ff = []
    print(testf_ff_corr)
    for index, row in testf_ff_corr.iterrows():
      if float(row["Corr_Value"]) < THRESHOLD:
        pred_ff.append(0)
      else: 
        pred_ff.append(1) 
    print(pred_ff)

    actual += ones
    predicted += pred_ff

    print("TESTF_NAT: ")
    pred_nat = []
    print(testf_nat_corr)
    for index, row in testf_nat_corr.iterrows():
      if float(row["Corr_Value"]) < THRESHOLD:
        pred_nat.append(0)
      else: 
        pred_nat.append(1) 
    print(pred_nat)

    actual += zeros
    predicted += pred_nat


    y_actu = pd.Series(actual, name='Actual')
    y_pred = pd.Series(predicted, name='Predicted')

    df_confusion = pd.crosstab(y_actu, y_pred)
    plot_confusion_matrix(df_confusion)


    import seaborn as sns
    plt.clf()

    sns.distplot(test_ff_corr, label="TEST FF", color="red", hist_kws=dict(alpha=1), norm_hist=True, bins=80)
    plt.legend()
    plt.show()

    plt.clf()

    sns.distplot(test_nat_corr, label="TEST NAT", color="darkcyan", hist_kws=dict(alpha=1), norm_hist=True, bins=80)
    plt.legend()
    plt.show()

    plt.clf()

    sns.distplot(testf_ff_corr, label="TESTF FF", color="green", hist_kws=dict(alpha=1), norm_hist=True, bins=80)
    plt.legend()
    plt.show()

    plt.clf()

    sns.distplot(testf_nat_corr, label="TESTF NAT", color="orange", hist_kws=dict(alpha=1), norm_hist=True, bins=80)
    plt.legend()
    plt.show()

