import numpy as np
import pandas as pd
import os
import pywt
import prnu
import matplotlib.pyplot as plt
from IPython.display import Image
from glob import glob
from scipy.ndimage import filters
from multiprocessing import cpu_count, Pool
from PIL import Image
from numpy.fft import fft2, ifft2
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm


if __name__ == "__main__":
    """ff_dirlist = GAN Images
        nat_dirlist = Natural Face Images
        """

    print('Loading StyleGAN images')
    ff_dirlist = np.array(sorted(glob(r'..\assets\stylegan3_generated\*.PNG')))[:10]
    ff_device = np.array([os.path.split(i)[1].rsplit('0', 1)[0] for i in ff_dirlist])[:10]
    print('Done!')

    print('Loading natural images')
    nat_dirlist = np.array(sorted(glob(r'..\assets\original\train\cat\*.PNG')))[:10]
    nat_device = np.array([os.path.split(i)[1].split('0', 1)[0] for i in nat_dirlist])[:10]
    print('Done!')

    print('Computing fingerprints for StyleGAN')
    fingerprint_stylegan_psi1 = sorted(np.unique(ff_device))
    ff = []
    for device in fingerprint_stylegan_psi1:
        imgs = []
        for img_path in ff_dirlist[ff_device == device]:
            im = Image.open(img_path)
            # im = im.resize((2000, 3008), Image.ANTIALIAS)
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

    ff = np.stack(ff, 0)
    print(ff.shape)

    print('Computing fingerprints for natural')
    normal_device = sorted(np.unique(nat_device))
    nat = []
    for device in normal_device:
        imgss = []
        for img_path in nat_dirlist[nat_device == device]:
            im = Image.open(img_path)
            # im = im.resize((2000, 3008), Image.ANTIALIAS)
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

    print('Computing residuals nat_dirlist')

    imgs = []
    for img_path in nat_dirlist:
        imgs += [prnu.cut_ctr(np.asarray(Image.open(img_path)), (512, 512, 3))]

    pool = Pool(cpu_count())
    nat_w = pool.map(prnu.extract_single, imgs)
    pool.close()

    nat_w = np.stack(nat_w, 0)
    print(nat_w.shape)

    print('Computing residuals ff_dirlist')

    imgs = []
    for img_path in ff_dirlist:
        imgs += [prnu.cut_ctr(np.asarray(Image.open(img_path)), (512, 512, 3))]

    pool = Pool(cpu_count())
    ff_w = pool.map(prnu.extract_single, imgs)
    pool.close()

    ff_w = np.stack(ff_w, 0)
    print(ff_w.shape)

    '''Creating correlations'''
    nat_ff_corr = pd.DataFrame()
    nat_nat_corr = pd.DataFrame()
    ff_ff_corr = pd.DataFrame()

    '''Correlation fake - fake'''
    print('Computing correlation fake - fake')
    ff_ff_corr = pd.DataFrame(prnu.aligned_cc(ff, ff_w[[0]])['ncc'])
    ff_ff_corr.columns = ["Corr Value"]
    for i in range(500):
        b = pd.DataFrame(prnu.aligned_cc(ff, ff_w[[i]])['ncc'])
        ff_ff_corr = ff_ff_corr.append(b)

    ff_ff_corr = ff_ff_corr.iloc[1:]
    ff_ff_corr.columns = ["Corr_Value", "Sıl"]
    ff_ff_corr.drop("Sıl", inplace=True, axis=1)
    ff_ff_corr.Corr_Value.hist()
    print("Correlation mean of Natural - Natural Pic", ff_ff_corr.Corr_Value.mean())

    '''Correlation natural - natural'''
    print('Computing correlation natural - natural')
    nat_nat_corr = pd.DataFrame(prnu.aligned_cc(nat, nat_w[[0]])['ncc'])
    nat_nat_corr.columns = ["Corr Value"]
    for i in range(500):
        b = pd.DataFrame(prnu.aligned_cc(nat, nat_w[[i]])['ncc'])
        nat_nat_corr = nat_nat_corr.append(b)

    nat_nat_corr = nat_nat_corr.iloc[1:]
    nat_nat_corr.columns = ["Corr_Value", "Sıl"]
    nat_nat_corr.drop("Sıl", inplace=True, axis=1)
    nat_nat_corr.Corr_Value.hist()
    print("Correlation mean of Natural - Natural Pic", nat_nat_corr.Corr_Value.mean())

    '''Correlation natural - fake'''
    print('Computing correlation natural - fake')
    nat_ff_corr = pd.DataFrame(prnu.aligned_cc(ff, nat_w[[0]])['ncc'])
    nat_ff_corr.columns = ["Corr Value"]
    for i in range(500):
        b = pd.DataFrame(prnu.aligned_cc(ff, nat_w[[i]])['ncc'])
        nat_ff_corr = nat_ff_corr.append(b)

    nat_ff_corr = nat_ff_corr.iloc[1:]
    nat_ff_corr.columns = ["Corr_Value", "Sıl"]
    nat_ff_corr.drop("Sıl", inplace=True, axis=1)
    nat_ff_corr.Corr_Value.hist()
    print("Correlation mean of Natural - Fake Pic", nat_ff_corr.Corr_Value.mean())


def threshold(wlet_coeff_energy_avg: np.ndarray, noise_var: float) -> np.ndarray:
    """
    Noise variance theshold as from Binghamton toolbox.
    :param wlet_coeff_energy_avg:
    :param noise_var:
    :return: noise variance threshold
    """
    res = wlet_coeff_energy_avg - noise_var
    return (res + np.abs(res)) / 2


def wiener_adaptive(x: np.ndarray, noise_var: float, **kwargs) -> np.ndarray:
    """
    WaveNoise as from Binghamton toolbox.
    Wiener adaptive flter aimed at extracting the noise component
    For each input pixel the average variance over a neighborhoods of different window sizes is first computed.
    The smaller average variance is taken into account when filtering according to Wiener.
    :param x: 2D matrix
    :param noise_var: Power spectral density of the noise we wish to extract (S)
    :param window_size_list: list of window sizes
    :return: wiener filtered version of input x
    """
    window_size_list = list(kwargs.pop('window_size_list', [3, 5, 7, 9]))

    energy = x ** 2

    avg_win_energy = np.zeros(x.shape + (len(window_size_list),))
    for window_idx, window_size in enumerate(window_size_list):
        avg_win_energy[:, :, window_idx] = filters.uniform_filter(energy,
                                                                  window_size,
                                                                  mode='constant')

    coef_var = threshold(avg_win_energy, noise_var)
    coef_var_min = np.min(coef_var, axis=2)

    x = x * noise_var / (coef_var_min + noise_var)

    return x


def pce(cc: np.ndarray, neigh_radius: int = 2) -> dict:
    """
    PCE position and value
    :param cc: as from crosscorr2d
    :param neigh_radius: radius around the peak to be ignored while computing floor energy
    :return: {'peak':(y,x), 'pce': peak to floor ratio, 'cc': cross-correlation value at peak position
    """
    assert (cc.ndim == 2)
    assert (isinstance(neigh_radius, int))

    out = dict()

    max_idx = np.argmax(cc.flatten())
    max_y, max_x = np.unravel_index(max_idx, cc.shape)

    peak_height = cc[max_y, max_x]

    cc_nopeaks = cc.copy()
    cc_nopeaks[max_y - neigh_radius:max_y + neigh_radius, max_x - neigh_radius:max_x + neigh_radius] = 0

    pce_energy = np.mean(cc_nopeaks.flatten() ** 2)

    out['peak'] = (max_y, max_x)
    out['pce'] = (peak_height ** 2) / pce_energy * np.sign(peak_height)
    out['cc'] = peak_height

    return out


def noise_extract(im: np.ndarray, levels: int = 4, sigma: float = 5) -> np.ndarray:
    """
    NoiseExtract as from Binghamton toolbox.
    :param im: grayscale or color image, np.uint8
    :param levels: number of wavelet decomposition levels
    :param sigma: estimated noise power
    :return: noise residual
    """

    assert (im.dtype == np.uint8)
    assert (im.ndim in [2, 3])

    im = im.astype(np.float32)

    noise_var = sigma ** 2

    if im.ndim == 2:
        im.shape += (1,)

    W = np.zeros(im.shape, np.float32)

    for ch in range(im.shape[2]):

        wlet = None
        while wlet is None and levels > 0:
            try:
                wlet = pywt.wavedec2(im[:, :, ch], 'db4', level=levels)
            except ValueError:
                levels -= 1
                wlet = None
        if wlet is None:
            raise ValueError('Impossible to compute Wavelet filtering for input size: {}'.format(im.shape))

        wlet_details = wlet[1:]

        wlet_details_filter = [None] * len(wlet_details)
        # Cycle over Wavelet levels 1:levels-1
        for wlet_level_idx, wlet_level in enumerate(wlet_details):
            # Cycle over H,V,D components
            level_coeff_filt = [None] * 3
            for wlet_coeff_idx, wlet_coeff in enumerate(wlet_level):
                level_coeff_filt[wlet_coeff_idx] = wiener_adaptive(wlet_coeff, noise_var)
            wlet_details_filter[wlet_level_idx] = tuple(level_coeff_filt)

        # Set filtered detail coefficients for Levels > 0 ---
        wlet[1:] = wlet_details_filter

        # Set to 0 all Level 0 approximation coefficients ---
        wlet[0][...] = 0

        # Invert wavelet transform ---
        wrec = pywt.waverec2(wlet, 'db4')
        try:
            W[:, :, ch] = wrec
        except ValueError:
            W = np.zeros(wrec.shape[:2] + (im.shape[2],), np.float32)
            W[:, :, ch] = wrec

    if W.shape[2] == 1:
        W.shape = W.shape[:2]

    W = W[:im.shape[0], :im.shape[1]]

    return W


def wiener_dft(im: np.ndarray, sigma: float) -> np.ndarray:
    """
    Adaptive Wiener filter applied to the 2D FFT of the image
    :param im: multidimensional array
    :param sigma: estimated noise power
    :return: filtered version of input im
    """
    noise_var = sigma ** 2
    h, w = im.shape

    im_noise_fft = fft2(im)
    im_noise_fft_mag = np.abs(im_noise_fft / (h * w) ** .5)

    im_noise_fft_mag_noise = wiener_adaptive(im_noise_fft_mag, noise_var)

    zeros_y, zeros_x = np.nonzero(im_noise_fft_mag == 0)

    im_noise_fft_mag[zeros_y, zeros_x] = 1
    im_noise_fft_mag_noise[zeros_y, zeros_x] = 0

    im_noise_fft_filt = im_noise_fft * im_noise_fft_mag_noise / im_noise_fft_mag
    im_noise_filt = np.real(ifft2(im_noise_fft_filt))

    return im_noise_filt.astype(np.float32)


def crosscorr_2d(k1: np.ndarray, k2: np.ndarray) -> np.ndarray:
    """
    PRNU 2D cross-correlation
    :param k1: 2D matrix of size (h1,w1)
    :param k2: 2D matrix of size (h2,w2)
    :return: 2D matrix of size (max(h1,h2),max(w1,w2))
    """
    assert (k1.ndim == 2)
    assert (k2.ndim == 2)

    max_height = max(k1.shape[0], k2.shape[0])
    max_width = max(k1.shape[1], k2.shape[1])

    k1 -= k1.flatten().mean()
    k2 -= k2.flatten().mean()

    k1 = np.pad(k1, [(0, max_height - k1.shape[0]), (0, max_width - k1.shape[1])], mode='constant', constant_values=0)
    k2 = np.pad(k2, [(0, max_height - k2.shape[0]), (0, max_width - k2.shape[1])], mode='constant', constant_values=0)

    k1_fft = fft2(k1, )
    k2_fft = fft2(np.rot90(k2, 2), )

    return np.real(ifft2(k1_fft * k2_fft)).astype(np.float32)


def rgb2gray(im: np.ndarray) -> np.ndarray:
    """
    RGB to gray as from Binghamton toolbox.
    :param im: multidimensional array
    :return: grayscale version of input im
    """
    rgb2gray_vector = np.asarray([0.29893602, 0.58704307, 0.11402090]).astype(np.float32)
    rgb2gray_vector.shape = (3, 1)

    if im.ndim == 2:
        im_gray = np.copy(im)
    elif im.shape[2] == 1:
        im_gray = np.copy(im[:, :, 0])
    elif im.shape[2] == 3:
        w, h = im.shape[:2]
        im = np.reshape(im, (w * h, 3))
        im_gray = np.dot(im, rgb2gray_vector)
        im_gray.shape = (w, h)
    else:
        raise ValueError('Input image must have 1 or 3 channels')

    return im_gray.astype(np.float32)


def extract_multiple_aligned(imgs: list, levels: int = 4, sigma: float = 5, processes: int = None,
                             batch_size=cpu_count(), tqdm_str: str = '') -> np.ndarray:
    """
    Extract PRNU from a list of images. Images are supposed to be the same size and properly oriented
    :param tqdm_str: tqdm description (see tqdm documentation)
    :param batch_size: number of parallel processed images
    :param processes: number of parallel processes
    :param imgs: list of images of size (H,W,Ch) and type np.uint8
    :param levels: number of wavelet decomposition levels
    :param sigma: estimated noise power
    :return: PRNU
    """
    assert (isinstance(imgs[0], np.ndarray))
    assert (imgs[0].ndim == 3)
    assert (imgs[0].dtype == np.uint8)

    h, w, ch = imgs[0].shape

    RPsum = np.zeros((h, w, ch), np.float32)
    NN = np.zeros((h, w, ch), np.float32)

    if processes is None or processes > 1:
        args_list = []
        for im in imgs:
            args_list += [(im, levels, sigma)]
        pool = Pool(processes=processes)

        for batch_idx0 in tqdm(np.arange(start=0, step=batch_size, stop=len(imgs)), disable=tqdm_str == '',
                               desc=(tqdm_str + ' (1/2)'), dynamic_ncols=True):
            nni = pool.map(prnu.inten_sat_compact, args_list[batch_idx0:batch_idx0 + batch_size])
            for ni in nni:
                NN += ni
            del nni

        for batch_idx0 in tqdm(np.arange(start=0, step=batch_size, stop=len(imgs)), disable=tqdm_str == '',
                               desc=(tqdm_str + ' (2/2)'), dynamic_ncols=True):
            wi_list = pool.map(prnu.noise_extract_compact, args_list[batch_idx0:batch_idx0 + batch_size])
            for wi in wi_list:
                RPsum += wi
            del wi_list

        pool.close()

    else:  # Single process
        for im in tqdm(imgs, disable=tqdm_str is None, desc=tqdm_str, dynamic_ncols=True):
            RPsum += prnu.noise_extract_compact((im, levels, sigma))
            NN += (prnu.inten_scale(im) * prnu.saturation(im)) ** 2

    K = RPsum / (NN + 1)
    K = rgb2gray(K)
    K = prnu.zero_mean_total(K)
    K = wiener_dft(K, K.std(ddof=1)).astype(np.float32)

    return K

