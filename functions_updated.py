import igraph
from loess.loess_1d import loess_1d
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats, interpolate
from scipy.interpolate import interp1d
from sklearn import svm, linear_model
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from supersmoother import SuperSmoother, LinearSmoother
from astropy.modeling.models import Lorentz1D
from loess import loess_1d
import math
from sklearn.metrics import mean_squared_error
import random
from matplotlib.pyplot import Polygon


def load_excelfile(filepath):
    '''

    :param filepath: the absolute/relative path of file
    :return: a panda dataframe and its columns(features) name
    '''

    file = pd.read_excel(filepath, engine='openpyxl')
    return file, list(file.columns)


def load_csvfile(filepath):
    '''

    :param filepath: the absolute/relative path of file
    :return: a panda dataframe and its columns(features) name
    '''

    file = pd.read_csv(filepath, engine='openpyxl')
    return file, list(file.columns)


def load_file(filepath, columns_toUse, header=True):
    data = pd.DataFrame()
    if filepath.endswith(".csv"):
        if header:
            data = pd.read_csv(filepath, usecols=columns_toUse)
        else:
            data = pd.read_csv(filepath, usecols=columns_toUse, header=None)
    elif filepath.endswith(".xlsx"):
        if header:
            data = pd.read_excel(filepath, engine='openpyxl', usecols=columns_toUse)
        else:
            data = pd.read_excel(filepath, engine='openpyxl', usecols=columns_toUse, header=None)

    column_list = data.columns.tolist()
    data = data[columns_toUse]
    return data.rename(columns={data.columns[0]: "rt", data.columns[1]: "mz", data.columns[2]: "fi"})


def iso_match_all(var_info, data, mz_T, rt_T, corr_T, mz_iso=1.003055):
    file = pd.read_excel(var_info, engine='openpyxl')
    name_mz = file['mzmed'].astype(str).str.split(".")
    name_rt = file['rtmed'].astype(str).str.split(".")
    file['MZRT_str'] = 'zzSLPOS_' + name_mz.str[0] + '.' + name_mz.str[1].str[:4] + '_' + name_rt.str[0] + '.' + \
                       name_rt.str[1].str[:4]
    var = file[['MZRT_str', 'mzmed', 'rtmed', 'fimed']]
    var = var.sort_values(by=['mzmed'])
    ind = var.index
    var = var.reset_index(drop=True)
    if data[-3:] == 'csv':
        data_pd = pd.read_csv(data, header=None)
    else:
        data_pd = pd.read_excel(data, header=None)
    iso_list = []
    for i in range(len(var) - 1):
        # set threshold
        rt_min = var.iloc[i, 2] - rt_T
        rt_max = var.iloc[i, 2] + rt_T
        mz_min = var.iloc[i, 1] + mz_iso - mz_T
        mz_max = var.iloc[i, 1] + mz_iso + mz_T
        for j in range(i + 1, len(var)):
            # mz threshold
            if mz_min <= var.iloc[j, 1] <= mz_max:
                # rt threshold
                if rt_min <= var.iloc[j, 2] <= rt_max:
                    data_1, data_2 = zip(*sorted(zip(list(data_pd.iloc[:, ind[i]]), list(data_pd.iloc[:, ind[j]]))))
                    corr_0, p_value = stats.pearsonr(data_1, data_2)
                    corr_1, p_value = stats.pearsonr(data_1[:round(len(data_1) / 2)], data_2[:round(len(data_1) / 2)])
                    corr_2, p_value = stats.pearsonr(data_1[round(len(data_1) / 2):], data_2[round(len(data_1) / 2):])
                    corr_3, p_value = stats.pearsonr(data_1[round(len(data_1) / 4):round(len(data_1) * 3 / 4)],
                                                     data_2[round(len(data_1) / 4):round(len(data_1) * 3 / 4)])
                    corr = max(corr_0, corr_1, corr_2, corr_3)
                    # corr threshold
                    if corr >= corr_T and var.iloc[i, 3] > var.iloc[j, 3]:
                        iso_list.append([var.iloc[i, 0], var.iloc[j, 0], corr, var.iloc[i, 1], var.iloc[j, 1],
                                         var.iloc[i, 2], var.iloc[j, 2], var.iloc[i, 3], var.iloc[j, 3]])
            elif var.iloc[j, 1] >= mz_max:
                break
    isotope_all = pd.DataFrame(iso_list, columns=['reference', 'target', 'correlation', 'mz_ref', 'mz_tar',
                                                  'rt_ref', 'rt_tar', 'fi_ref', 'fi_tar'])
    return isotope_all


def plot_matches(feature, rtfilter=None, mzfilter=None, tar_gofirst=False):

    if not tar_gofirst:
        x_all_rt = feature['rt_ref'].values
        y_all_rt = (feature['rt_tar'] - feature['rt_ref']).values

        x_all_mz = feature['mz_ref'].values
        y_all_mz = (feature['mz_tar'] - feature['mz_ref']).values

        x_all_fi = np.log10(feature['fi_ref'].values)
        y_all_fi = np.log10(feature['fi_tar'].values) - np.log10(feature['fi_ref'].values)

        x_all_fi_2 = np.log10(feature['fi_ref'].values)
        y_all_fi_2 = np.log10(feature['fi_tar'].values)
    else:
        x_all_rt = feature['rt_tar'].values
        y_all_rt = (feature['rt_ref'] - feature['rt_tar']).values

        x_all_mz = feature['mz_tar'].values
        y_all_mz = (feature['mz_ref'] - feature['mz_tar']).values

        x_all_fi = np.log10(feature['fi_tar'].values)
        y_all_fi = np.log10(feature['fi_ref'].values) - np.log10(feature['fi_tar'].values)

        x_all_fi_2 = np.log10(feature['fi_tar'].values)
        y_all_fi_2 = np.log10(feature['fi_ref'].values)

    # if rtfilter is not None or mzfilter is not None:
    #     for i in range(0, len(x_all_mz)):

    # model = SuperSmoother(alpha=5)
    # model.fit(x_all_rt, y_all_rt)
    # rt_xfit = np.linspace(0, 12, 2000)
    # rt_yfit = model.predict(rt_xfit)
    # model = SuperSmoother()
    # model.fit(x_all_mz, y_all_mz)
    # mz_xfit = np.linspace(0, 2000, 2000)
    # mz_yfit = model.predict(mz_xfit)

    plt.style.use('seaborn-darkgrid')
    fig, axs = plt.subplots(ncols=4, nrows=1, figsize=(13, 13 / 4))

    axs[0].scatter(x_all_rt, y_all_rt, s=0.3, c='red')
    axs[0].set_xlabel('RT ref', fontsize=7)
    axs[0].set_ylabel('RT diff', fontsize=7)
    # axs[0].set_ylim([-0.2, 0.2])
    # axs[0].plot(rt_xfit, rt_yfit, '-k')

    axs[1].scatter(x_all_mz, y_all_mz, s=0.3, c='red')
    axs[1].set_xlabel('MZ ref', fontsize=7)
    axs[1].set_ylabel('MZ diff', fontsize=7)
    # axs[1].set_ylim([-0.02, 0.01])
    # axs[1].plot(mz_xfit, mz_yfit, '-k')

    axs[2].scatter(x_all_fi, y_all_fi, s=0.3, c='blue')
    axs[2].set_xlabel('log10 FI ref', fontsize=7)
    axs[2].set_ylabel('log10 FI diff', fontsize=7)

    # axs[2].scatter(x_all_fi, y_all_fi, s=0.3, c='red')
    axs[3].scatter(x_all_fi_2, y_all_fi_2, s=0.3, c='blue')
    axs[3].set_xlabel('log10 FI ref', fontsize=7)
    axs[3].set_ylabel('log10 FI tar', fontsize=7)
    # axs[2].set_ylim([-10, 10])
    # axs[2].plot(fi_xfit, fi_yfit, '-k')

    fig.tight_layout()
    fig.show()
    plt.savefig('matches.png', dpi=200)
    # plt.close()


def successive_distance_2(matched_ref, matched_tar, threshold, step=1, startpoint=None, endpoint=None, mode="min"):
    matched_ref = np.array(matched_ref, dtype=np.float32)
    matched_tar = np.array(matched_tar, dtype=np.float32)

    datasize = matched_ref.shape[0]

    ref_and_tar = np.concatenate((np.reshape(matched_ref, (datasize, 1)), np.reshape(matched_tar, (datasize, 1))),
                                 axis=1)

    ref_and_tar = ref_and_tar[np.argsort(ref_and_tar[:, 0])]

    smoother_output = []
    diff_output = []

    if startpoint is None or startpoint < 2:
        startpoint = 0
    else:
        startpoint = startpoint - 1
    if endpoint is None or endpoint > datasize or endpoint < startpoint:
        endpoint = datasize
    if step < 1:
        step = 1

    currentpoint = startpoint
    smoother_output.append(ref_and_tar[currentpoint])
    flag_finsh = False
    while not flag_finsh:
        nextpoint_idx = -1
        n = 1
        while nextpoint_idx < 0:
            if currentpoint + step * n < endpoint:
                temparray = ref_and_tar[currentpoint + step * (n - 1) + 1:currentpoint + step * n + 1, :]
                end_of_temarray = currentpoint + step * n + 1
            else:
                temparray = ref_and_tar[currentpoint + step * (n - 1) + 1:endpoint, :]
                end_of_temarray = endpoint

            diff_of_current = np.absolute(ref_and_tar[currentpoint, 0] - ref_and_tar[currentpoint, 1])
            diff_of_reftar = np.absolute(temparray[:, 0] - temparray[:, 1])

            if mode == "min":
                leatest_diff = None
                for i in range(diff_of_reftar.shape[0]):
                    absdiff = abs(diff_of_reftar[i] - diff_of_current)
                    if absdiff < threshold:
                        if leatest_diff is None:
                            leatest_diff = absdiff
                            nextpoint_idx = currentpoint + step * (n - 1) + 1 + i
                        elif leatest_diff is not None and leatest_diff > absdiff:
                            leatest_diff = absdiff
                            nextpoint_idx = currentpoint + step * (n - 1) + 1 + i
            elif mode == "fix":
                absdiff = abs(diff_of_reftar[-1] - diff_of_current)
                if absdiff < threshold:
                    nextpoint_idx = currentpoint + step * (n - 1) + diff_of_reftar.shape[1]

            if nextpoint_idx != -1:
                currentpoint = nextpoint_idx
                smoother_output.append(ref_and_tar[nextpoint_idx])
                if nextpoint_idx == endpoint:
                    flag_finsh = True
                    nextpoint_idx = 0
            elif nextpoint_idx == -1:
                if end_of_temarray == endpoint:
                    flag_finsh = True
                    nextpoint_idx = 0
                else:
                    n = n + 1

    return smoother_output


def successive_distance(matched_ref, matched_tar, title, threshold=None, distribution="normal", factor=1):
    matched_ref = np.array(matched_ref, dtype=np.float32)
    matched_tar = np.array(matched_tar, dtype=np.float32)

    datasize = matched_ref.shape[0]

    ref_and_tar = np.concatenate((np.reshape(matched_ref, (datasize, 1)), np.reshape(matched_tar, (datasize, 1))),
                                 axis=1)
    new_order = np.argsort(ref_and_tar[:, 0])
    ref_and_tar = ref_and_tar[new_order]
    idxes = np.arange(matched_ref.size)[new_order]
    temp = np.diff(ref_and_tar[:, 0] - ref_and_tar[:, 1])
    plotting_unit = np.abs(
        np.max(ref_and_tar[:, 0] - ref_and_tar[:, 1]) - np.min(ref_and_tar[:, 0] - ref_and_tar[:, 1]))

    fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(13, 13 / 4))
    axs[0].scatter(ref_and_tar[:np.size(temp), 0], temp, s=2)
    axs[0].set_xlabel(title + ' value', fontsize=5)
    axs[0].set_ylabel(title + ' delta diff', fontsize=5)
    axs[0].set_title("scatterplot of "+title+' delta diff')
    diff = np.absolute(temp)
    if threshold is not None:
        fig.suptitle("Sd Filter of " + title + " with " + str(threshold) +" as threshold")
    else:
        fig.suptitle("Sd Filter of " + title + " with " + str(factor) + " as MAD's factor")

    smoother_output = []
    idx_in_originaldata = []
    pre_i = -10
    filter_range = None
    if threshold is not None:
        for i in range(np.size(diff)):
            if diff[i] < threshold:
                if i == pre_i + 1:
                    smoother_output.append(ref_and_tar[i + 1])
                    idx_in_originaldata.append(idxes[i + 1])
                else:
                    smoother_output.append(ref_and_tar[i])
                    smoother_output.append(ref_and_tar[i + 1])
                    idx_in_originaldata.extend([idxes[i], idxes[i + 1]])
                pre_i = i
        smoother_output = np.array(smoother_output, dtype=np.float32)
        axs[2].scatter(smoother_output[:, 0], smoother_output[:, 1] - smoother_output[:, 0], s=0.3, c='red')
        axs[2].set_ylabel(title + ' diff', fontsize=5)
        axs[2].set_xlabel(title, fontsize=5)
        axs[2].set_title('result after filtering', fontsize=5)
        return smoother_output, idx_in_originaldata

    else:
        n, bins, patches = axs[1].hist(temp,
                                       bins=np.linspace(-1 * plotting_unit, plotting_unit,
                                                        (int)(datasize * 1)), alpha=0.5,
                                       label="histogram")
        mean_his = np.mean(temp)
        peak = n.max()
        x_peak = bins[np.where(n == peak)][0]

        if distribution == "cauchy":
            x_distribution = np.linspace(x_peak - plotting_unit, x_peak + plotting_unit,
                                         (int)(datasize * 5))
            gamma = 1 / (np.pi * peak)
            y_distribution = 1 / (np.pi * gamma * (1 + ((x_distribution - x_peak) / gamma) ** 2))
            y_distribution_modeled = y_distribution * np.pi * gamma * peak
            axs[1].plot(x_distribution, y_distribution_modeled, color="orange", label=distribution + " distribution")
            filter_range = np.array([x_peak - gamma * factor, x_peak + gamma * factor])
            axs[1].fill_between(x_distribution, y_distribution_modeled, 0,
                                where=(x_distribution > filter_range[0]) & (x_distribution < filter_range[1]),
                                color='g',
                                alpha=0.2, label=str(factor) + " * FWHM")
            axs[1].legend(fontsize=7, loc='upper left', frameon=True, edgecolor="black")
            # plt.xlim((-0.1, 0.1))

        elif distribution == "normal" or distribution is None:
            x_distribution = np.linspace(mean_his - plotting_unit, mean_his + plotting_unit,
                                         (int)(datasize * 5))
            median_his = np.median(temp)
            mad_his = np.median(np.abs(temp - median_his))
            # std_his = 1.4826 * mad_his
            std_his = 1.4826 * mad_his
            filter_range = np.array([mean_his - std_his * factor, mean_his + std_his * factor])

            if distribution is not None:
                y_distribution = (1 / (math.sqrt(2 * np.pi * std_his))) * np.exp(
                    -0.5 * ((x_distribution - mean_his) / std_his) ** 2)
                y_distribution_modeled = y_distribution / (1 / (math.sqrt(2 * np.pi * std_his))) * peak
                axs[1].plot(x_distribution, y_distribution_modeled, color="orange",
                            label=distribution + " distribution")

                # print(filter_range)
                axs[1].fill_between(x_distribution, y_distribution_modeled, 0,
                                    where=(x_distribution > filter_range[0]) & (x_distribution < filter_range[1]),
                                    color='g', alpha=0.2, label=str(factor) + " * STD")
            else:
                axs[1].fill_between(x_distribution, peak, 0,
                                    where=(x_distribution > filter_range[0]) & (x_distribution < filter_range[1]),
                                    color='g', alpha=0.2, label=str(factor) + " * STD")
            axs[1].legend(fontsize=7, loc='upper left', frameon=True, edgecolor="black")

        for i in range(np.size(temp)):
            if filter_range[0] <= temp[i] <= filter_range[1]:
                if i == pre_i + 1:
                    smoother_output.append(ref_and_tar[i + 1])
                    idx_in_originaldata.append(idxes[i + 1])
                else:
                    smoother_output.append(ref_and_tar[i])
                    smoother_output.append(ref_and_tar[i + 1])
                    idx_in_originaldata.extend([idxes[i], idxes[i + 1]])
                pre_i = i
    smoother_output = np.array(smoother_output, dtype=np.float32)
    axs[2].scatter(smoother_output[:, 0], smoother_output[:, 1] - smoother_output[:, 0], s=0.3, c='red')
    axs[2].set_ylabel(title + ' diff', fontsize=5)
    axs[2].set_xlabel(title, fontsize=5)
    axs[2].set_title('result after filtering', fontsize=5)

    fig.show()
    return smoother_output, idx_in_originaldata


def trendline_smoother(data, featurename, degree=2, frac=0.25, thre_for_Rui=None,
                       successive_distribution="cauchy", rui_factor=2,
                       outliers=True, distribution="cauchy", factor=1, MAD=False, skip_successive_distance=False):
    plt.style.use('seaborn-darkgrid')
    fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(12, 9))
    ref_and_tar = np.array(data, dtype=np.float32)
    ruismoother_output = ref_and_tar[np.argsort(ref_and_tar[:, 0])]
    bins_unit = np.abs(
        np.max(ref_and_tar[:, 0] - ref_and_tar[:, 1]) - np.min(ref_and_tar[:, 0] - ref_and_tar[:, 1]))

    axs[0, 0].scatter(data[:, 0], data[:, 1] - data[:, 0], s=0.3, c='red')
    axs[0, 0].set_xlabel(featurename + ' ref', fontsize=5)
    axs[0, 0].set_ylabel(featurename + ' diff', fontsize=5)
    axs[0, 0].set_title("Original dataset")

    # Rui algorithm
    if not skip_successive_distance:
        if thre_for_Rui is None:
            ruismoother_output, idx_in_originaldata = successive_distance(ref_and_tar[:, 0], ref_and_tar[:, 1],
                                                                          distribution=successive_distribution,
                                                                          factor=rui_factor)
        else:
            ruismoother_output, idx_in_originaldata = successive_distance(ref_and_tar[:, 0], ref_and_tar[:, 1],
                                                                          threshold=thre_for_Rui)

        axs[0, 1].scatter(ruismoother_output[:, 0], ruismoother_output[:, 1] - ruismoother_output[:, 0], s=0.3, c='red')
        axs[0, 1].set_xlabel(featurename + ' ref', fontsize=5)
        axs[0, 1].set_ylabel(featurename + ' diff', fontsize=5)
        axs[0, 1].set_title("After Rui Filter")

    # LOESS smoother
    x_ref = ruismoother_output[:, 0]
    y_diff = ruismoother_output[:, 1] - ruismoother_output[:, 0]
    xout, yout, wout = loess_1d.loess_1d(x_ref, y_diff, xnew=None, degree=degree, frac=frac,
                                         npoints=None,
                                         rotate=False, sigy=None)

    if outliers:
        axs[0, 2].scatter(x_ref, y_diff, s=0.3, c='red')
        y_his = np.array(y_diff, dtype=np.float32) - np.array(yout, dtype=np.float32)
        print(y_his.shape)
    else:
        axs[0, 2].scatter(x_ref[wout != 0], y_diff[wout != 0], s=0.3, c='red')
        axs[0, 2].scatter(x_ref[wout == 0], y_diff[wout == 0], s=0.3, c='blue', label="outliers")
        x_ref = x_ref[wout != 0]
        y_diff = y_diff[wout != 0]
        y_his = np.array(y_diff, dtype=np.float32) - (np.array(yout, dtype=np.float32)[wout != 0])
    axs[0, 2].plot(xout, yout, '-k', label="LOESS smoother")
    axs[0, 2].set_xlabel(featurename + ' ref', fontsize=5)
    axs[0, 2].set_ylabel(featurename + ' diff', fontsize=5)
    axs[0, 2].set_title("Applying LOESS Smoother")
    axs[0, 2].legend(fontsize=7, loc='upper left', frameon=True, edgecolor="black")

    # distribution
    mean_his = np.mean(y_his)
    if MAD:
        median_his = np.median(y_his)
        mad_his = np.median(np.abs(y_his - median_his))
        std_his = 1.4826 * mad_his
    else:
        std_his = np.std(y_his)
    print("std: ", std_his)

    n, bins, patches = axs[1, 0].hist(y_his,
                                      bins=np.linspace(-1 * bins_unit, bins_unit,
                                                       (int)(data.shape[0] * 1)), alpha=0.5,
                                      label="histogram")
    # plt.figure()
    # plt.hist(y_his, bins=np.linspace(-1 * threshold_for_matching*2, threshold_for_matching*2,
    #                                                           (int)(data.shape[0] * 1)))
    # print(sum(n))
    axs[1, 0].set_title(distribution + " Distribution for distance to smoother line")
    axs[1, 0].set_xlabel(featurename + ' distance', fontsize=5)
    peak = n.max()
    x_peak = bins[np.where(n == peak)][0]
    new_ref = []
    new_diff = []
    filtered_ref = []
    filtered_diff = []
    if distribution == "normal":
        x_distribution = np.linspace(mean_his - bins_unit, mean_his + bins_unit,
                                     (int)(data.shape[0] * 5))
        y_distribution = (1 / (math.sqrt(2 * np.pi * std_his))) * np.exp(
            -0.5 * ((x_distribution - mean_his) / std_his) ** 2)
        y_distribution_modeled = y_distribution / (1 / (math.sqrt(2 * np.pi * std_his))) * peak
        axs[1, 0].plot(x_distribution, y_distribution_modeled, color="orange", label=distribution + "distribution")
        if factor is not None:
            if MAD:
                FWHM_range = np.array(
                    [mean_his - mad_his * factor,
                     mean_his + mad_his * factor])
            else:
                FWHM_range = np.array(
                    [mean_his - std_his * factor,
                     mean_his + std_his * factor])

            # print(FWHM_range)
            axs[1, 0].fill_between(x_distribution, y_distribution_modeled, 0,
                                   where=(x_distribution > FWHM_range[0]) & (x_distribution < FWHM_range[1]), color='g',
                                   alpha=0.2, label=str(factor) + " * MAD")
            axs[1, 0].legend(fontsize=7, loc='upper left', frameon=True, edgecolor="black")
            for i in range(len(x_ref)):
                if FWHM_range[0] <= y_his[i] <= FWHM_range[1]:
                    new_ref.append(x_ref[i])
                    new_diff.append(y_diff[i])
                else:
                    filtered_ref.append(x_ref[i])
                    filtered_diff.append(y_diff[i])

    elif distribution == "cauchy":
        gamma = 1 / (np.pi * peak)
        x_distribution = np.linspace(x_peak - bins_unit, x_peak + bins_unit,
                                     (int)(data.shape[0] * 5))
        y_distribution = 1 / (np.pi * gamma * (1 + ((x_distribution - x_peak) / gamma) ** 2))
        y_distribution_modeled = y_distribution * np.pi * gamma * peak
        axs[1, 0].plot(x_distribution, y_distribution_modeled, color="orange", label=distribution + "distribution")
        if factor is not None:
            FWHM_range = np.array([x_peak - gamma * factor, x_peak + gamma * factor])
            # print("fwhm range ", FWHM_range)
            axs[1, 0].fill_between(x_distribution, y_distribution_modeled, 0,
                                   where=(x_distribution > FWHM_range[0]) & (x_distribution < FWHM_range[1]), color='g',
                                   alpha=0.2, label=str(factor) + " * FWHM")
            axs[1, 0].legend(fontsize=7, loc='upper left', frameon=True, edgecolor="black")
            for i in range(len(x_ref)):
                if FWHM_range[0] <= y_his[i] <= FWHM_range[1]:
                    new_ref.append(x_ref[i])
                    new_diff.append(y_diff[i])
                else:
                    filtered_ref.append(x_ref[i])
                    filtered_diff.append(y_diff[i])

    axs[1, 1].scatter(new_ref, new_diff, s=0.3, c='red')
    axs[1, 1].scatter(filtered_ref, filtered_diff, s=0.3, c='green', label="filtered points")
    axs[1, 1].set_title("Filtered Points")
    axs[1, 1].set_xlabel(featurename + ' ref', fontsize=5)
    axs[1, 1].set_ylabel(featurename + ' diff', fontsize=5)
    axs[1, 1].legend(fontsize=7, loc='upper left', frameon=True, edgecolor="black")

    axs[1, 2].scatter(new_ref, new_diff, s=0.3, c='red')
    axs[1, 2].plot(xout, yout, '-k', label="LOESS smoother")
    axs[1, 2].set_title("Final result")
    axs[1, 2].set_xlabel(featurename + ' ref', fontsize=5)
    axs[1, 2].set_ylabel(featurename + ' diff', fontsize=5)
    axs[1, 2].legend(fontsize=7, loc='upper right', frameon=True, edgecolor="black")

    fig.tight_layout()
    # fig.show()

    return xout, yout, wout, y_his, std_his, new_ref, new_diff


def plot_fig(data):
    plt.style.use('seaborn-darkgrid')
    # fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(13, 13 / 4))
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1] - data[:, 0], s=0.3, c='red')
    plt.xlabel('RT ref', fontsize=5)
    plt.ylabel('RT diff', fontsize=5)
    plt.show()


def plot_withSuperSmoother(data, startpoint, endpoint, pieces, alpha=None):
    '''

    :param data: structure [[ref, tar], [ref, tar], ...]
    :param startpoint:
    :param endpoint:
    :param pieces:
    :param alpha:
    :return:
    '''
    plt.style.use('seaborn-darkgrid')
    # fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(13, 13 / 4))
    if alpha is not None:
        model = SuperSmoother(alpha=alpha)
    else:
        model = SuperSmoother()
    x = data[:, 0]
    y = data[:, 1] - data[:, 0]
    model.fit(x, y)
    xfit = np.linspace(startpoint, endpoint, pieces)
    yfit = model.predict(xfit)
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, s=0.3, c='red')
    plt.plot(xfit, yfit, '-k')
    plt.xlabel('RT ref', fontsize=5)
    plt.ylabel('RT diff', fontsize=5)
    plt.show()


def plot_withLoess(data, degree, frac, curr_threshold, outliers=None, FWHM=True, distribution="normal"):
    x_ref = data[:, 0]
    y_diff = data[:, 1] - data[:, 0]

    xout, yout, wout = loess_1d.loess_1d(x_ref, y_diff, xnew=None, degree=degree, frac=frac,
                                         npoints=None,
                                         rotate=False, sigy=None)

    plt.figure(figsize=(8, 6))
    if outliers is None or outliers is True:
        plt.scatter(x_ref, y_diff, s=0.3, c='red')
        y_his = np.array(y_diff, dtype=np.float32) - np.array(yout, dtype=np.float32)
    elif outliers is False:
        plt.scatter(x_ref[wout != 0], y_diff[wout != 0], s=0.3, c='red')
        plt.scatter(x_ref[wout == 0], y_diff[wout == 0], s=0.3, c='blue', label="outliers")
        x_ref = x_ref[wout != 0]
        y_diff = y_diff[wout != 0]
        y_his = np.array(y_diff, dtype=np.float32) - (np.array(yout, dtype=np.float32)[wout != 0])
    plt.plot(xout, yout, '-k')
    plt.xlabel('RT ref', fontsize=5)
    plt.ylabel('RT diff', fontsize=5)
    plt.legend()

    mean_his = np.mean(y_his)
    std_his = np.std(y_his)
    x_distribution = np.linspace(mean_his - curr_threshold, mean_his + curr_threshold, (int)(data.shape[0] * 2))
    print(mean_his)
    print(std_his)

    fig, ax = plt.subplots()
    n, bins, patches = ax.hist(y_his, bins=np.linspace(-1, 1, (int)(data.shape[0] * 1)), rwidth=20, alpha=0.5)
    peak = n.max()
    new_ref = []
    new_diff = []
    filtered_ref = []
    filtered_diff = []
    if distribution == "normal":
        y_distribution = (1 / (math.sqrt(2 * np.pi * std_his))) * np.exp(
            -0.5 * ((x_distribution - mean_his) / std_his) ** 2)
        y_distribution_modeled = y_distribution / (1 / (math.sqrt(2 * np.pi * std_his))) * peak
        ax.plot(x_distribution, y_distribution_modeled)
        if FWHM is True:
            FWHM_range = np.array(
                [mean_his - math.sqrt(2 * np.log(2)) * std_his, mean_his + math.sqrt(2 * np.log(2)) * std_his])
            print(FWHM_range)
            ax.fill_between(x_distribution, y_distribution_modeled, 0,
                            where=(x_distribution > FWHM_range[0]) & (x_distribution < FWHM_range[1]), color='g',
                            alpha=0.2)
            plt.show()
            for i in range(len(x_ref)):
                if FWHM_range[0] <= y_his[i] <= FWHM_range[1]:
                    new_ref.append(x_ref[i])
                    new_diff.append(y_diff[i])
                else:
                    filtered_ref.append(x_ref[i])
                    filtered_diff.append(y_diff[i])
    if distribution == "cauchy":
        gamma = 1 / (np.pi * peak)
        y_distribution = 1 / (np.pi * gamma * (1 + ((x_distribution - mean_his) / gamma) ** 2))
        y_distribution_modeled = y_distribution * np.pi * gamma * peak
        ax.plot(x_distribution, y_distribution_modeled)
        if FWHM is True:
            FWHM_range = np.array([mean_his - gamma, mean_his + gamma])
            # print(FWHM_range)
            ax.fill_between(x_distribution, y_distribution_modeled, 0,
                            where=(x_distribution > FWHM_range[0]) & (x_distribution < FWHM_range[1]), color='g',
                            alpha=0.2)
            plt.show()
            for i in range(len(x_ref)):
                if FWHM_range[0] <= y_his[i] <= FWHM_range[1]:
                    new_ref.append(x_ref[i])
                    new_diff.append(y_diff[i])
                else:
                    filtered_ref.append(x_ref[i])
                    filtered_diff.append(y_diff[i])

    plt.figure(figsize=(8, 6))
    plt.scatter(new_ref, new_diff, s=0.3, c='red')

    return xout, yout, wout, y_his, std_his, new_ref, new_diff


def bad_diffs_filter(data, threshould=None):
    # diff = np.abs(data[:,0]-data[:,1])
    diff = data[:, 1] - data[:, 0]
    plt.figure()
    if threshould is not None:
        return data[diff < threshould]
    else:
        loz_model = Lorentz1D()
        plt.plot(diff, loz_model(diff), '-k')
        plt.show()


def feature_matching(ref_dataset, tar_dataset, rt_threshold, mz_threshold, fi_threshold=None):
    ref_dataset = np.array(ref_dataset.values, dtype=np.float32)
    tar_dataset = np.array(tar_dataset.values, dtype=np.float32)

    print(ref_dataset)

    ref_size = ref_dataset.shape[0]

    matched_data = np.zeros([1, 6])
    for i in range(ref_size):
        temp_diff = tar_dataset[:, 0:2] - ref_dataset[i, 0:2]
        rows = np.where(np.all(
            ([rt_threshold[0], mz_threshold[0]] <= temp_diff) & (temp_diff <= [rt_threshold[1], mz_threshold[1]]),
            axis=1))
        if np.size(tar_dataset[rows]) != 0:
            tmp_ref = np.repeat(np.reshape(ref_dataset[i], (1, 3)), tar_dataset[rows].shape[0], axis=0)
            temp_matches = np.concatenate((tmp_ref, tar_dataset[rows]), axis=1)
            matched_data = np.concatenate((matched_data, temp_matches), axis=0)

    matched_frame = pd.DataFrame(matched_data[1:, [0, 3, 1, 4, 2, 5]], columns=['rt_ref', 'rt_tar', 'mz_ref', 'mz_tar',
                                                                                'fi_ref', 'fi_tar'])
    plot_matches(matched_frame)
    plot_RTMZ_style(matched_frame)
    return matched_frame


def filter_badmathces(data_to_filter, original_dataset, outliers_idx_list, threshold_closest_points, threshold_outliers,
                      distancefunction="Euclidean"):
    original_dataset = np.array(original_dataset, dtype=np.float32)
    ori_len = original_dataset.shape[0]

    data_to_filter = np.array(data_to_filter, dtype=np.float32)
    data_len = data_to_filter.shape[0]

    ori_idxes = np.arange(ori_len)
    result_dataset = []
    goodmatches_idx = []
    badmatches_idx = []
    for i in range(data_len):
        if distancefunction == "Euclidean":
            temp_distance = np.sqrt(np.sum(np.square(original_dataset - data_to_filter[i]), axis=1))
            new_order = np.argsort(temp_distance)
            temp_distance = temp_distance[new_order]
            ori_idxes = ori_idxes[new_order]

            outliers_number = 0
            for j in ori_idxes[:threshold_closest_points]:
                if j in outliers_idx_list:
                    outliers_number = outliers_number + 1
            print(outliers_number)
            if outliers_number < threshold_outliers:
                result_dataset.append(data_to_filter[i])
                goodmatches_idx.append(i)
            else:
                badmatches_idx.append(i)

    return np.array(result_dataset, dtype=np.float32), goodmatches_idx, badmatches_idx


def combine_reftar_tarref(reftar_data, tarref_data, distance_threshold, matches_threshold,
                          distancefunction="Euclidean"):
    reftar_data = np.array(reftar_data, dtype=np.float32)
    reftar_matches = np.copy(reftar_data)
    reftar_matches[:, 1] = np.transpose(reftar_matches[:, 1] - reftar_matches[:, 0])

    tarref_data = np.array(tarref_data, dtype=np.float32)
    tarref_matches = np.copy(tarref_data)
    tarref_matches = tarref_matches[:, [1, 0]]
    tarref_matches[:, 1] = np.transpose(tarref_matches[:, 1] - tarref_matches[:, 0])

    reftar_matches = (reftar_matches - reftar_matches.min(0)) / reftar_matches.ptp(0)
    tarref_matches = (tarref_matches - tarref_matches.min(0)) / tarref_matches.ptp(0)

    # plt.figure(figsize=(8, 6))
    # plt.scatter(reftar_matches[:,0], reftar_matches[:,1], s=2, c='blue')
    # plt.show()

    temp = np.concatenate((reftar_matches, tarref_matches), axis=0)

    new_reftar = combine_reftar_tarref_helper(reftar_data, reftar_matches, temp, distance_threshold, matches_threshold,
                                              "Density filtering for SD filter result in Ref-Tar direction",
                                              distancefunction=distancefunction)
    new_tarref = combine_reftar_tarref_helper(tarref_data, tarref_matches, temp, distance_threshold, matches_threshold,
                                              "Density filtering for SD filter result in Tar-Ref direction",
                                              distancefunction=distancefunction)

    return new_reftar, new_tarref


def combine_reftar_tarref_helper(data, matches, filter, distance_threshold, matches_threshold, title,
                                 distancefunction="Euclidean"):
    data_len = data.shape[0]

    new_data = []
    dis_mean = []
    num_within_thre = []

    for i in range(data_len):
        if distancefunction == "Euclidean":
            temp_distance = np.sqrt(np.sum(np.square(filter - matches[i]), axis=1))
            temp_distance = np.sort(temp_distance)
            # print(temp_distance)
            # print("---------------------------------------------------")
            dis_mean.append(np.mean(temp_distance[:matches_threshold]))
            num_within_thre.append((temp_distance <= distance_threshold).sum())
            if temp_distance[matches_threshold - 1] <= distance_threshold:
                new_data.append(data[i])

    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(12, 12))
    axs[0, 0].hist(dis_mean, bins=50)
    im = axs[0, 1].scatter(matches[:, 0], matches[:, 1], s=2, c=dis_mean, cmap='coolwarm')
    axs[0, 0].set_title("Hist: Within threshold=" + str(matches_threshold), fontsize=10)
    axs[0, 0].set_xlabel("distances")
    axs[0, 1].set_title("Colormap: Within threshold=" + str(matches_threshold), fontsize=10)
    fig.colorbar(im, ax=axs[0, 1], pad=0.05)

    axs[1, 0].hist(num_within_thre, bins=50)
    im = axs[1, 1].scatter(matches[:, 0], matches[:, 1], s=2, c=num_within_thre, cmap='coolwarm')
    axs[1, 0].set_title("Hist: Within distance=" + str(distance_threshold), fontsize=10)
    axs[1, 0].set_xlabel("counts within distance threshold")
    axs[1, 1].set_title("Colormap: Within distance=" + str(distance_threshold), fontsize=10)
    fig.colorbar(im, ax=axs[1, 1], pad=0.05)

    fig.suptitle(title, fontsize=15)
    fig.show()
    return np.array(new_data, dtype=np.float32)


def activated_residuals(dataframe, loess_x, loess_y, feature_name, activation="tanh"):
    if feature_name == "fi":
        data = np.array(dataframe[["fi_ref", "fi_tar"]].values, dtype=np.float32)
    elif feature_name == "mz":
        data = np.array(dataframe[["mz_ref", "mz_tar"]].values, dtype=np.float32)
    elif feature_name == "rt":
        data = np.array(dataframe[["rt_ref", "rt_tar"]].values, dtype=np.float32)
    else:
        data = np.array(dataframe[["rt_ref", "rt_tar"]].values, dtype=np.float32)
    temp_ref_rt = data[:, 0]
    temp_tar_rt = data[:, 1]
    idx_order = np.argsort(loess_x)
    xOut = loess_x[idx_order]
    yOut = loess_y[idx_order]
    interpolate_function = interpolate.interp1d(xOut, yOut)
    ynew = interpolate_function(temp_ref_rt[np.where((temp_ref_rt > xOut[0]) & (temp_ref_rt < xOut[-1]))])

    diff_from_match_to_loess = []
    n = 0
    temp_difference = temp_tar_rt - temp_ref_rt
    for i in range(len(temp_ref_rt)):

        if temp_ref_rt[i] <= xOut[0]:
            diff_from_match_to_loess.append(temp_difference[i] - yOut[0])
        elif temp_ref_rt[i] >= xOut[-1]:
            diff_from_match_to_loess.append(temp_difference[i] - yOut[-1])
        else:
            diff_from_match_to_loess.append(temp_difference[i] - ynew[n])
            n = n + 1

    fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(13, 13 / 4))

    diff_from_match_to_loess = np.array(diff_from_match_to_loess)
    median = np.median(diff_from_match_to_loess)
    mad = np.median(np.abs(diff_from_match_to_loess - median))
    diff_from_match_to_loess = diff_from_match_to_loess / (3*mad)

    axs[0].scatter(temp_ref_rt, temp_tar_rt - temp_ref_rt, s=2, c='green')
    axs[0].scatter(temp_ref_rt[np.where((temp_ref_rt > np.min(xOut)) & (temp_ref_rt < np.max(xOut)))], ynew, s=2,
                   c='red')
    axs[0].hlines(y=yOut[0], xmin=0, xmax=xOut[0], linewidth=1, color='red')
    axs[0].hlines(y=yOut[-1], xmin=xOut[-1], xmax=np.max(temp_ref_rt), linewidth=1, color='red')

    if activation == "tanh":
        score = np.tanh(diff_from_match_to_loess)
    elif activation == "sigmoid":
        score = 1 / (1 + np.exp(-1 * diff_from_match_to_loess))
    else:
        score = np.tanh(diff_from_match_to_loess)

    axs[1].scatter(temp_ref_rt, score, s=2, c='blue')

    im = axs[2].scatter(temp_ref_rt, temp_difference, s=2, c=score, cmap='coolwarm')

    fig.colorbar(im)
    fig.tight_layout()
    return score


def plot_colored_scores(feature, score, normalization=False):
    title = "reference"
    x_all_rt = feature['rt_ref'].values
    y_all_rt = (feature['rt_tar'] - feature['rt_ref']).values

    x_all_mz = feature['mz_ref'].values
    y_all_mz = (feature['mz_tar'] - feature['mz_ref']).values

    x_all_fi = np.log10(feature['fi_ref'].values)
    y_all_fi = np.log10(feature['fi_tar'].values) - np.log10(feature['fi_ref'].values)

    x_all_fi_2 = np.log10(feature['fi_ref'].values)
    y_all_fi_2 = np.log10(feature['fi_tar'].values)

    if normalization:
        score = (score - np.min(score))/score.ptp(0)

    plt.style.use('seaborn-darkgrid')
    fig, axs = plt.subplots(ncols=4, nrows=1, figsize=(13, 13 / 4))

    im = axs[0].scatter(x_all_rt, y_all_rt, s=1, c=score, cmap='cool')
    axs[0].set_xlabel('RT ' + title, fontsize=5)
    axs[0].set_ylabel('RT diff', fontsize=5)
    # axs[0].set_ylim([-0.2, 0.2])
    # axs[0].plot(rt_xfit, rt_yfit, '-k')

    axs[1].scatter(x_all_mz, y_all_mz, s=1, c=score, cmap='cool')
    axs[1].set_xlabel('MZ ' + title, fontsize=5)
    axs[1].set_ylabel('MZ diff', fontsize=5)
    # axs[1].set_ylim([-0.02, 0.01])
    # axs[1].plot(mz_xfit, mz_yfit, '-k')

    axs[2].scatter(x_all_fi, y_all_fi, s=1, c=score, cmap='cool')
    axs[2].set_xlabel('log10 FI ' + title, fontsize=5)
    axs[2].set_ylabel('log10 FI diff', fontsize=5)

    # axs[2].scatter(x_all_fi, y_all_fi, s=0.3, c='red')
    axs[3].scatter(x_all_fi_2, y_all_fi_2, s=1, c=score, cmap='cool')
    axs[3].set_xlabel('log10 FI ' + title, fontsize=5)
    axs[3].set_ylabel('log10 FI tar', fontsize=5)
    # axs[2].set_ylim([-10, 10])
    # axs[2].plot(fi_xfit, fi_yfit, '-k')

    fig.colorbar(im)
    fig.show()


class PolynomialRegression(object):
    def __init__(self, degree=3, coeffs=None):
        self.degree = degree
        self.coeffs = coeffs

    def fit(self, X, y):
        self.coeffs = np.polyfit(X.ravel(), y, self.degree)

    def get_params(self, deep=False):
        return {'coeffs': self.coeffs}

    def set_params(self, coeffs=None, random_state=None):
        self.coeffs = coeffs

    def predict(self, X):
        poly_eqn = np.poly1d(self.coeffs)
        y_hat = poly_eqn(X.ravel())
        return y_hat

    def score(self, X, y):
        return mean_squared_error(y, self.predict(X))


def RANSAC_scoring(filterd_data, original_data, title):
    plt.style.use('seaborn-darkgrid')
    fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(12, 9))
    ref_and_tar = np.array(filterd_data, dtype=np.float32)
    ref_and_tar = ref_and_tar[np.argsort(ref_and_tar[:, 0])]

    axs[0].scatter(ref_and_tar[:, 0], ref_and_tar[:, 1] - ref_and_tar[:, 0], s=0.3, c='red')
    axs[0].set_xlabel(title + ' ref', fontsize=5)
    axs[0].set_ylabel(title + ' diff', fontsize=5)
    axs[0].set_title("Original filtered dataset")

    ransac = linear_model.RANSACRegressor(PolynomialRegression(degree=3))
    ransac.fit(ref_and_tar[:, 0].reshape(-1, 1), (ref_and_tar[:, 1] - ref_and_tar[:, 0]))
    anchors = ransac.inlier_mask_
    outliers = np.logical_not(anchors)

    new_reftar = ref_and_tar[anchors]
    axs[1].scatter(new_reftar[:, 0], new_reftar[:, 1] - new_reftar[:, 0], s=0.3, c='green')
    axs[1].set_xlabel(title + ' ref', fontsize=5)
    axs[1].set_ylabel(title + ' diff', fontsize=5)
    axs[1].set_title("Inliers found by ransac")

    pre_y = ransac.predict(ref_and_tar[:, 0].reshape(-1, 1))
    axs[2].scatter(ref_and_tar[:, 0], ref_and_tar[:, 1] - ref_and_tar[:, 0], s=0.3, c='red', alpha=0.5)
    axs[2].scatter(new_reftar[:, 0], new_reftar[:, 1] - new_reftar[:, 0], s=0.3, c='green')
    axs[2].plot(ref_and_tar[:, 0], pre_y, '-k', label="LOESS smoother")
    axs[2].set_xlabel(title + ' ref', fontsize=5)
    axs[2].set_ylabel(title + ' diff', fontsize=5)

    fig.tight_layout()
    return ransac


def remove_duplicates(data, scores, threshold=1):
    arr_ref = np.transpose(
        np.array(
            (data["rt_ref"].astype(str) + "_" + data["mz_ref"].astype(str) + "_" + data["fi_ref"].astype(str) + "_ref"),
            ndmin=2))
    arr_tar = np.transpose(
        np.array(
            (data["rt_tar"].astype(str) + "_" + data["mz_tar"].astype(str) + "_" + data["fi_tar"].astype(str) + "_tar"),
            ndmin=2))
    temp_groupmatches = np.concatenate((arr_ref, arr_tar), axis=1)
    temp_groupmatches = np.concatenate((temp_groupmatches, np.transpose(np.array(scores, ndmin=2))), axis=1)

    G = nx.Graph()
    G.add_weighted_edges_from(temp_groupmatches)
    subgraphs = list((G.subgraph(c) for c in nx.connected_components(G)))

    res = []
    for g in subgraphs:
        # nodes_list.append(g.number_of_nodes())
        temp_g = g.copy()
        while temp_g.number_of_edges() > 0:
            print(temp_g.number_of_nodes())
            edgeData = list(e for e in temp_g.edges.data("weight"))
            edgeData = sorted(edgeData, key=lambda x: x[2])
            print(edgeData)
            res.append(edgeData[0])
            temp_g.remove_node(edgeData[0][0])
            temp_g.remove_node(edgeData[0][1])

    for i in range(len(res)):
        if res[i][0].endswith("tar"):
            temp_tuple = (res[i][1], res[i][0], res[i][2])
            res[i] = temp_tuple

    res_df = pd.DataFrame(list(e[2] for e in res), columns=["score"])
    reference = list(e[0].split("_")[:3] for e in res)
    target = list(e[1].split("_")[:3] for e in res)
    res_df[["rt_ref", "mz_ref", "fi_ref"]] = reference
    res_df[["rt_tar", "mz_tar", "fi_tar"]] = target
    res_df = res_df.astype(float)

    plt.figure(figsize=(8, 6))
    plt.hist(res_df["score"].values, bins=50, color="skyblue", label="unfiltered matches")
    plt.xlabel("socres")
    plt.ylabel("scouts")
    plt.legend()

    if threshold is not None:
        res_df = res_df[res_df["score"] <= threshold]
        plt.hist(res_df["score"].values, bins=50, color='orange', alpha=0.2, label="good matches")

    plt.show()

    plot_matches(res_df)
    plot_colored_scores(res_df, res_df["score"].values)
    plot_RTMZ_style(res_df)
    return res_df


def subgroups_filter(data, node_threshold, plots=False):
    arr_ref = np.transpose(
        np.array(
            (data["rt_ref"].astype(str) + "_" + data["mz_ref"].astype(str) + "_" + data["fi_ref"].astype(str) + "_ref"),
            ndmin=2))
    arr_tar = np.transpose(
        np.array(
            (data["rt_tar"].astype(str) + "_" + data["mz_tar"].astype(str) + "_" + data["fi_tar"].astype(str) + "_tar"),
            ndmin=2))
    temp_groupmatches = np.concatenate((arr_ref, arr_tar), axis=1)

    G = nx.Graph()
    G.add_edges_from(temp_groupmatches)
    subgraphs = list((G.subgraph(c) for c in nx.connected_components(G)))

    res = []
    edges_list = []
    picture_subgroups = nx.Graph()
    for g in subgraphs:
        edges_list.append(g.number_of_edges())
        if g.number_of_edges() <= node_threshold:
            res.extend(list(n for n in g.edges))
            picture_subgroups = nx.compose(picture_subgroups, g)
    for i in range(len(res)):
        if res[i][0].endswith("tar"):
            temp_tuple = (res[i][1], res[i][0])
            res[i] = temp_tuple

    res_df = pd.DataFrame()
    reference = list(e[0].split("_")[:3] for e in res)
    target = list(e[1].split("_")[:3] for e in res)
    res_df[["rt_ref", "mz_ref", "fi_ref"]] = reference
    res_df[["rt_tar", "mz_tar", "fi_tar"]] = target
    res_df = res_df.astype(float)

    plot_matches(res_df)
    plt.figure(figsize=(8, 6))
    plt.hist(edges_list, bins=50, color="skyblue", label="all")
    plt.xlabel("num of matches")
    plt.ylabel("counts of subgroups")
    plt.show()

    if plots:
        plt.figure()
        nx.draw(picture_subgroups, node_size=4)
    return res_df



def top_FI(dataframe, percentage):
    temp_frame = dataframe.copy()
    temp_frame = temp_frame.sort_values(by=['fi'], ascending=False)
    fi_arr = temp_frame["fi"].values
    new_len = (int)(percentage * len(fi_arr))
    new_fi = fi_arr[:new_len]
    new_frame = temp_frame.iloc[:new_len]

    plt.figure(figsize=(8, 6))
    plt.hist(fi_arr, bins=50, color="skyblue", label="all")
    plt.hist(new_fi, bins=50, color='yellow', label="picked")
    plt.legend()
    plt.show()
    return new_frame


def mutual_dimension_filtering(dataframe, abandon_outliers=True, rt_factor=1,
                               rt_filterthreshold=None, rt_distribution=None, mz_factor=1,
                               mz_filterthreshold=None, mz_distribution=None, order=0):
    temp_matched_data = dataframe.copy()
    orders = ['rt', 'mz']
    factors = [rt_factor, mz_factor]
    filter_thres = [rt_filterthreshold, mz_filterthreshold]
    distributions = [rt_distribution, mz_distribution]

    if order == 1:
        orders = orders[::-1]
        factors = factors[::-1]
        filter_thres = filter_thres[::-1]
        distributions = distributions[::-1]

    tmp1, idx_in_original_ref = successive_distance(temp_matched_data[orders[0] + '_ref'].values,
                                                    temp_matched_data[orders[0] + '_tar'].values,
                                                    distribution=distributions[0],
                                                    factor=factors[0], threshold=filter_thres[0], title=orders[0])

    if abandon_outliers:
        temp_matched_data = np.take(temp_matched_data, idx_in_original_ref, axis=0)

    tmp2, idx_in_original_tar = successive_distance(temp_matched_data[orders[1] + '_ref'].values,
                                                    temp_matched_data[orders[1] + '_tar'].values,
                                                    distribution=distributions[1],
                                                    factor=factors[1], threshold=filter_thres[1], title=orders[1])

    if not abandon_outliers:
        union_idx = list(set(idx_in_original_ref).union(set(idx_in_original_tar)))
        temp_matched_data = np.take(temp_matched_data, union_idx, axis=0)
    else:
        temp_matched_data = np.take(temp_matched_data, idx_in_original_tar, axis=0)
    plot_matches(temp_matched_data)

    return temp_matched_data


def SD_density_filtering(dataframe, factor=1,
                         filterthreshold=None, distribution=None,
                         feature=0, distance_thre=0.05, matches_threshold=30, distancefun="Euclidean"):
    if feature == 0:
        fea = 'rt'
    elif feature == 1:
        fea = 'mz'
    elif feature == 2:
        fea = 'fi'
    else:
        fea = 'rt'

    tmp_ref = dataframe[[(fea + "_ref"), (fea + "_tar")]].values
    tmp_tar = dataframe[[(fea + "_tar"), (fea + "_ref")]].values

    tmp_ref, idx_in_original_tmp_ref = successive_distance(dataframe[fea + "_ref"].values,
                                                           dataframe[fea + "_tar"].values,
                                                           distribution=distribution,
                                                           factor=factor, threshold=filterthreshold,
                                                           title=fea + ": Ref-Tar")

    tmp_tar, idx_in_original_tmp_tar = successive_distance(dataframe[fea + "_tar"].values,
                                                           dataframe[fea + "_ref"].values,
                                                           distribution=distribution,
                                                           factor=factor, threshold=filterthreshold,
                                                           title=fea + ": Tar-Ref")

    new_rt_reftar, new_rt_tarref = combine_reftar_tarref(tmp_ref, tmp_tar, distance_thre, matches_threshold,
                                                         distancefunction=distancefun)

    plt.figure(figsize=(8, 6))
    plt.scatter(tmp_ref[:, 0], tmp_ref[:, 1] - tmp_ref[:, 0], s=2, c='red', label="Ref-Tar")
    plt.scatter(tmp_tar[:, 1], tmp_tar[:, 0] - tmp_tar[:, 1], s=10, c='black',
                alpha=0.5, label="Tar-Ref")
    plt.scatter(new_rt_reftar[:, 0], new_rt_reftar[:, 1] - new_rt_reftar[:, 0], s=2, c='orange', label="good matches")
    plt.scatter(new_rt_tarref[:, 1], new_rt_tarref[:, 0] - new_rt_tarref[:, 1], c='orange', alpha=0.3)
    plt.xlabel('Ref ' + fea, fontsize=5)
    plt.ylabel('Ref ' + fea + ' diff', fontsize=5)
    plt.title("Result after SD & Density filtering")
    plt.legend()
    plt.show()

    onefeature_matches = np.concatenate((new_rt_reftar, new_rt_tarref[:, [1, 0]]), axis=0)
    return onefeature_matches


def score_fun(dataframe, rt_score, mz_score, fi_score=None, w_rt=1, w_mz=1, w_fi=0):
    if fi_score is None:
        scores = w_rt * np.abs(rt_score) + w_mz * np.abs(mz_score)
    else:
        scores = w_rt * np.abs(rt_score) + w_mz * np.abs(mz_score) + w_fi * np.abs(fi_score)

    scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
    plot_colored_scores(dataframe, scores)
    return scores


def unmatchedData_reflection(matched_df, ref_df, tar_df):
    ori_ref_str = np.sort(np.array(ref_df["rt"].astype(str) + "_" + ref_df["mz"].astype(str), dtype=np.str))
    ori_tar_str = np.sort(np.array(tar_df["rt"].astype(str) + "_" + tar_df["mz"].astype(str), dtype=np.str))

    matched_ref = np.sort(
        np.array(matched_df["rt_ref"].astype(str) + "_" + matched_df["mz_ref"].astype(str), dtype=np.str))
    matched_tar = np.sort(
        np.array(matched_df["rt_tar"].astype(str) + "_" + matched_df["mz_tar"].astype(str), dtype=np.str))

    return ori_ref_str, matched_ref

    ori_ref_idx, ori_tar_idx = [], []

    for i in range(matched_ref.size):
        idx_ref = np.where(ori_ref_str == matched_ref[i])[0]
        idx_tar = np.where(ori_tar_str == matched_tar[i])[0]
        print(idx_ref)
        print(idx_tar)

        ori_ref_idx.append(idx_ref)
        ori_tar_idx.append(idx_tar)
        # ori_ref_str = ori_ref_str[idx_ref:]
        # ori_tar_str = ori_tar_str[idx_tar:]

    return ori_ref_idx, ori_tar_idx


def plot_in_group(data):
    arr_ref = np.transpose(
        np.array(
            (data["rt_ref"].astype(str) + "_" + data["mz_ref"].astype(str) + "_" + data["fi_ref"].astype(str) + "_ref"),
            ndmin=2))
    arr_tar = np.transpose(
        np.array(
            (data["rt_tar"].astype(str) + "_" + data["mz_tar"].astype(str) + "_" + data["fi_tar"].astype(str) + "_tar"),
            ndmin=2))
    temp_groupmatches = np.concatenate((arr_ref, arr_tar), axis=1)

    G = nx.Graph()
    G.add_edges_from(temp_groupmatches)
    plt.figure(figsize=(8, 6))
    nx.draw(G, node_size=4)
    plt.show()

    # plt.figure(1, figsize=(8, 8))
    # # layout graphs with positions using graphviz neato
    # pos = nx.nx_agraph.graphviz_layout(G, prog="neato")
    # # color nodes the same in each connected subgraph
    # C = (G.subgraph(c) for c in nx.connected_components(G))
    # for g in C:
    #     c = [random.random()] * nx.number_of_nodes(g)  # random color...
    #     nx.draw(g, pos, node_size=40, node_color=c, vmin=0.0, vmax=1.0, with_labels=False)
    # plt.show()


def plot_RTMZ_style(data):
    ref_rt = data['rt_ref'].values
    ref_mz = data['mz_ref'].values
    ref_logfi = np.log10(data['fi_ref'].values)

    tar_rt = data['rt_tar'].values
    tar_mz = data['mz_tar'].values
    tar_logfi = np.log10(data['fi_tar'].values)

    fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(13, 13 / 2))

    im1 = axs[0].scatter(ref_rt, ref_mz, s=2, c=ref_logfi, cmap='bwr')
    axs[0].set_xlabel('RT', fontsize=7)
    axs[0].set_ylabel('MZ', fontsize=7)
    axs[0].set_title('Reference Feature Set', fontsize=8)

    im2 = axs[1].scatter(tar_rt, tar_mz, s=2, c=tar_logfi, cmap='bwr')
    axs[1].set_xlabel('RT', fontsize=7)
    axs[1].set_ylabel('MZ', fontsize=7)
    axs[1].set_title('Target Feature Set', fontsize=8)

    fig.colorbar(im1, ax=axs[0], pad=0.05)
    fig.colorbar(im2, ax=axs[1], pad=0.05)
    fig.show()
