from DS_19_pastcode.functions_updated import *
import numpy as np


# load two datasets to do the simple match
var_info_1 = 'D:/Downloads/jinweihao/2021-2022/DS-19/data/harderData/Airwave1xcms_SLPOS_scaled_VarInfo.xlsx'
var_info_2 = 'D:/Downloads/jinweihao/2021-2022/DS-19/data//harderData/MESA_phase1_LPOS_DC_VarInfo.xlsx'
ref_df = load_file(var_info_1, ['rtmed', 'mzmed', 'fimed'])
tar_df = load_file(var_info_2, ['rtmed', 'mzmed', 'fimed'])

matched_frame = feature_matching(ref_df, tar_df, [-5, 1], [-0.015, 0.007])
matched_frame2 = subgroups_filter(matched_frame, 5)
# plot_RTMZ_style(matched_frame)
# a, b = unmatchedData_reflection(matched_frame, ref_df, tar_df)
# res = subgroups_filter(matched_frame, 3)
temp_matched = mutual_dimension_filtering(matched_frame2)
rt_matches = SD_density_filtering(matched_frame2, factor=5, feature=0)
mz_matches = SD_density_filtering(matched_frame2, feature=1, matches_threshold=35)
xOut_rt, yOut_rt = trendline_smoother(rt_matches, "RT", distribution="normal", MAD=True,
                                   skip_successive_distance=True)[0:2]
xOut_mz, yOut_mz = trendline_smoother(mz_matches, "MZ", distribution="normal", MAD=True,
                                   skip_successive_distance=True, frac=0.4)[0:2]
rt_score = activated_residuals(matched_frame, xOut_rt, yOut_rt, feature_name='rt')
mz_score = activated_residuals(matched_frame, xOut_mz, yOut_mz, feature_name='mz')
scores = score_fun(matched_frame, rt_score, mz_score)
res_df = remove_duplicates(matched_frame, scores, threshold=0.7)

plot_in_group(matched_frame)

# -----------------------------------------------------------

var_info_1 = 'D:/Downloads/jinweihao/2021-2022/DS-19/data/harddata_pairs/pair1/Airwave1_PLPOS_xcms.csv'
var_info_2 = 'D:/Downloads/jinweihao/2021-2022/DS-19/data/harddata_pairs/pair1/Zett_plasma_Pos_knimet.csv'
ref_df = load_file(var_info_2, [0, 1, 2], header=None)
tar_df = load_file(var_info_1, [0, 1, 2], header=None)
ref_df = top_FI(ref_df, 0.6)
tar_df = top_FI(tar_df, 0.6)
matched_frame = feature_matching(ref_df, tar_df, [-6, 1], [-0.005, 0.015])
matched_frame = subgroups_filter(matched_frame, 10)
temp_matched = mutual_dimension_filtering(matched_frame)
rt_matches = SD_density_filtering(temp_matched, feature=0, matches_threshold=25, factor=2)
mz_matches = SD_density_filtering(temp_matched, feature=1, distance_thre=0.06, filterthreshold=0.05, matches_threshold=25)
xOut_rt, yOut_rt = trendline_smoother(rt_matches, "RT", distribution="normal", MAD=True,
                                   skip_successive_distance=True)[0:2]
xOut_mz, yOut_mz = trendline_smoother(mz_matches, "MZ", distribution="normal", MAD=True,
                                   skip_successive_distance=True, frac=0.35)[0:2]
ref_df = load_file(var_info_2, [0, 1, 2], header=None)
tar_df = load_file(var_info_1, [0, 1, 2], header=None)
matched_frame = feature_matching(ref_df, tar_df, [-6, 1], [-0.005, 0.015])
rt_score = activated_residuals(matched_frame, xOut_rt, yOut_rt, feature_name='rt')
mz_score = activated_residuals(matched_frame, xOut_mz, yOut_mz, feature_name='mz')
scores = score_fun(matched_frame, rt_score, mz_score)
res_df = remove_duplicates(matched_frame, scores, threshold=0.7)

# -----------------------------------------------

var_info_2 = 'D:/Downloads/jinweihao/2021-2022/DS-19/mesa_rotter/mesa.xlsx'
var_info_1 = 'D:/Downloads/jinweihao/2021-2022/DS-19/mesa_rotter/rotter.xlsx'
ref_df = load_file(var_info_2, [0, 1, 2], header=None)
tar_df = load_file(var_info_1, [0, 1, 2], header=None)
ref_df = top_FI(ref_df, 0.6)
tar_df = top_FI(tar_df, 0.6)
matched_frame = feature_matching(ref_df, tar_df, [-0.1, 0.1], [-0.005, 0.004])
matched_frame = subgroups_filter(matched_frame, 10)
temp_matched = mutual_dimension_filtering(matched_frame)
rt_matches = SD_density_filtering(temp_matched, feature=0)
mz_matches = SD_density_filtering(temp_matched, feature=1)
xOut_rt, yOut_rt = trendline_smoother(rt_matches, "RT", distribution="normal", MAD=True,
                                   skip_successive_distance=True)[0:2]
xOut_mz, yOut_mz = trendline_smoother(mz_matches, "MZ", distribution="normal", MAD=True,
                                   skip_successive_distance=True)[0:2]
ref_df = load_file(var_info_2, [0, 1, 2], header=None)
tar_df = load_file(var_info_1, [0, 1, 2], header=None)
matched_frame = feature_matching(ref_df, tar_df, [-0.1, 0.1], [-0.005, 0.004])
rt_score = activated_residuals(matched_frame, xOut_rt, yOut_rt, feature_name='rt')
mz_score = activated_residuals(matched_frame, xOut_mz, yOut_mz, feature_name='mz')
scores = score_fun(matched_frame, rt_score, mz_score)
res_df = remove_duplicates(matched_frame, scores, threshold=0.3)


def plot_matches_tmp(feature, rtfilter=None, mzfilter=None, tar_gofirst=False):

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
    axs[0].set_ylim([-0.1, 0.1])
    # axs[0].plot(rt_xfit, rt_yfit, '-k')

    axs[1].scatter(x_all_mz, y_all_mz, s=0.3, c='red')
    axs[1].set_xlabel('MZ ref', fontsize=7)
    axs[1].set_ylabel('MZ diff', fontsize=7)
    axs[1].set_ylim([-0.005, 0.004])
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

def plot_colored_scores_tmp(feature, score, normalization=False):
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
    axs[0].set_ylim([-0.1, 0.1])
    # axs[0].plot(rt_xfit, rt_yfit, '-k')

    axs[1].scatter(x_all_mz, y_all_mz, s=1, c=score, cmap='cool')
    axs[1].set_xlabel('MZ ' + title, fontsize=5)
    axs[1].set_ylabel('MZ diff', fontsize=5)
    axs[1].set_ylim([-0.005, 0.004])
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

plot_matches_tmp(res_df)
plot_colored_scores_tmp(res_df, res_df["score"].values)