# hist_skew_score is a function to plot the histplot along with skew score
import matplotlib.pyplot as plt
from scipy.stats import skew
import numpy as np


def hist_skew_score(continuous_feature, df):
    df[continuous_feature].hist(bins=25, figsize=(20, 20))
    for feature in continuous_feature:
        data = df.copy()
        print(feature, ' skew score ----> ', skew(data[feature]))


# remove_skewed_outliers is used to d
def remove_skewed_outliers(feature, data, skew_score):
    if (skew_score > 0.25 or skew_score < -0.25) == True:
        ##Computing interquantile range to find boundaries

        iqr = data[feature].quantile(0.75) - data[feature].quantile(0.25)

        lower_bridge = data[feature].quantile(0.25) - (iqr * 3)

        upper_bridge = data[feature].quantile(0.25) + (iqr * 3)

        data.loc[data[feature] <= lower_bridge, feature] = lower_bridge

        data.loc[data[feature] >= upper_bridge, feature] = upper_bridge


# Using correlation function we can select highly correlated features

def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if corr_matrix.iloc[i, j] > threshold:  # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr

# METRICS

# 1. Absolute Avg Treatment Effect
def abs_ate(effect_true, effect_pred):
    """
    Absolute error for the Average Treatment Effect (ATE)
    :param effect_true: true treatment effect value
    :param effect_pred: predicted treatment effect value
    :return: absolute error on ATE
    """
    return abs(sum(effect_true)/len(effect_true) - sum(effect_pred)/len(effect_pred))

# 2. Precision in Estimation of Heterogeneous Effect(PEHE)
def pehe(effect_true, effect_pred):
    """
    Precision in Estimating the Heterogeneous Treatment Effect (PEHE)
    :param effect_true: true treatment effect value
    :param effect_pred: predicted treatment effect value
    :return: PEHE
    """
    return np.sqrt(np.mean((effect_true - effect_pred)**2))

# get_ps_weights to compute the sample weights
def get_ps_weights(clf, x, t):
  ti = np.squeeze(t)
  clf.fit(x, ti)
  ptx = clf.predict_proba(x).T[1].T + 0.0001 # add a small value to avoid dividing by 0
  # Given ti and ptx values, compute the weights wi (see formula above):
  wi =  ti/(ptx)  +  (1 - ti)/(1 - (ptx))   # YOUR CODE HERE
  return wi


# abs_att to compute absolute average treatment effect on treated
def abs_att(effect_pred, yf, t, e):
    """
    Absolute error for the Average Treatment Effect on the Treated
    :param effect_pred: predicted treatment effect value
    :param yf: factual (observed) outcome
    :param t: treatment status (treated/control)
    :param e: whether belongs to the experimental group
    :return: absolute error on ATT
    """
    att_true = np.mean(yf[t > 0]) - np.mean(yf[(1 - t + e) > 1])
    att_pred = np.mean(effect_pred[(t + e) > 1])

    return np.abs(att_pred - att_true)


# policy_risk to compute policy
def policy_risk(effect_pred, yf, t, e):
    """
    Computes the risk of the policy defined by predicted effect
    :param effect_pred: predicted treatment effect value
    :param yf: factual (observed) outcome
    :param t: treatment status (treated/control)
    :param e: whether belongs to the experimental group
    :return: policy risk
    """
    # Consider only the cases for which we have experimental data (i.e., e > 0)
    t_e = t[e > 0]
    yf_e = yf[e > 0]
    effect_pred_e = effect_pred[e > 0]

    if np.any(np.isnan(effect_pred_e)):
        return np.nan

    policy = effect_pred_e > 0.0
    treat_overlap = (policy == t_e) * (t_e > 0)
    control_overlap = (policy == t_e) * (t_e < 1)

    if np.sum(treat_overlap) == 0:
        treat_value = 0
    else:
        treat_value = np.mean(yf_e[treat_overlap])

    if np.sum(control_overlap) == 0:
        control_value = 0
    else:
        control_value = np.mean(yf_e[control_overlap])

    pit = np.mean(policy)
    policy_value = pit * treat_value + (1.0 - pit) * control_value

    return 1.0 - policy_value

# change_to_1d is used convert to 1 Dimensional
def change_to_1d(y_test):
    s = []
    for el in y_test:
        s.append(float(el))
    y_test = np.array(s)
    return y_test