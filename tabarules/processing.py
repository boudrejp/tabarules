# functions for preprocessing standard dataframes into 
# list representations for mlxtend's association rules

import os
import numpy as np
import pandas as pd
import mlxtend
from math import ceil

# functions

def float_processer(series, \
                    title, \
                    cutoffs = 4, \
                    na_action = ['label', 'return_na'][1], \
                    print_labels = False):
    '''
    SUMMARY: Gives bin labeling to float data based off of splits in the data

    INPUTS:
    - series: A pandas series object of type float
    - title: A title used for the feature of type string
    - cutoffs: the number of bins to place data into. The more bins, the more granular.
    - na_action: desired action for missing values. "label" will return a string with "title_is_na", "return_na" will return NA
    - print_labels: a boolean value to print out the label values before assignment. Useful for debugging.

    OUTPUTS:
    - a pandas series object of the bin-labelled data

    NOTES:
    - string labels can get long. opt for shorter ones if possible.
    - string labels will come out in the format "title_lessthan_cutoffval" or "title_morethan_cutoffval"
    - Cutoff values are calulated by sorting the data and taking the value at each index 1/cutoffs way through the data.

    '''


    # find splits in data based on ordering
    not_nas = np.where(np.logical_not(np.isnan(series)))
    not_na_series = series[not_nas[0]]
    len_series = not_na_series.shape[0]
    len_tot_series = series.shape[0]
    series_sorted = not_na_series.sort_values().reset_index(drop = True)
    cutoff_val_ind = [ceil( (i+1) / cutoffs * len_series) for i in range(cutoffs-1)]
    cutoff_vals = series_sorted[cutoff_val_ind].values

    # label creation
    str_labels = [title + "_lessthan_" + str(val) for val in cutoff_vals]
    str_labels.append(title + "_morethan_" + str(cutoff_vals[cutoffs-2]))

    if print_labels:
        print(str_labels)

    output_list = []
    # likely can have better parallelization if needed
    # populate a list with the appropriate label for the data
    for i in range(len_tot_series):
        if pd.isna(series[i]):
            if na_action == "label":
                output_list.append(title + "_is_na")
                continue
            elif na_action == "return_na":
                output_list.append(np.nan)
                continue
        for j in range(len(cutoff_vals)):
            if series[i] <= cutoff_vals[j]:
                output_list.append(str_labels[j])
                break
            elif j == len(cutoff_vals)-1:
                output_list.append(str_labels[len(cutoff_vals)])
                break

    # can debate whether list or series is better later
    return pd.Series(output_list)


def boolean_processer(series, title, which_yes = 1, na_action = ['label', 'return_na'][1], print_labels = False):
    '''
    SUMMARY: Gives bin labeling to boolean data

    INPUTS:
    - series: A pandas series object of type float
    - title: A title used for the feature of type string
    - which_yes: which number in the data is used as the "yes" marking
    - print_labels: a boolean value to print out the label values before assignment. Useful for debugging.

    OUTPUTS:
    - a pandas series object of the bin-labelled data

    NOTES:
    - string labels can get long. opt for shorter ones if possible.
    - string labels will come out in the format "title_yes" or "title_no"
    - Cutoff values are calulated by sorting the data and taking the value at each index 1/cutoffs way through the data.

    '''
    len_series = series.shape[0]

    # label creation
    str_labels = [title + "_no", title + "_yes"]

    if print_labels:
        print(str_labels)

    output_list = []
    # likely can have better parallelization if needed
    # populate a list with the appropriate label for the data

    ###
    for i in range(len_series):
        # deal with nas first
        if pd.isna(series[i]):
            if na_action == "label":
                output_list.append(title + "_is_na")
                continue
            elif na_action == "return_na":
                output_list.append(np.nan)
                continue
        # logic for dealing with regular values
        if series[i] == which_yes:
            output_list.append(str_labels[1])
        else:
            output_list.append(str_labels[0])

    # can debate whether list or series is better later
    return pd.Series(output_list)


def cat_processer(series, title, na_action = ['label', 'return_na'][1], print_labels = False):
    '''
    SUMMARY: Gives bin labeling to categorical data based off of splits in the data

    INPUTS:
    - series: A pandas series object of type float
    - title: A title used for the feature of type string
    - print_labels: a boolean value to print out the label values before assignment. Useful for debugging.

    OUTPUTS:
    - a pandas series object of the bin-labelled data

    NOTES:
    - string labels can get long. opt for shorter ones if possible.
    - string labels will come out in the format "title_is_label"

    '''


    # find splits in data based on ordering
    len_series = series.shape[0]
    not_nas = np.where(np.logical_not(pd.isna(series)))
    not_na_series = series[not_nas[0]]
    unique_vals = not_na_series.unique()

    # label creation
    str_labels = [title + "_is_" + val for val in unique_vals]

    if print_labels:
        print(str_labels)

    output_list = []
    # likely can have better parallelization if needed
    # populate a list with the appropriate label for the data
    for i in range(len_series):
        # deal with nas first
        if pd.isna(series[i]):
            if na_action == "label":
                output_list.append(title + "_is_na")
                continue
            elif na_action == "return_na":
                output_list.append(np.nan)
                continue
        for j in range(unique_vals.shape[0]):
            if series[i] == unique_vals[j]:
                output_list.append(str_labels[j])
                next

    # can debate whether list or series is better later
    return pd.Series(output_list)


# function to take individual series and impose the right function on it
def list_featurize_series(series,\
                          title, \
                          cutoffs = 4, \
                          which_yes = 1, \
                          na_action = ['label', 'return_na'][1], \
                          print_labels = False):
    # first figure out what type the series is
    typeof = series.dtypes

    # categorical
    if typeof == 'O':
        ret_series = cat_processer(series, \
                                   title, \
                                   na_action, \
                                   print_labels)

    # only boolean
    elif np.array_equal(series.unique()[~pd.isna(series.unique())],\
                        np.array([0,1])):

        ret_series = boolean_processer(series, \
                                       title, \
                                       which_yes, \
                                       na_action, \
                                       print_labels)

    # floats/ints
    else:
        ret_series = float_processer(series, \
                    title, \
                    cutoffs, \
                    na_action, \
                    print_labels)

    return ret_series

# function to take entire dataframe and process into listed strings
# todo: customizable for non-defaults
def list_featurize_df(df, \
                      index_list = [i for i in range(df.shape[0])]):
    # loop thru series
    df_out = pd.DataFrame()
    for column in df.columns:
        out_series = list_featurize_series(df[column], title = column)
        df_out[column] = out_series

    store =[]
    # upon completion, go thru and make a key-value store
    for i in range(df.shape[0]):
        writing_vals = df_out.loc[i].values[~pd.isna(df_out.loc[i].values)]
        store.append([val for val in writing_vals])

    return(store)
    # then iterate thru and make row-wise key value store
