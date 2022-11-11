import pandas as pd
import numpy as np
from nltk.metrics.agreement import AnnotationTask
from nltk.metrics import interval_distance, binary_distance
import datetime
from matplotlib import pyplot as plt
import seaborn as sns
import operator
from subprocess import PIPE, run
import pathlib

sns.set_style("darkgrid")



def get_alpha(triplet):
#   print('-> Krippendorff\'s alpha: {:.8f}'.format(AnnotationTask(triplet, distance = interval_distance).alpha()))
    return AnnotationTask(triplet, distance = interval_distance).alpha()


def removekey(d, key):
    r = dict(d)
    del r[key]
    return r


def transform_dict(a_dict):
    keys = [str(e[0]) for e in list(a_dict.keys())]
    values = list(a_dict.values())
    return removekey(dict(zip(keys, values)), 'False')



def get_summary_choice(data):
    return pd.DataFrame(data).fillna(0).T


def conf_mat(y_true, y_pred, title):
    import seaborn as sns
    import matplotlib.pyplot as plt     
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax, fmt='d', cmap="OrRd") #annot=True to annotate cells
    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title(title)
    ax.xaxis.set_ticklabels(['Arg A', 'Arg B', 'None'])
    return ax.yaxis.set_ticklabels(['Arg A', 'Arg B', 'None'])
    

def get_alpha(col1, col2, col3, batch):
    triplet = list(zip(batch.id_worker, [col1] * batch.shape[0], batch[col1])) + list(zip(batch.id_worker, [col2] * batch.shape[0], batch[col2])) + list(zip(batch.id_worker, [col3] * batch.shape[0], batch[col3]))
    return AnnotationTask(triplet, distance = interval_distance).alpha()




def get_barchart(summary, title, caption, batch):
    ax= summary.plot(kind='bar', rot=0)
    plt.title(label=title)
    plt.figtext(0.5, 0.01, caption, wrap=True, horizontalalignment='center', fontsize=12)
    for p in ax.patches:
        percentage = '{:.1f}%'.format(100 * p.get_height()/float(len(batch)))
        x = p.get_x() + p.get_width()
        y = p.get_height()
        ax.annotate(percentage, (x, y),ha='center')
    plt.show()



def validate_time(arr):
    return 'approved' if arr[ arr >= 180 ].size >= 4 else 'internal reject' if arr[ arr >= 90 ].size >= 4 else 'rejected'

    

def validate_answers(a_df, interval_1, interval_2):
    a_lis = []
    for e in a_df.index:
        a_lis.append(validate_time(a_df.iloc[e].to_numpy()[interval_1:interval_2]))
    
    #assign
    a_df['quality_control'] = a_lis
    return a_df
    



def transform_one_hot_df(a_df, a_lis_of_cols, col_name):
    age_groups_df = a_df[a_lis_of_cols]
    age_groups_df = age_groups_df.set_index('id_worker')
    age_groups_df = age_groups_df.dot(age_groups_df.columns).to_frame(col_name).reset_index()
    age_groups_df[col_name] = age_groups_df[col_name].apply(lambda x : ''.join(x.split('.')[-1:]))
    return age_groups_df

    

def read_json(path):
    import json
    with open(path) as json_file:
        data = json.load(json_file)
    return data




def get_mace_competence(a_df, hit_id):
    pd.options.mode.chained_assignment = None  # default='warn'

    # Section 1

    part_1 = a_df[['id_worker', 'question1_1_1.arg_a', 'question1_1_1.arg_b', 'question1_1_1.none']]

    part_2 = a_df[['id_worker', 'question1_1_2.arg_a', 'question1_1_2.arg_b', 'question1_1_2.none']]

    # Section 2
    part_3 = a_df[['id_worker', 'question1_1_2.arg_a', 'question1_1_2.arg_b', 'question1_1_2.none']]

    #Section 3
        #  1. Which text has more pro stances (paragraphs that agree with the topic)? 
    part_4 = a_df[['id_worker', 'question2_1_3.val1', 'question2_1_3.val2', 'question2_1_3.val3']]

        #  2. Which text has more con stances (paragraphs that disagree with the topic)? 

    part_5 = a_df[['id_worker', 'question2_2_3.val1', 'question2_2_3.val2', 'question2_2_3.val3']]

        #  3. Which text is more one-sided? 


    part_6 = a_df[['id_worker', 'question2_3_3.val1', 'question2_3_3.val2', 'question2_3_3.val3']]

        # 4. How sure you are?

    part_7 = a_df[['id_worker', 'question2_4_3.val1', 'question2_4_3.val2', 'question2_4_3.val3']]

    #section 4

        #  1. Which text has more pro stances (paragraphs that agree with the topic)? 

    part_8 = a_df[['id_worker', 'question3_3_4.val1', 'question3_3_4.val2', 'question3_3_4.val3']]

        #  2. Which text has more con stances (paragraphs that disagree with the topic)?

    #section 5

        #  1. We believe that A is more one-sided, are you agree? 
    part_9 = a_df[['id_worker', 'question1_1_repeatedLabeled1.no',
                 'question1_1_repeatedLabeled1.none', 'question1_1_repeatedLabeled1.yes']]

        # 2.  How sure you are? 
    part_10 =  a_df[['id_worker', 'question1_2_repeatedLabeled1.val1',
                'question1_2_repeatedLabeled1.val2', 'question1_2_repeatedLabeled1.val3']]



    mace_data_format_path = f"../data/crowdsourced/mace_temp/{hit_id}_mace_data.csv"
    competence_path = "../scripts/competence"

    mace_data = pd.concat([part_1, part_2, part_3, part_4,
           part_5, part_6, part_7, part_8, 
           part_9, part_10], axis=1).T.drop_duplicates()

    mace_data = mace_data.rename(columns=mace_data.iloc[0]).drop(mace_data.index[0])
    mace_cols = mace_data.columns

    mace_data.to_csv(mace_data_format_path, index=False, header=False)
    command = ['java', '-jar', '../MACE.jar', mace_data_format_path]
    
    result = run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True)

    
    mace_competence = pd.DataFrame({"id_worker":mace_cols, 
                                    "mace_competence":list(np.loadtxt("../scripts/competence"))})

    pathlib.Path("../scripts/competence").unlink()
    pathlib.Path("../scripts/prediction").unlink()


    time = a_df[["id_worker", "start_time",
                "submit_time", "time_elapsed_1", 
                "time_elapsed_2", "time_elapsed_3",
                "time_elapsed_id_4", "time_elapsed_last"]]
    time['start_time'] =  pd.to_datetime(time['start_time'], infer_datetime_format=True)
    time['submit_time'] =  pd.to_datetime(time['submit_time'], infer_datetime_format=True)
    time['δ_minutes'] = (time['submit_time'] - time['start_time']).astype("timedelta64[m]")

    output = pd.concat([time, mace_competence], axis=1).T.drop_duplicates().T
    output = output.drop(['start_time', 'submit_time'], axis=1)
    return output



def plot_mace_time_corr(a_df):
    import seaborn as sns
    corr = a_df[['mace_competence', 'δ_minutes']].astype(float).corr()
    heatmap =sns.heatmap(corr, annot = True, fmt='.2g',cmap= 'coolwarm')
    return heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':18}, pad=12);


def get_krippendorff(a_df):
    from simpledorff import calculate_krippendorffs_alpha_for_df
    a_lis = list(a_df.columns)
    data = transform_one_hot_df(a_df, a_lis,'annotation')
    data['document_id'] = a_lis[1].split('.')[0]
    return calculate_krippendorffs_alpha_for_df(data,
                                                experiment_col='document_id',
                                                annotator_col='id_worker',
                                                class_col='annotation')


def create_format_for_quica(a_df):
    a_df = a_df.T
    
    a_df.rename(columns=a_df.iloc[0], inplace = True)
    a_df.drop(a_df.index[0], inplace = True)
    a_df = a_df*1
    data = a_df.values.T    
    

    dataframe = pd.DataFrame({
        "coder_0" : data[0].tolist(),
        "coder_1" : data[1].tolist(), 
        "coder_2" : data[2].tolist(),
        "coder_3" : data[3].tolist(),
        "coder_4" : data[4].tolist(),
        "coder_5" : data[5].tolist(),
        "coder_6" : data[6].tolist(),
        "coder_7" : data[7].tolist(),
        "coder_8" : data[8].tolist()})

    return dataframe



    

