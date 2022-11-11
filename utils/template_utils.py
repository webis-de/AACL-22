import os
import pandas as pd
import os.path as path
import re
from pprint import pprint
import readtime
from jinja2 import Template
import numpy as np



from jinja2 import Template
from bs4 import BeautifulSoup

def read_json(path):
    import json
    with open(path) as json_file:
        data = json.load(json_file)
    return data

def get_arg_length(sentences):
    import nltk
    from nltk.tokenize import sent_tokenize
    number_of_sentences = sent_tokenize(sentences)
    return (len(number_of_sentences))

def wrap_paragraph(row):
    return '<div class="container-box"><p span style="font-:normal">{}</p><div class="select-box">select</div></div>'.format(' '.join(row.split()))


def delete_keys_from_dict(dict_del, lst_keys):
    for k in lst_keys:
        try:
            del dict_del[k]
        except KeyError:
            pass
    for v in dict_del.values():
        if isinstance(v, dict):
            delete_keys_from_dict(v, lst_keys)
    return dict_del


def shuffle_list(a_lis):
    import random
    random.seed(8)
    return sorted(a_lis, key=lambda k: random.random())



def format_table(a_df):
    from jinja2 import Environment, BaseLoader
    col1 = a_df.columns[0]
    col2 = a_df.columns[1]
    part_1 = ' '.join(a_df[col1].tolist())
    part_2 = ' '.join(a_df[col2].tolist())
    html_table = '''<section class="row"> <div class="layout-one"> <h3>{{col_1}}</h3> <div> {{part_1}} </div></div><div class="layout-two"><h3>{{col_2}}</h3><div>{{part_2}}</div></div></section>''' 
    template = Environment(loader=BaseLoader()).from_string(html_table)
    return template.render(col_1 = col1, col_2=col2, part_1= part_1, part_2 = part_2)



from bs4 import BeautifulSoup


def create_tuple(a_df):
    return zip(a_df['argument'].tolist(), a_df['stance'].tolist())




def wrap(output_path_unlabeled, tupled_list):
    
    soup = BeautifulSoup(open(output_path_unlabeled), 'html.parser')
    
    for i, elem in enumerate(tupled_list):
        for j, e in enumerate(elem):

            matches =\
            soup.find_all(lambda x: x.text == e[0])
            for k, match in enumerate(matches):
                target = match.parent.find("div", {"class": "select-box"})
            
                # TODO 1, 2, 3
            
                if e[1] == 'Con':
                    r = '1'
                elif e[1] == "Pro":
                    r = '2'
                elif e[1] == "Unknown":
                    r = '3'
                else:
                    r = '4'
                        
                s = f'''<div class="container"><input type="radio" value="1-{r}" name="question-select_{str(i)+str(j)+str(k)}" class="answer_question" data-id-question="type3" required >Con</input>
                <input type="radio" value="2-{r}" name="question-select_{str(i)+str(j)+str(k)}" class="answer_question" data-id-question="type3" required >Pro</input>
                <input type="radio" value="3-{r}" name="question-select_{str(i)+str(j)+str(k)}" class="answer_question" data-id-question="type3" required >Unknown</input>
                <input type="radio" value="4-{r}" name="question-select_{str(i)+str(j)+str(k)}" class="answer_question" data-id-question="type3" required >Neutral</input></div>'''
            

                target.string = target.text.replace("select", s)

            
            # fix the path
            with open(output_path_unlabeled, "w") as file:
                file.write(soup.prettify(formatter=None))            
            



def create_argument(df, a_dict):
    import numpy as np # Remove this for getting different results for each run
    np.random.seed(123)

    topic = df['topic'].tolist()[0]

    df_pro = df[df['stance'] == 'Pro']
    #import pdb; pdb.set_trace() # debugging starts here
    rows = np.random.choice(df_pro.index.values, a_dict['arg_line_A']['pro'], replace=False)
    chunk_a_1 = df_pro.loc[rows]

    to_filter = chunk_a_1['index'].tolist()
    updated_df = df[~df['index'].isin(to_filter)]


    df_con = updated_df[updated_df['stance'] == 'Con']
    rows = np.random.choice(df_con.index.values, a_dict['arg_line_A']['con'], replace=False)
    chunk_a_2 = df_con.loc[rows]

    to_filter = chunk_a_2['index'].tolist()
    updated_df = updated_df[~updated_df['index'].isin(to_filter)]
    

    arg_line_a = pd.concat([chunk_a_1, chunk_a_2], ignore_index=True)
    arg_line_a = arg_line_a.sample(frac=1, random_state=0).reset_index(drop=True)



    df_pro = updated_df[updated_df['stance'] == 'Pro']
    rows = np.random.choice(df_pro.index.values, a_dict['arg_line_B']['pro'], replace=False)
    chunk_b_1 = df_pro.loc[rows]

    # Update the DF
    to_filter_2 = chunk_b_1['index'].tolist()
    updated_df = updated_df[~updated_df['index'].isin(to_filter_2)]
    

    df_con = updated_df[updated_df['stance'] == 'Con']
    rows = np.random.choice(df_con.index.values, a_dict['arg_line_B']['con'], replace=False)
    chunk_b_2 = df_con.loc[rows]

    # Update the DF
    to_filter_2 = chunk_b_2['index'].tolist()
    updated_df = updated_df[~updated_df['index'].isin(to_filter_2)]

    arg_line_b = pd.concat([chunk_b_1, chunk_b_2], ignore_index=True)
    arg_line_b = arg_line_b.sample(frac=1, random_state=0).reset_index(drop=True)



    line_a = arg_line_a.argument.tolist()
    line_b = arg_line_b.argument.tolist()
    
    ert = readtime.of_text(' '.join([item for sublist in [line_a, line_b] for item in sublist]), wpm=260)
    

    
    stances_line_a = arg_line_a.stance.tolist()
    stances_line_b = arg_line_b.stance.tolist()

    
    a = pd.DataFrame(line_a, columns=['A'])
    b = pd.DataFrame(line_b, columns=['B'])
        
    d = pd.concat([a, b], axis=1)
    d['A'] = d['A'].apply(wrap_paragraph)
    d['B'] = d['B'].apply(wrap_paragraph)

    
    return format_table(d.iloc[np.random.permutation(len(d))]), topic, ert.seconds



def create_view_topic_stance(arg_data):
    a_lis = [tuple([e[0],e[1]]) for e in arg_data]
    final_string = []
    for e in a_lis:
        s = f'''<div class="container"> {e[1]}?:
        <div  style="margin: -2px 15px 14px">
            <input type="radio" value="agree" name="personal_stance_{e[0]}" class="answer_question" data-id-question="agree" required >Agree</input>
            <input type="radio" value="disagree" name="personal_stance_{e[0]}" class="answer_question" data-id-question="agree" required >Disagree</input>
            <input type="radio" value="neutral" name="personal_stance_{e[0]}" class="answer_question" data-id-question="agree" required >Neutral</input>
            <input type="radio" value="unknown" name="personal_stance_{e[0]}" class="answer_question" data-id-question="agree" required >Unknown</input>
        </div>
        </div>'''
        final_string.append(s)
    return " ".join(final_string)


def split_list(a_list):
    l = []
    half = len(a_list)//2    
    return a_list[:half], a_list[half:][:-1], a_list[-1]

def produce_template(topics, args, template_path, output_path):
    template = Template(open(template_path).read())
    idx = list(map(lambda x:x+1, list(range(0, len(topics)))))
    arg_data = list(zip(idx, topics, args))
    
    part_1, part_2, part_3 = split_list(arg_data)
    

    with open(output_path, 'w') as f:
        f.write(template.render(arg_data_0 = create_view_topic_stance(arg_data),
                                arg_data_1 = part_1,
                                arg_data_2 = part_2, 
                                arg_data_3 = part_3))
        
def shuffle_df(df, n=1, axis=0):     
    df = df.copy()
    for _ in range(n):
        df.apply(np.random.shuffle, axis=axis)
    return df


        
def create_setup(dic_list, arg_list, **kwargs):
    read_time = []
    args = []
    topics = []

    if kwargs.get('shuffling', True):
        dic_list = shuffle_list(dic_list)
    else:
        pass
    for x, y in zip(arg_list, dic_list):
        
        try:
            output, topic, ert = create_argument(x, y)
            args.append(output)
            topics.append(topic)
            read_time.append(ert)
        except ValueError:
            pass 
    return topics, args



def t_tuple(e):
    l = []
    for i in e:
        l.append(abs(i['pro']-i['con']))
    return tuple(l)




