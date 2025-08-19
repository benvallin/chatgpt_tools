# %% File description ----

# ChatGPTbq.py performs ChatGPT-powered bulk questioning.

# => Asks ChatGPT a user-defined question for every item in a user-supplied list. 
# - All items should represent possible values of the same concept, which can be anything (e.g: 'gene symbol', 'MSigDB gene set')
# - The question should be consistent with the concept the items represent. Ideally, it should be one that can be answered by 'yes' or 'no' (e.g: 'Is it involved in Parkinson's disease?')
 
# %% Define paths ----

# ==> To be set by user <==

# Path to working directory
work_dir_path = '/scratch/ben/rnaseq/'

# Path to .env file
dotenv_path = 'tools/chatgpt_tools/.env'

# %% Set up ----

# Import required libraries
import os
import numpy as np
import pandas as pd
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, conlist
from typing import Literal

# Append trailing '/' to directory paths if missing
for i in [x for x in dir() if x.endswith('_dir_path')]:
  globals()[i] = os.path.join(globals()[i], '')

# Set working directory
os.chdir(work_dir_path)

# Load .env file containing OPENAI_API_KEY
load_dotenv(dotenv_path)

client = OpenAI()

# %% Define analysis parameters ----  

# Define query_name
# => Query name used to construct output text file name
# query_name = 'deg_down_div67_129_involved_in_pd_dan_degeneration'
# query_name = 'deg_up_div67_129_involved_in_pd_dan_degeneration'
query_name = 'gs_fora_down_div67_exp_neg_enrich_in_pd_dan'
# query_name = 'gs_fora_up_div67_involved_in_pd_dan_degeneration'

# Define item_type
# => Type of items contained in input list
# item_type = 'gene symbol'
item_type = 'MSigDB gene set'

# Define item_list_path
# => Path to user's query item list file (csv)
# item_list_path = 'data/ben/general/output/dge_gsea/compare_gsea_dge_zinbwave_deseq2/deg_down_div67_129.csv'
# item_list_path = 'data/ben/general/output/dge_gsea/compare_gsea_dge_zinbwave_deseq2/deg_up_div67_129.csv'
item_list_path = 'data/ben/general/output/dge_gsea/compare_gsea_dge_zinbwave_deseq2/gs_fora_down_div67.csv'
# item_list_path = 'data/ben/general/output/dge_gsea/compare_gsea_dge_zinbwave_deseq2/gs_fora_up_div67.csv'

# Define system_role
system_role = 'researcher in molecular biology'

# Define user_question
# => User's question to be answered for every item in input list
# user_question = "Could it be involved in the neurodegeneration of Parkinson's disease dopaminergic neurons?"
# user_question = "Is its transcription expected to be downregulated in dopaminergic neurons in Parkinson's disease?"
user_question = "Is it expected to be negatively enriched in Parkinson's disease vs control dopaminergic neurons as part of a gene set enrichment analysis?"
# user_question = "Please give a description of this gene set"
# user_question = "What are the gene symbols included in this gene set?"

# Path to output directory
# => Path to output directory where ChatGPTbq results should be written to
out_dir_path = 'data/ben/general/output/chatgptbq/' 

# Define model
# => Model to use for ChatGPT request
model = 'gpt-5'

# Define method
# => Method to be used for ChatGPT request
method = 'client.chat.completions.parse'
# method = 'client.responses.parse'

# %% Check analysis parameter validity ----

# Check query_name validity
if not isinstance(query_name, str):
  raise TypeError("input query_name is not of type 'str'")

# Check item_type validity
if not isinstance(item_type, str):
  raise TypeError("input item_type is not of type 'str'")

# Check item_list_path validity
if not os.path.exists(item_list_path):
  raise FileNotFoundError('Input item_list_path does not exist')

if not item_list_path.endswith('.csv'):
  raise ValueError('Input item_list_path does not match a CSV file')

# Check system_role validity
if not isinstance(system_role, str):
  raise TypeError("input system_role is not of type 'str'")

# Check user_question validity
if not isinstance(user_question, str):
  raise TypeError("input user_question is not of type 'str'")

# Check model validity
gpt_models = [x.id for x in client.models.list().data if 'gpt' in x.id]

if model not in gpt_models:
    raise ValueError('Input model must be a valid GPT model')

# Check method validity
if method not in ['client.responses.parse', 'client.chat.completions.parse']:
  raise ValueError("Input method must be one of 'client.responses.parse' or 'client.chat.completions.parse'")

# Check out_dir_path validity
if not isinstance(out_dir_path, str):
  raise TypeError("input out_dir_path is not of type 'str'")

# Append trailing '/' to out_dir_path if missing
out_dir_path = os.path.join(out_dir_path, '')

# Create output directory if necessary
if not os.path.exists(out_dir_path):
  os.makedirs(out_dir_path)

# %% Prepare ChatGPT request ----

# Load and tidy user's item list
# => With the input item list file, extract item_list from:
# - the column named <item_type> if it exists
# - the first column otherwise
item_list = pd.read_csv(item_list_path, dtype=str)

if item_type in item_list.columns:
  
  item_list = item_list.loc[:, item_type]

else:
  
  item_list = item_list.iloc[:, 0]

item_list = item_list.drop_duplicates().sort_values()
 
# Record item list size
n_items = len(item_list)

# Define user_content
user_content = ', '.join(item_list)

# Define system_content
system_content = f'''
You are a {system_role}.
For all {n_items} items representing {item_type} values in the user-provided list, answer the following question: 
{user_question}
'''

# Define instructions
instructions = f"""
Output must match the following schema:
- results: a list of exactly {n_items} objects, in the same order as the input items.
Each object must be:
  - item: string (the original {item_type})
  - verdict: one of 'Yes', 'No', or 'Unknown'
  - reason: a concise explanation (1–3 sentences) supporting the verdict.
Rules:
- Keep the same item order as provided.
- Do not add or drop items.
- 'verdict' is the single source of truth; 'reason' must support it.
"""

# Combine system_content and instructions
final_instructions = ''.join([system_content, instructions])

# Define expected output format
class item_answer(BaseModel):
    item: str
    verdict: Literal['Yes', 'No', 'Unknown']
    reason: str

class output_format(BaseModel):
    results: conlist(item_answer, min_length=n_items, max_length=n_items)
    
# %% Send ChatGPT request ----

print(f"Using method '{method}' and model '{model}' with the following instructions:\n{final_instructions}\n")
   
if method == 'client.chat.completions.parse':
  
  response = client.chat.completions.parse(
    model=model,
    messages=[
      {'role':'system', 'content':final_instructions},
      {'role':'user', 'content':user_content}
      ],
    response_format=output_format,
    seed=123
  )
  
  response = response.choices[0].message.parsed
  
elif method == 'client.responses.parse':
  
  response = client.responses.parse(
    model=model,
    input=[
      {'role':'system', 'content':final_instructions},
      {'role':'user', 'content':user_content}
      ],
    text_format=output_format
    )
  
  response = response.output_parsed
  
# %% Tidy ChatGPT output ----

# Build dataframe output
df = []

for i in response.results:
            
  df.append(
    {'item': i.item,
     'answer': i.verdict,
     'explanation': f'{i.verdict} — {i.reason}'
    })

df = pd.DataFrame(df)

df['question'] = user_question

df['item_type'] = item_type

df = df.loc[:, ['question', 'item_type', 'item', 'answer', 'explanation']]

df['answer'] = df['answer'].astype('category') 

exp_cat_order = ['Yes', 'No', 'Unknown']

cur_cat_order = list(df['answer'].cat.categories)

new_cat_order = [x for x in exp_cat_order if x in cur_cat_order]

df['answer'] = df['answer'].cat.reorder_categories(new_cat_order)

df = df.sort_values(by=['answer', 'item'])

# Build text output
output = '\n'.join(['*** ChatGPTbq results ***\n',  
                    f'GPT model: {model} / Method: {method}\n',
                    f'Item list path: {item_list_path}\n',
                    f'Item type: {item_type}\n',  
                    f"User's question: {system_content}\n",
                    f'n queried items: {n_items}'])

n_pos_answers = len(df.loc[df['answer'] == 'Yes',])

if n_pos_answers == 0:
  
  output = '\n'.join([output,
                      f'n positive answers: {n_pos_answers}',
                      '\n------\n'])
  
else:
  
  pos_items = ', '.join(df.loc[df['answer'] == 'Yes', 'item'])
  
  output = '\n'.join([output,
                      f'n positive answers: {n_pos_answers}',
                      f'Items with positive answer: {pos_items}',
                      '\n------\n'])
  
for i in df['item']:
  
  item = f'Item: {i}'
  
  answer = str(df.loc[df['item'] == i, 'answer'].tolist()[0])
  answer = f'Answer: {answer}'
  
  explanation = df.loc[df['item'] == i, 'explanation'].tolist()[0]
  explanation = f'Explanation:\n{explanation}\n\n---\n'
  
  temp = '\n'.join([item, answer, explanation])
  
  output = '\n'.join([output, temp])
  
print(output)

# %% Write output to disk ----

# Write dataframe output
df.to_csv(''.join([out_dir_path, 'chatgptbq_', query_name, '.csv']), index=False)

# Write text output
with open(''.join([out_dir_path, 'chatgptbq_', query_name, '.txt']), 'w') as file:
  file.write(output)

# %%



