from datasets import load_dataset
from transformers import LlamaTokenizerFast

max_len = 512
dataset_path = 'bawe_sample_set.csv'
prompt_path = 'prompt.txt'

checkpoint = 'openlm-research/open_llama_7b_v2'

bool_to_label = {
    True: 'earlier',
    False: 'later',
}

ds = load_dataset('csv', data_files=dataset_path)
tokenizer = LlamaTokenizerFast.from_pretrained(checkpoint)

# 不能截断模板，只能截断中间的文章，因此计算去除模板文本后的最大文章长度
prompt_text = open(prompt_path, 'r', encoding='utf-8').read()
prompt_len = tokenizer(prompt_text, return_tensors='pt').input_ids.size()[1]
per_essay_max_len = int((max_len - prompt_len) / 2)


def tokenize_function(example):     # 只截断，因为SFTTrainer有参数可以设置填充
    map_batch_size = len(example['text1'])
    labels = [bool_to_label[i] for i in example['bool']]
    essay1 = tokenizer(example['text1'], max_length=per_essay_max_len, return_tensors='pt')
    essay2 = tokenizer(example['text2'], max_length=per_essay_max_len, return_tensors='pt')

    # 截断后的token索引
    trunc_token_index = per_essay_max_len - 1

    # 截断后的char索引
    essay1_trunc_char_indices = [essay1.token_to_chars(i, trunc_token_index).end for i in range(map_batch_size)]
    essay2_trunc_char_indices = [essay2.token_to_chars(i, trunc_token_index).end for i in range(map_batch_size)]

    # 利用char索引截断原文本
    text_set = []
    for i in range(map_batch_size):
        essay1_trunc_text = example['text1'][i][:essay1_trunc_char_indices[i]]
        essay2_trunc_text = example['text2'][i][:essay2_trunc_char_indices[i]]
        text_set.append(prompt_text.format(essay1=essay1_trunc_text, essay2=essay2_trunc_text, label=labels[i]))
    return {'text': text_set}


ds = ds.map(tokenize_function, remove_columns=ds.column_names['train'], batched=True)
ds['train'].to_pandas().to_csv('trunc_df.csv')
