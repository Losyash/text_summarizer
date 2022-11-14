import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "IlyaGusev/rugpt3medium_sum_gazeta"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

article_text = "Актуальность проблемы. Электронная информация играет все большую роль во всех сферах жизни современного общества. \
В последние годы объем научно-технической текстовой информации в электронном виде возрос настолько, что возникает угроза \
обесценивания этой информации в связи с трудностями поиска необходимых сведений среди множества доступных текстов.  \
Развитие информационных ресурсов Интернет многократно усугубило проблему информационной перегрузки. В этой ситуации \
особенно актуальными становятся методы автоматизации реферирования текстовой информации, то есть методы получения сжатого \
представления текстовых документов–рефератов (аннотаций). Постановка проблемы автоматического реферирования текста и \
соответственно попытки ее решения с использованием различных подходов предпринимались многими исследователями. \
"

text_tokens = tokenizer(
    article_text,
    max_length=600,
    add_special_tokens=False, 
    padding=False,
    truncation=True
)["input_ids"]
input_ids = text_tokens + [tokenizer.sep_token_id]
input_ids = torch.LongTensor([input_ids])

output_ids = model.generate(
    input_ids=input_ids,
    no_repeat_ngram_size=4
)

summary = tokenizer.decode(output_ids[0], skip_special_tokens=False)
summary = summary.split(tokenizer.sep_token)[1]
summary = summary.split(tokenizer.eos_token)[0]
print(summary)
