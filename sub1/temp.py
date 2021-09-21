# from transformers import RobertaTokenizer
# text_model_path = "roberta-large" # '/data/project/rw/rung/02_source/model/roberta-large'
# model_text_tokenizer = RobertaTokenizer.from_pretrained(text_model_path)
# special_token_list = ['[USER]', '[SYSTEM]']
# special_tokens = {'additional_special_tokens': special_token_list}
# model_text_tokenizer.add_special_tokens(special_tokens)

# from transformers import DeiTFeatureExtractor
# image_model_path = "facebook/deit-base-distilled-patch16-224" # '/data/project/rw/rung/02_source/model/deit-base-distilled-patch16-224'
# image_feature_extractor = DeiTFeatureExtractor.from_pretrained(image_model_path)

from transformers import GPT2Tokenizer
gpt_model_path = "gpt2-large" # '/data/project/rw/rung/02_source/model/gpt2-large'
gpt_tokenizer = GPT2Tokenizer.from_pretrained(gpt_model_path)