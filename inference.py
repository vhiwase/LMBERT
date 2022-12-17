# https://github.com/codegram/LMBERT

from transformers import pipeline


# [
#  'audio-classification', 
#  'automatic-speech-recognition', 
#  'conversational', 
#  'feature-extraction', 
#  'fill-mask', 
#  'image-classification', 
#  'image-segmentation', 
#  'ner', 
#  'object-detection', 
#  'question-answering', 
#  'sentiment-analysis', 
#  'summarization', 
#  'table-question-answering', 
#  'text-classification', 
#  'text-generation', 
#  'text2text-generation', 
#  'token-classification', 
#  'translation', 
#  'visual-question-answering', 
#  'vqa', 
#  'zero-shot-classification', 
#  'zero-shot-image-classification', 
#  'translation_XX_to_YY'
# ]
LMBERT_fill_mask  = pipeline("fill-mask", model="LMBERT/best-model-3.412820339202881", tokenizer="LMBERT/best-model-3.412820339202881")
text = " mera naam [MASK] hai"
results = LMBERT_fill_mask(text)

