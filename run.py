from LMBERT import LMBERT
from SiamesePreTrainer import SiamesePreTrainer
from LMBERTDataset import LMBERTDataset

import pandas as pd

model = LMBERT('bert-base-uncased', device="cpu")

df = pd.read_csv('dataset.csv')
base_language_sentences = df['translation'].tolist()
target_language_sentences = df['transliteration'].tolist()

train_dataset = LMBERTDataset(base_language_sentences[:32], target_language_sentences[:32])
eval_dataset = LMBERTDataset(base_language_sentences[32:48], target_language_sentences[32:48])

# trainer = SiamesePreTrainer(model, dataset, batch_size=2, epochs=1, model_dir='./LMBERT/model/0', save_best_model=False)

trainer = SiamesePreTrainer(
            model=model, 
            train_dataset=train_dataset, 
            eval_dataset=train_dataset,
            epochs=1,
            batch_size=16,
            device="cpu") 

trainer.train()
