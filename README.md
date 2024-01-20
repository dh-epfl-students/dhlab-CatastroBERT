# CatastroBERT: Extreme Weather Event detection based on the Gazette de Lausanne dataset

<div align=center>
    <img src="ressources/bert.png" width="500" height="500" />
</div>

## OVERVIEW

This project was a journey towards exploring and extracting trends from the historical climate events reporting in the Gazette de Lausanne, a daily newspaper, with over 4 millions articles spanning from 1798 to 1990. Given the scale of the project, we leveraged natural language processing (NLP) techniques to efficiently process the data.

The core of the project was the development of a specific manually annotated dataset and the creation of a tailored language model (LM), CatastroBERT. This LM identified approximately 15,000 pertinent articles, demonstrating not only a high degree of precision and efficiency but also an ability to generalize and predict extreme weather events in years not included in its training. This robustness underscores CatastroBERT’s potential for a wide array of future research applications.

[CatastroBERT](https://huggingface.co/epfl-dhlab/CatastroBERT) and its experimental multilingual variant, [CatastroBERT-M](https://huggingface.co/epfl-dhlab/CatastroBERT), are now accessible for future research on HuggingFace. While CatastroBERT-M shows promise, it may require further tuning to optimize its performance across languages, reflecting our commitment to continually enhancing these tools’ capabilities. This project provides valuable tools and insights for ongoing and future research in the field.

## GOALS

## Example usage

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


model_name = "epfl-dhlab/CatastroBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification(model_name)

def predict(text):
    # Prepare the text data
    inputs = tokenizer.encode_plus(
        text,
        None,
        add_special_tokens=True,
        return_token_type_ids=True,
        padding=True,
        max_length=512,
        truncation=True,
        return_tensors='pt'
    )

    ids = inputs['input_ids'].to('cuda' if torch.cuda.is_available() else 'cpu')
    mask = inputs['attention_mask'].to('cuda' if torch.cuda.is_available() else 'cpu')

    # Get predictions
    with torch.no_grad():
        outputs = model(ids, mask)
        logits = outputs.logits

    # Apply sigmoid function to get probabilities
    probs = torch.sigmoid(logits).cpu().numpy()

    # Return the probability of the class (1)
    return probs[0][0]

#example usage 
text = "Un violent ouragan du sud-ouest est passé cette nuit sur Lausanne."
print(f"Prediction: {predict(text)}")
```
