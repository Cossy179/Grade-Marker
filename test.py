from transformers import BertForSequenceClassification, BertTokenizerFast, BertConfig
import torch

def test(model, tokenizer, brief, assignment):
    inputs = tokenizer(brief + assignment, truncation=True, padding='longest', max_length=512, return_tensors='pt')
    outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    grade = torch.argmax(probabilities).item()
    return grade


def main():
    # Define the model configuration
    config = BertConfig.from_pretrained('bert-base-uncased', num_labels=4)

    # Define the model architecture
    model = BertForSequenceClassification(config)

    # Load the weights
    model.load_state_dict(torch.load('model.pt'))

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    with open('guidance.txt', 'r', encoding='utf-8') as f:
        brief = f.read()

    with open('test_assignment.txt', 'r', encoding='utf-8') as f:
        assignment = f.read()

    grade = test(model, tokenizer, brief, assignment)
    print(['U', 'Pass', 'Merit', 'Distinction'][grade])


if __name__ == '__main__':
    main()
