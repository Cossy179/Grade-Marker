import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
import os
import logging

class AssignmentDataset(Dataset):
    def __init__(self, assignments, tokenizer, max_length=512):
        self.assignments = assignments
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.assignments)

    def __getitem__(self, idx):
        assignment = self.assignments[idx]
        encodings = self.tokenizer(assignment['brief'] + ' ' + assignment['assignment'],
                                   truncation=True,
                                   max_length=self.max_length,
                                   padding='max_length',
                                   return_tensors='pt')
        encodings['labels'] = torch.tensor([assignment['grade']])
        return {key: val.squeeze() for key, val in encodings.items()}

def train(model, data_loader, optimizer):
    model.train()
    total_loss = 0
    for idx, batch in enumerate(data_loader):
        optimizer.zero_grad()
        output = model(**batch)
        loss = output.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (idx + 1) % 10 == 0:
            logger.info(f'Batch: {idx + 1}, Loss: {total_loss / (idx + 1)}')


def load_data(guidance_file, assignment_dir):
    with open(guidance_file, 'r', encoding='utf-8') as f:
        guidance = f.read()

    assignments = []
    assignment_files = [f for f in os.listdir(assignment_dir) if not f.endswith('_grade.txt')]
    for assignment_file in assignment_files:
        with open(os.path.join(assignment_dir, assignment_file), 'r', encoding='utf-8') as f:
            assignment = f.read()
            
        grade_file = assignment_file.replace('.txt', '_grade.txt')
        with open(os.path.join(assignment_dir, grade_file), 'r', encoding='utf-8') as f:
            grade_text = f.read().strip()
            if grade_text == 'U':
                grade = 0
            elif grade_text == 'Pass':
                grade = 1
            elif grade_text == 'Merit':
                grade = 2
            elif grade_text == 'Distinction':
                grade = 3
            else:
                raise ValueError(f'Invalid grade: {grade_text}')

        assignments.append({
            "brief": guidance,
            "assignment": assignment,
            "grade": grade
        })
    return assignments

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Loading data...")
    assignments = load_data('guidance.txt', 'assignments')
    logger.info(f"Loaded {len(assignments)} assignments.")

    logger.info("Loading tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    logger.info("Creating dataset and dataloader...")
    dataset = AssignmentDataset(assignments, tokenizer)
    data_loader = DataLoader(dataset, batch_size=8, shuffle=True)

    logger.info("Loading model...")
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)
    
    logger.info("Setting up optimizer...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    logger.info("Starting training...")
    for epoch in range(10):
        logger.info(f'Starting epoch {epoch + 1}')
        train(model, data_loader, optimizer)
    logger.info("Training complete.")
    
    logger.info("Saving model...")
    torch.save(model.state_dict(), f'model.pt')
    logger.info("Model saved.")

if __name__ == '__main__':
    main()

