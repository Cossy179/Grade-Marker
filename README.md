# AI Assignment Grader

This project contains two Python scripts for training and testing a model that grades assignments based on provided guidance and a set of pre-graded assignments.

## Files

- `train.py`: This script loads the assignment guidance, assignment texts, and grades, then trains a BERT model for sequence classification to predict the grade based on the guidance and assignment text.

- `test.py`: This script loads a trained model and uses it to predict the grade of a test assignment.

- `guidance.txt`: The guidance provided for the assignments.

- `test_assignment.txt`: A test assignment for which to predict the grade.

- `assignments/`: A directory that contains the assignment texts and their grades.

## Usage

1. Place your assignment guidance in the `guidance.txt` file.

2. Place your pre-graded assignments in the `assignments/` directory. Each assignment should have a `.txt` file for the assignment text and a `_grade.txt` file for the grade (which should be 'U', 'Pass', 'Merit', or 'Distinction').

3. Run the `train.py` script to train the model:

    ```bash
    python train.py
    ```

4. Place the text of the assignment you want to grade in the `test_assignment.txt` file.

5. Run the `test.py` script to predict the grade of the test assignment:

    ```bash
    python test.py
    ```

The grade of the test assignment will be printed to the console.

## Requirements

- Python 3.7 or later
- PyTorch
- Transformers
- Other dependencies can be installed with `pip install -r requirements.txt`.

## Disclaimer

This model is not guaranteed to provide accurate grades and should be used as a tool to assist in grading rather than a replacement for human grading.
This currently only works with assignments with 400ish words, so not ideal for anything above and would give very inaccurate results
