# Subtask2
This approach is the same as [sub2_1](https://github.com/rungjoo/simmc2.0/tree/master/sub2_1), but without mention_inform in the training step. Therefore, the model learns more about mismatched objects because it learns matching with all object information.

Therefore, the method of the test is changed as follows.

- All Objects corresponding to system utterances for which system matching is predicted to be 1: true (=method 3) or matching score (=method 4)
- All Objects corresponding to system utterances for which system matching is predicted to be 0: false
- Other objects that do not appear in previous system utterances: predicted by matching score

## Download the trained model 
Download model from the [model folder](https://github.com/rungjoo/simmc2.0/tree/master/sub2_34/model).

## Training
```bash
bash train.sh
```
**train.py** is for **dstc10-simmc-entry** and **train_final.py** is for **dstc10-simmc-final-entry**. **dstc10-simmc-entry** is the result for devtest data and **dstc10-simmc-final-entry** is the result for teststd data. The model (model.pt / model_final.pt) is saved in the model folder, and the log (train.log / train_final.log) is saved in the ./results/{model_type}.

## Testing
```bash
bash test.sh
```
There are two test methods here. You can give *method 3 or 4* as a command argument. (refer to test.sh) dstc10-simmc-{devtest/teststd}-pred-subtask-3_{method}.txt is saved in the ./results/{test_type}.