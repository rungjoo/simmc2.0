# Subtask2
This approach is the same as [sub2_12](https://github.com/rungjoo/simmc2.0/tree/master/sub2_12), but without mention_inform in the training step. Therefore, the model learns more about mismatched objects because it learns matching with all object information.

Therefore, the method of the test is changed as follows.

1. objects corresponding to system utterances for which system matching is predicted to be 1: true
2. Objects corresponding to system utterances for which system matching is predicted to be 0: false
3. Objects that do not appear in previous system utterances: predicted by matching score

## Download the trained model 
Download model from the [model folder](https://github.com/rungjoo/simmc2.0/tree/master/sub2_4/model).

## Training
```bash
bash train.sh
```
**train.py** is for **dstc10-simmc-entry** and **train_final.py** is for **dstc10-simmc-final-entry**. **dstc10-simmc-entry** is the result for devtest data and **dstc10-simmc-final-entry** is the result for teststd data. The model (model.pt / model_final.pt) is saved in the model folder, and the log (train.log / train_final.log) is saved in the ./results/{model_type}.

## Testing
```bash
bash test.sh
```
dstc10-simmc-teststd-pred-subtask-3_4.txt is saved in the ./results/{test_type}.