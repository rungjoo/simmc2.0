# Subtask2
This is an approach that learned only system matching, which is one of the multi-task learnings in [sub2](https://github.com/rungjoo/simmc2.0/tree/master/sub2). That is, it is a model that matches previous system utterances with the same object as the current utterance without using visual information. In the test, all objects corresponding to the utterance predicted by the system utterance with the same object are predicted to be true.

## Download the trained model 
Download model from the [model folder](https://github.com/rungjoo/simmc2.0/tree/master/sub2_2/model).

## Training
```bash
bash train.sh
```
**train.py** is for **dstc10-simmc-entry** and **train_final.py** is for **dstc10-simmc-final-entry**. The model (model.pt / model_final.pt) is saved in the model folder, and the log (train.log / train_final.log) is saved in the ./results/{model_type}.

## Testing
```bash
bash test.sh
```
dstc10-simmc-teststd-pred-subtask-3_3.txt is saved in the ./results/{model_type}.