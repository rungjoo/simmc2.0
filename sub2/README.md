# Subtask2
Our apporach description will be updated.

## Download the pre-trained model 
Download model from the [model folder](https://github.com/rungjoo/simmc2.0/tree/master/sub2/model).

## Training
```bash
bash train.sh
```
**train.py** is for **dstc10-simmc-entry** and **train_final.py** is for **dstc10-simmc-final-entry**. The model (model.pt / model_final.pt) is saved in the model folder, and the log (train.log / train_final.log) is saved in the ./results/{model_type}.

## Testing
```bash
bash test.sh
```
dstc10-simmc-teststd-pred-subtask-3.txt is saved in the ./results/{model_type}.