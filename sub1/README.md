# Subtask1
Our apporach description will be updated.

## Download the trained model 
Download pre-trained model from the [model folder](https://github.com/rungjoo/simmc2.0/tree/master/sub1/model).

## Training
```bash
bash train.sh
```
**train.py** is for **dstc10-simmc-entry** and **train_final.py** is for **dstc10-simmc-final-entry**. dstc10-simmc-entry is the result for devtest data and dstc10-simmc-final-entry is the result for teststd data. The model (model.pt / model_final.pt) is saved in the model folder, and the log (train.log / train_final.log) is saved in the ./results/{model_type}. 

## Testing
```bash
bash test.sh
```
dstc10-simmc-teststd-pred-subtask-1.json is saved in the ./results/{model_type}.