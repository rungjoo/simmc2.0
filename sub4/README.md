# Subtask4
Our apporach description will be updated.

## Download the pre-trained model 
Download model from the [model folder](https://github.com/rungjoo/dstc10/tree/master/sub4/model).

## Training

### for dstc10-simmc-entry
```bash
bash train.sh
```

### for dstc10-simmc-entry
```bash
bash train_final.sh
```
The model (model.pt / model_final.pt) is saved in the model folder, and the log (train.log / train_final.log) is saved in the results.

## Testing
```bash
bash test.sh
```
dstc10-simmc-teststd-pred-subtask-4-generation and the log (test.log) is saved in the ./results/{model_type}.