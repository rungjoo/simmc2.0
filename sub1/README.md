# Subtask1: Multimodal Disambiguation

## Approach
First, we use all previous utterances to connect as input to our model. The text input is passed through Roberta-large to get a representation of the text. To use multimodal information, the visual representation of an object is extracted. The visual model is a pre-trained DeIT at ITM. Disambiguation is predicted by projecting representations of text and objects to the same dimension.

## Download the trained model 
Download pre-trained model from the [model folder](https://github.com/rungjoo/simmc2.0/tree/master/sub1/model).

## Training
```bash
bash train.sh
```
**train.py** is for **dstc10-simmc-entry** and **train_final.py** is for **dstc10-simmc-final-entry**. **dstc10-simmc-entry** is the result for devtest data and **dstc10-simmc-final-entry** is the result for teststd data. The model (model.pt / model_final.pt) is saved in the model folder, and the log (train.log / train_final.log) is saved in the ./results/{model_type}. 

## Testing
```bash
bash test.sh
```
dstc10-simmc-teststd-pred-subtask-1.json is saved in the ./results/{test_type}.