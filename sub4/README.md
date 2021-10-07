# Subtask4
Our model uses as input the previous utterances and the slot_values of the current turn. That is, concatenated $$(utt_{1},utt_{2}, ..., utt_{k}, flattening of slots)$$ is the text input of the model. The logits $$t_{h}$$ of the last token are computed with GPT2-large, the backbone of the text input. To use multimodal, the visual representations of objects are extracted with an ITM model and averaged to $$o_{h}$$. Finally, these two logits are added to predict the next token.
$$a^2 + b^2 = c^2$$
## Download the trained model 
Download model from the [model folder](https://github.com/rungjoo/dstc10/tree/master/sub4/model).

## Training
```bash
bash train.sh
```
**train.py** is for **dstc10-simmc-entry** and **train_final.py** is for **dstc10-simmc-final-entry**. dstc10-simmc-entry is the result for devtest data and dstc10-simmc-final-entry is the result for teststd data. The model (model.pt / model_final.pt) is saved in the model folder, and the log (train.log / train_final.log) is saved in the ./results/{model_type}.

## Testing
```bash
bash test.sh
```
dstc10-simmc-teststd-pred-subtask-4-generation.json saved in the ./results/{model_type}.