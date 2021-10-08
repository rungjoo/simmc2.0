# Subtask2
Our approach is to match the representation of an object with a text. Text input is previous utterances, and reprsentation of text is extracted through Roberta-large. The visual representation of the object is extracted with DeIT. Roberta-large and DeIT use parameters learned from ITM as initial states. We also uses the background image of the scene that contains the object. The visual representation of the background is extracted using DeIT learned from BTM. The final visual representation is computed by adding the visual representation of the object and the background. The matching score (0~1) is calculated by calculating the cosine similarity between the text and visual reprsentation, scaling it up (100) and passing it through the sigmoid. When training, mention_inform is used, and object matching is trained not only in user's utterances but also in system's utterances.


Additionally, we apply the following 3 multi-task learning.

1. utterance classification
    - Whether there is an object to match the current utterance
2. system matching
    - In each previous system utterance, whether there is an object to share with the utterance of the current turn.
3. meta-visual matching
    - Matching task between visual meta information of objects
    
When we test, we use the following method:

- The model judges the objects corresponding to the system utterance predicted to share the same objects as the current turn as a matching score.
- All other objects are considered False.

## Download the trained model 
Download model from the [model folder](https://github.com/rungjoo/simmc2.0/tree/master/sub2_1/model).

## Training
```bash
bash train.sh
```
**train.py** is for **dstc10-simmc-entry** and **train_final.py** is for **dstc10-simmc-final-entry**. **dstc10-simmc-entry** is the result for devtest data and **dstc10-simmc-final-entry** is the result for teststd data. The model (model.pt / model_final.pt) is saved in the model folder, and the log (train.log / train_final.log) is saved in the ./results/{model_type}.

## Testing
```bash
bash test.sh
```
There are two test methods here. You can give *method 1 or 2* as a command argument. (refer to test.sh) dstc10-simmc-{devtest/teststd}-pred-subtask-3_{method}.txt is saved in the ./results/{test_type}.