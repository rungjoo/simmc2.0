# Background Image vs Context Matching

BTM (background-text-matching) is a process of pre-learning the representation of the background image for use in subtasks. Here, we learn to match the background image and context of the dialog session. Negative data is randomly sampled to form the same number as positive data. This is a method similar to ITM, but ITM is a concept of transferring the learned model to subtasks as it is, and BTM is used to extract the background image representation by importing only the learned image model.

## Download the pre-trained model 
Download model.pt from the [post_training](https://github.com/rungjoo/dstc10/tree/master/BTM/post_training).

## Training
```bash
bash train.sh
```
The model and log (train.log) is saved in the bg_model.