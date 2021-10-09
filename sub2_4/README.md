# Subtask2
This approach is to test by combining the two trained models ([sub2_1](https://github.com/rungjoo/simmc2.0/tree/master/sub2_1), [sub2_3](https://github.com/rungjoo/simmc2.0/tree/master/sub2_3)).

1. The objects corresponding to the utterances for which system_matching is predicted to be 1 are predicted by the sub2_1 model.
2. The objects corresponding to the utterances for which system_matching is predicted to be 0 are False
3. All other objects are predicted by the sub2_3 model.

The reason for doing this is that the sub2_1 model is good at predicting previously appeared objects because it is a method learned using mention_inform, and the sub2_3 model has the ability to fit unseen objects well because it is a method learned without using mention_inform.

## Testing
```bash
bash test.sh
```
dstc10-simmc-{devtest/teststd}-pred-subtask-3_4.txt is saved in the ./results/{test_type}.