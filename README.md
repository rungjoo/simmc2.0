# The Submission for SIMMC 2.0 Challenge 2021
- [challenge website](https://github.com/facebookresearch/simmc2)

## Requirements
- python 3.8.8
- pytorch 1.8.1
- [transformers 4.8.2](https://huggingface.co/transformers/v4.8.1/)
- [apex](https://github.com/NVIDIA/apex) for multi-gpu
- nltk

## Preprocessing

1. Download Data

- Download the [data](https://github.com/facebookresearch/simmc2/tree/master/data) provided by the challenge organizer and put it in the data folder.
- Unzip data files

2. Image saving

- Preprocess the image files in advance. The preprocessed result has the image name as the key and visual as the value.
```bash
python3 image_preprocessor.py
python3 image_preprocessor_final.py
```
- The result(.pickle) is saved in [res folder](https://github.com/rungjoo/simmc2.0/tree/master/res).

## Step 1 (ITM)

First, the model is post-trained by image-to-text matching. Here, image is each object and text is the visual metadata of the object.
Code is provided in the [ITM folder](https://github.com/rungjoo/simmc2.0/tree/master/ITM).

## Step 2 (BTM)
Second, pretraining is performed to use background reprsentation of image in subtasks. Similar to [ITM](https://github.com/rungjoo/simmc2.0/tree/master/ITM), it is trained to match image and text, and the image is the background of the dialog and the text is the entire context of the dialog. Code is provided in the [BTM folder](https://github.com/rungjoo/simmc2.0/tree/master/BTM).

## Step 3

This is the learning process for each subtask. You can train the model in each folder ([sub1](https://github.com/rungjoo/simmc2.0/tree/master/sub1), [sub2_1](https://github.com/rungjoo/simmc2.0/tree/master/sub2_1), [sub2_2](https://github.com/rungjoo/simmc2.0/tree/master/sub2_2), [sub2_3](https://github.com/rungjoo/simmc2.0/tree/master/sub2_3), [sub2_4](https://github.com/rungjoo/simmc2.0/tree/master/sub2_4), [sub4](https://github.com/rungjoo/simmc2.0/tree/master/sub4)).

## Model

All models can be downloaded from the following [link](https://drive.google.com/drive/folders/10dhrgH7HNenwHSZjvQYY-sk5VpAuoA7G?usp=sharing)

**model.pt** is a model for evaluating devtest, and the result is saved in the **dstc10-simmc-entry** folder. **model_final.pt** is a model for evaluating teststd, and the result is saved in the **dstc10-simmc-final-entry** folder. *However, the training of the model was not completed within the challenge period, so we inferred to model.pt for the teststd data in subtask2.*

## Evlauation

Using the evaluation script suggested by the challenge organizer

* [Evaluation script for subtask1](https://github.com/rungjoo/simmc2.0/blob/master/evaluation/model/utils/test_sub1.sh)
* [Evaluation script for subtask2](https://github.com/rungjoo/simmc2.0/blob/master/evaluation/model/mm_dst/test_sub2.sh)
* [Evaluation script for subtask4-generation](https://github.com/rungjoo/simmc2.0/blob/master/evaluation/model/utils/test_sub4.sh)

[The SIMMC organizers introduce the scripts](https://github.com/facebookresearch/simmc2/blob/master/SUBMISSION_INSTRUCTIONS.md):
```
<Subtask 1>
$ python tools/disambiguator_evaluation.py \
	--pred_file="{PATH_TO_PRED_FILE}" \
	--test_file="{PATH_TO_TEST_FILE}" \

<Subtask 2 & 3>
(line-by-line evaluation)
$ python -m gpt2_dst.scripts.evaluate \
  --input_path_target={PATH_TO_GROUNDTRUTH_TARGET} \
  --input_path_predicted={PATH_TO_MODEL_PREDICTIONS} \
  --output_path_report={PATH_TO_REPORT}

(Or, dialog level evaluation)
$ python -m utils.evaluate_dst \
    --input_path_target={PATH_TO_GROUNDTRUTH_TARGET} \
    --input_path_predicted={PATH_TO_MODEL_PREDICTIONS} \
    --output_path_report={PATH_TO_REPORT}
    
<Subtask 4 Generation>
$ python tools/response_evaluation.py \
    --data_json_path={PATH_TO_GOLD_RESPONSES} \
    --model_response_path={PATH_TO_MODEL_RESPONSES} \
    --single_round_evaluation

<Subtask 4 Retrieval>
$ python tools/retrieval_evaluation.py \
    --retrieval_json_path={PATH_TO_GROUNDTRUTH_RETRIEVAL} \
    --model_score_path={PATH_TO_MODEL_CANDIDATE_SCORES} \
    --single_round_evaluation    
```

## DevTest Results

**Subtask #1: Multimodal Disambiguation**
| Test Method | Accuracy |
| :------: | :-------: |
| GPT2 from [CO(Challenge Organizer)](https://github.com/facebookresearch/simmc2/tree/master/model/disambiguate#performance-on-simmc-20) | 73.9 |
| **Ours** | **92.28** |

**Subtask #2: Multimodal Coreference Resolution**
| Test Method | Object F1 |
| :------: | :-------: |
| GPT2 from [CO](https://github.com/facebookresearch/simmc2/tree/master/model/mm_dst#results) | 0.366 |
| **Ours-1** ([sub2_1](https://github.com/rungjoo/simmc2.0/tree/master/sub2_1)) | **0.595** |
| **Ours-2** ([sub2_2](https://github.com/rungjoo/simmc2.0/tree/master/sub2_2)) | **0.604** |
| **Ours-3** ([sub2_3](https://github.com/rungjoo/simmc2.0/tree/master/sub2_3)) | **0.607** |
| **Ours-4** ([sub2_4](https://github.com/rungjoo/simmc2.0/tree/master/sub2_4)) | **0.608** |

**Subtask #3: Multimodal Dialog State Tracking**

No Training/Testing

**Subtask #4: Multimodal Dialog Response Generation**

**Generation** 
| Baseline |      BLEU |
| :------: | :-------: |
| GPT2 from [CO](https://github.com/facebookresearch/simmc2/tree/master/model/mm_dst#results) | 0.192 |
| MTN-SIMMC2 from [CO](https://github.com/facebookresearch/simmc2/tree/master/model/mm_dst#results) | 0.217 |
| **Ours** | **0.285** |

**Retrieval**

No Training/Testing