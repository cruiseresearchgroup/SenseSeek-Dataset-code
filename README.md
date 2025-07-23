## SenseSeek Dataset: Multimodal Sensing to Study Information Seeking Behaviors

The repository contains several Jupyter Notebooks with data analysis and visualizations and Python files with model training. All required packages are listed in the requirements.txt file. The training data, i.e., the features, are included in the 'training data' folder, in a csv format. 

### Data Availability:
The use of the SenseSeek dataset is limited to academic research purposes. The SenseSeek dataset is publicly available and can be accessed on the OSF platform at https://osf.io/waunb/?view_only=f50ce00dc96f486e83ff5ccef71d13f3. An additional description of the dataset is included in the paper appendix. 

The link contains two zip files:
  - `SenseSeek-dataset.zip` (2.2 GB) with the data, features, task responses, task materials, and demographic information.
  - `SenseSeek_Screen_Recordings.zip` (4.6 GB) with the gaze-annotated screen recordings.

### Dataset Instruction
The dataset folder is structured as follows:<br />
- `EYETRACKER`<br />
  -- contains 3 sub-folders (features, processed, raw), with each containing the following 2 sub-folders corresponding to the data type:<br />
      --- `GAZE`<br />
      --- `PUPIL`<br />
- `HEADSET`<br />
  -- contains 3 sub-folders (features, processed, raw), with each containing the following 2 sub-folders corresponding to the data type:<br />
      --- `EEG`<br />
      --- `head motion`<br />
- `WRISTBAND`<br />
  -- contains 3 sub-folders (features, processed, raw), with each containing the following 2 sub-folders corresponding to the data type:<br />
      --- `ACC` (for the wrist motion)<br />
      --- `EDA`<br />
- `TIME` (contains csv file with all event timestamps for each participant)<br />
- `attention_checks.csv` (contains the participants' answers to the judgement questions)<br />
- `demographic.csv` (contains the demographic information)<br />
- `self_ratings.csv` (contains the self-rated answers)<br />
- `survey_durations.csv` (contains the durations for all events in the experiments of all participants) <br />
- `task_materials.xlsx` (contains the topic, backstories, search results, and the judgment questions used in the experiment)<br />

***The raw EEG and head motion data are saved in EDF files. The rest of the data is saved in CSV files.***

***The data files are structured as follows:*** <br />
Apart from the data columns, all processed files are annotated with the following event information at each row:<br />
<br/>
For time information:<br />
  - `ts`: timestamps of current data point in ISO 8601 time format with milliseconds (e.g., 2023-08-15 10:14:53.950000+10:00), calculated based on the recorded starting time and the sampling rate.
  - `sec`: the time in seconds related to the experiment starting time.
  - `sec2`: the time in seconds related to each session starting time. There are 2 sessions in each experiment. 
  - `durations`: the total duration of the event.
  - `start`, `end`: the starting and ending timestamps of the event recorded during the experiment.
  - `start_sec`, `end_sec`: the starting and ending times in seconds related to the experiment starting time during the experiment.
<br />

For task information: <br />
  - `task`: the sequence of the task, from 0 to 11. 
  - `name`: the full name of the event.
  - `stage`: the abbreviation of the current event.
  - `Topic`: the topic ID of the current search task.

## Citation

If you use this resource, please use the following citation:

`Kaixin Ji, Danula Hettiachchi, Falk Scholer, Flora D. Salim, and Damiano Spina. 2025. SenseSeek Dataset: Multimodal Sensing to
Study Information Seeking Behaviors . Proc. ACM Interact. Mob. Wearable Ubiquitous Technol. 9, 3, Article 92 (September 2025),
29 pages. https://doi.org/10.1145/3749501`

```bibtex
@article{ji2025senseseek,
  author       = {Kaixin Ji and Danula Hettiachchi and Falk Scholer and Flora D. Salim and Damiano Spina},
  title        = {{SenseSeek Dataset: Multimodal Sensing to Study Information Seeking Behaviors}},
  journal      = {Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies},
  volume       = {9},
  number       = {3},
  article      = {92},
  year         = {2025},
  month        = {September},
  numpages     = {29},
  doi          = {10.1145/3749501},
  publisher    = {ACM}
}
```


