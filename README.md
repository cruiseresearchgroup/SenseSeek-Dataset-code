## SenseSeek Dataset: Multimodal Sensing to Study Information Seeking Behaviors

The repository contains several Jupyter Notebooks with data analysis and visualizations and Python files with model training. All required packages are listed in the requirements.txt file. The training data, i.e., the features, are included in the 'training data' folder, in a csv format. 

### Data Availability:
The use of the SenseSeek dataset is limited to academic research purposes. The SenseSeek dataset is publicly available and can be accessed on the OSF platform at https://osf.io/waunb/?view_only=f50ce00dc96f486e83ff5ccef71d13f3.Additional descriptions of the dataset are included in the paper appendix. 
The link contains two zip files:
  - SenseSeek-dataset.zip (2.2 GB) with the data, features, task responses, task materials, and demographic information.
  - SenseSeek_Screen_Recordings.zip (4.6 GB) with the gaze-annotated screen recordings.

### Dataset Instruction
The dataset folder is structured as follows:
|EYETRACKER
  |contains 3 sub-folders (features, processed, raw), with each containing the following 2 sub-folders corresponding to the data type:
      |GAZE
      |PUPIL
|HEADSET
  |contains 3 sub-folders (features, processed, raw), with each containing the following 2 sub-folders corresponding to the data type:
      |EEG
      |head motion
|WRISTBAND
  |contains 3 sub-folders (features, processed, raw), with each containing the following 2 sub-folders corresponding to the data type:
      |ACC (for the wrist motion)
      |EDA
|TIME (contains csv file with all event timestamps for each participant)
| attention_checks.csv (contains the participants' answers to the judgement questions)
| demographic.csv (contains the demographic information)
| self_ratings.csv (contains the self-rated answers)
| survey_durations.csv (contains the durations for all events in the experiments of all participants) 
| task_materials.xlsx (contains the topic, backstories, search results, and the judgment questions used in the experiment)

* The raw EEG and head motion data are saved in EDF files. The rest of the data is saved in CSV files.

The data files are structured as follows:
Apart from the data columns, all processed files are annotated with the following event information at each row:

For time information:
- ts: timestamps of current data point in ISO 8601 time format with milliseconds (e.g., 2023-08-15 10:14:53.950000+10:00), calculated based on the recorded starting time and the sampling rate.
- sec: the time in seconds related to the experiment starting time.
- sec2: the time in seconds related to each session starting time. There are 2 sessions in each experiment. 
- durations: the total duration of the event.
- start, end: the starting and ending timestamps of the event recorded during the experiment.
- start_sec, end_sec: the starting and ending times in seconds related to the experiment starting time during the experiment.

For task information:
- task: the sequence of the task, from 0 -- 11. 
- name: the full name of the event.
- stage: the abbreviation of the current event.
- Topic: the topic ID of the current search task.


