# NL QL data augmentation

This repo served all codes related to data augmentation approach for NL-to-SQL tasks.

## Components introduction

### Data folder `/data`

Each subfolder under `data` holds the data for different datasets respectively. After entering the subfolder, the subsubfolders contain the data that stands for,

- `data_aug` for data augmentation files for scripts in `src/data_augmentation`
- `generative` for data augmentation files for scripts in `src/syntehtic_data`
- `handmade_training_data` for generated data to train / evaluate ValueNet
- `original` for containing data schemata for the database

### Script folder `/src`

Folder `src` holds all scripts for the data augmentation. The following subfolders serve for different puposes,

- `data_augmentation` for data augmentation pipe with seeding data and training CRITIC model.
  
- `synthetic_data` for data augmentation pipe with data generative schema
  
- `intermediate_representation`, `preprocessing` and `spider` for all AST related helper files

- `tools` for all other helper files

## Usage

### Install dependencies and prerequisites

1. Run `pip install -r requirements.txt` to install all required dependencies

2. Sign up in [openai.com](https://openai.com/api/) and configure an API key in the page [https://beta.openai.com/account/api-keys](https://beta.openai.com/account/api-keys)

3. Add the api key as the line shown below in `.env` file under the root folder of the project. If you haven't this file, please create it.

```shell
OPENAI_API_KEY=<api_key_from_openai>
```

### Data augmentation based on shuffling on AST


#### with analytics-based re-ranking via sentence transformers

1. Prepare generative schema
   - Run te script shown as below

   ```bash
   python3 src/tools/transform_generative_schema.py --data_path <datapath>
   ```

   , where `<datapath>` is the path in `data` which contains the subfolder `data/<datapath>/generative` and original schema `data/<datapath>/original/tables.json`

2. Add more query types into `src/synthetic_data/common_query_types` if neccessary
   -- find out query types with

   ```bash
   python3 src/synthetic_data/group_paris_to_find_templates.py \
   --data_containing_semql <training_or_dev_json_file>
   ```

   -- Add query types found into file `src/synthetic_data/common_query_types.py`

3. Generate data
   - Write your own generating file as similar as `src/synthetic_data/generate_synthetical_data_cordis` (TODO: generalize this step for all dataset)
4. re-ranking data and generate handmade training data

   ```bash
   python3 src/synthetic_data/apply_sentence_embedding_reranking.py \
   --input_data_path <input_data_path> \
   --output_file <output_path> \
   --db_id <db_id>
   ``` 

## Hyperparameters and hardware specifications in evaluation experiments

### Evaluation on ValueNet

#### Hyperparameters

|Hyperparameter|Selected Values|
|--|:---:|
|Pretrained model of encoder|bart-base
|Seed | 90 |
|Dropout | 0.3|
|Optimizer| Adam|
|Learning rate base| 0.001|
|Beam size|1|
|Clip grad |5|
|Accumulated iterations | 4 |
|Batch size| 4 |
|Column attention| affine|
|Number of epochs| 100|
|Hidden size| 300|
|Attention vector size| 300|
|Learning rate connection| 0.0001|
|Max grad norm| 1|
|Column embedding size |300|
|Column poniter| true|
|Learning rate of Transformer| 2e-5|
|Max sequence length| 1024|
|Scheduler gamma| 0.5|
|Type embedding size| 128|
|Action embedding size| 128|
|Sketch loss weight| 1|
|Decode max time step| 80|
|Loss epoch threshold| 70|

#### Hardware specifications

|Hardware specifications|Values|
|:--|:---:|
|CPU count|8|
|GPU count|1|
|GPU type|nVidia V100|
|Total running time| ca. 12 days|

### Evaluation on T5-Large

#### Hyperparameters

|Hyperparameter|Selected Values|
|--|:---:|
|Pretrained model| T5-Large|
|Dropout| 0.1| 
|Learning rate| base 0.0001|
|Optimizer| Adafactor|
|Clip grad| 5|
|Accumulated iterations| 4|
|Batch size| 4|
|Gradient Accum.| 135|
|Max sequence length| 512|
|No of steps| 6500|

Hardware specifications Values

#### Hardware specifications

|Hardware specifications|Values|
|:--|:---:|
|CPU count| 8|
|GPU count| 1|
|GPU type| nVidia A100|
|Total running time| ca. 16 days|

### Evaluation on SmBoP

#### Hyperparameters

|Hyperparameter|Selected Values|
|--|:---:|
|Pretrained model| GraPPa-Large|
|Dropout| 0.1|
|Learning rate base| 0.000186|
|Optimizer| Adam|
|RAT Layers| 8|
|Beam Size| 30|
|Batch size| 16|
|Gradient Accum.| 4|
|Max sequence length| 512|
|Max steps| 60000|
|Early Stopping |5 epochs|

#### Hardware specifications


|Hardware specifications|Values|
|:--|:---:|
|CPU count| 8|
|GPU count| 1|
|GPU type| nVidia T4|
|Total running time |ca. 26 days|

## Contributing

## License
