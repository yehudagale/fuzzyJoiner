# fuzzyJoiner
# fuzzyJoiner

This repository has the code used to build machine learning models to preform fuzzy joins. Here is how to run a preloaded model from https://drive.google.com/drive/folders/1zivCTGkq2_AkfjGLHMnlehzTmYUwcQ9e on your own data:

first make sure to install requirements.txt using pip.

To run the pre-compiled models first download two files (model weights file and serialized tokenizer)
then run:

 python ./pre-loaded_runner.py --input $DATA_TO_TEST --entity_type $ENTITY_TYPE --loss_function $LOSS_FUCNTION --model $MODEL_FILE --tokenizer $TOKENIZER_FILE
where:

$DATA_TO_TEST is the uncleansed unaugmented data to test, a sample input for people is included in data_for_testing.txt
$ENTITY_TYPE is people or companies
$LOSS_FUCNTION is the loss function (triplet-loss, improved-loss, angular-loss, or adapted-loss)
$MODEL_FILE is the weights file downloaded separately
$TOKENIZER_FILE is the serialized tokenizer downloaded separately
if you have test data in a pickle file ending in '.test_data.pickle' but otherwise the same as the model file, you can use the '--previous_test' flag to load that data instead of the input file.

here is an example:
python ./preloaded_runner.py --loss_function adapted-loss --input ./data_for_testing.txt --entity_type people --model adapted.people.model --tokenizer adapted_people_model_1.tokenizer.pickle 

You can use random_test_selecter.py to choose a random subset of the data to test
simply type random_test_selecter.py $INPUT_FILE $OUTPUT_FILE $NUMBER_TO_SELECT
where 
$INPUT_FILE is the original file
$OUTPUT_FILE is the file you want to write to
$NUMBER_TO_SELECT is the number of items to select (before augmentation and cleansing)
