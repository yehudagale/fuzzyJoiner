# fuzzyJoiner
# fuzzyJoiner
To run the pre-compiled models run:
 python ./preloaded_runner.py --input $DATA_TO_TEST --entity_type $ENTITY_TYPE --loss_function $LOSS_FUCNTION --model $MODEL_FILE
where 
$DATA_TO_TEST is the uncleased unaugmented data to test
$ENTITY_TYPE is people or companies
$LOSS_FUCNTION is the loss function (triplet-loss, improved-loss, angular-loss, or adapted-loss)
$MODEL_FILE is the weights file dowloaded seperately

you can use random_test_selecter.py to choose a random subset of the data to test
simply type random_test_selecter.py $INPUT_FILE $OUTPUT_FILE $NUMBER_TO_SELECT
where 
$INPUT_FILE is the original file
$OUTPUT_FILE is the file you want to write to
$NUMBER_TO_SELECT is the number of items to select (before augmentation and cleasing)
