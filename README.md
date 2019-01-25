# Name matching across surface forms of entity names

This repository has the code used to build machine learning models to perform joins across variations in people's and company names; e.g. to match across names such as <it>Douglas Adams</it> and <it>Adams, Douglas</it>.

For details about how these models were built, or how they may be used for fuzzy joins, [see here](https://arxiv.org/abs/1809.01604).  All the models built using the techniques described in this paper are available at https://drive.google.com/drive/folders/1zivCTGkq2_AkfjGLHMnlehzTmYUwcQ9e.

# Installation
First make sure to install requirements.txt using pip.  Use Python3.

`pip install -r requirements.txt`

# Testing with existing models:

Here is how to test a model with precomputed weights from https://drive.google.com/drive/folders/1zivCTGkq2_AkfjGLHMnlehzTmYUwcQ9e.  The models directory is organized with a separate directory for the people data and a separate directory for the company data.  In each directory we have the results organized by loss function.  Adapted loss contains the test/train/validation splits, and it contains the model and the corresponding tokenizer.  

To replicate the results reported in the paper for the adpapted loss function for people data use:

`python3 preloaded_runner.py --input testdata --entity_type people --loss_function adapted-loss --model model --tokenizer tokenizer --previous_test yes`


Here's example output from that run:

mean closest positive count (precision-all):0.6699706305093758

mean positive distance:0.7825878046258231

stdev positive distance:0.6112825495769008

max positive distance:25.267065048217773

mean neg distance:2.9535758542870725

stdev neg distance:1.380804040010406

max neg distance:28.699949264526367

mean all positive distance:1.5341067807035704

stdev all positive distance:2.2664244913768132

max all positive distance:84.53878784179688

mean all neg distance:2.9164267418596084

stdev all neg distance:1.3876352138587738

max all neg distance:28.699949264526367

Accuracy in the ANN for triplets that obey the distance func:0.9359872199075374

Precision at 1: 0.850601929164278

Test stats (Recall):0.8387458185950479

These results are replicable but may vary slightly across machines.

# Training your own model
If you have your own data you would like to use to train your own model (your own sets of people or company data), ensure you have entities organized as in names_to_cleanse/peoplesNames.txt.  You can then create a new model with the following command as an example (here we are building a people model with the adapted loss function):

`python3 build_model.py --input names_to_cleanse/peoplesNames.txt --loss_function adapted-loss --use_l2_norm False --num_layers 3 --entity_type people --model /tmp/model`

Note that if you have a different set of entities you have to change the code in NamesCleanser and add some cleansing code if you need to.  Also you will need to add support for that entity in `build_model.py`.

For a description of various parameter settings (e.g. `--use_l2_norm`) refer to the paper.  

# How does this relate to entity resolution?
This approach is not directly the same as entity resolution or entity linking.  In entity resolution *multiple* attributes of an entity are considered in linking an entity.  For instance, a person's social security number, their address, their home phone number, their name etc are different attributes that are considered in linking the entity with another.  In most entity linking systems, *similarity* functions are used to determine similarity of the attribute in question.  Typically, generic similarity functions are included such as string similarity functions (Levenshtein distance, Jaccard similarity etc).  The point of this work is that when there is sufficient data for a specific entity type (e.g. people's names, company names etc), one can use deep neural net models like the ones built here as similarity functions.  How effective is this compared to existing string similarity functions?  The trouble with existing string similarity functions is that one needs to compare every name with every other name or implement *blocking* (i.e., only compare names that might be comparable in some form).  We implemented a straightforward type of blocking and examined if we considered the top 20 neighbors (20 closest in terms of Levenshtein distance) what the precision@1, precision-all and recall would be.  To reproduce these results, run `python Levenstien_Rule_Based.py names_to_cleanse/peoplesNames.txt people`.

Here are the results:

mean closest positive count (Precision-all):0.1360894849189292 

Precision at 1: 0.4194811175928769 

(Recall): 0.3109830209635783
