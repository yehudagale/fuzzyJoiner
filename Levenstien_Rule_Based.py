from sys import argv
import string
import Levenshtein
import statistics 

from names_cleanser import NameDataCleanser, CompanyDataCleanser


def read_entities(filepath):
    entities = []
    with open(filepath, 'r', encoding='utf8') as fl:
        for line in fl:
            entities.append(line)

    return entities
def generate_names(entities, people, limit_pairs=False):
    if people:
        num_names = 4
        generator = NameDataCleanser(0, num_names, limit_pairs=limit_pairs)
    else:
        generator = CompanyDataCleanser(limit_pairs)
        num_names = 2

    entity2same = {}
    for entity in entities:
        ret = generator.cleanse_data(entity)
        if ret and len(ret) >= num_names:
            entity2same[ret[0]] = ret[1:]
    return entity2same
def load_buckets(entity2same):
	#used https://stackoverflow.com/questions/34860982/replace-the-punctuation-with-whitespace
	translator = str.maketrans(string.punctuation, ' '*len(string.punctuation))
	bucket_dict = {}
	for item in entity2same:
		for entity in entity2same[item] + [item]:
			entity_no_punct = entity.translate(translator).lower()
			split_entity = [item for item in entity_no_punct.split(' ') if item]
			for word in split_entity:
				if word not in bucket_dict:
					bucket_dict[word] = []
				bucket_dict[word].append(entity)
	return bucket_dict
def get_stats(entity2same, bucket_dict):
    def get_closest(items, key, number_to_get):
        low_key = key.lower()
        closest = [(x, Levenshtein.distance(low_key, x.lower())) for x in items]
        # print('key')
        # # print(key.encode('utf-8'))
        # print('unsorted')
        # # print([(thing[0].encode('utf-8'), thing[1]) for thing in closest])
        # print('sorted')
        closest = sorted(closest, key=lambda a: a[1])
        # print([(thing[0].encode('utf-8'), thing[1]) for thing in closest])
        # print('dist removed')
        closest = [item[0] for item in closest]
        # print([thing.encode('utf-8') for thing in closest])
        # print([thing.encode('utf-8') for thing in closest[:number_to_get]])
        return closest[:number_to_get]

    # predictions = model.predict(sequences)
    # t = AnnoyIndex(len(predictions[0]), metric='euclidean')  # Length of item vector that will be indexed
    # t.set_seed(123)
    # for i in range(len(predictions)):
    #     # print(predictions[i])
    #     v = predictions[i]
    #     t.add_item(i, v)

    # t.build(100) # 100 trees

    match = 0
    no_match = 0
    lev_accuracy = 0
    total = 0
    precise = 0
    
    triplets = {}
    closest_positive_counts = []
    
    pos_distances = []
    neg_distances = []
    all_pos_distances = []
    all_neg_distances = []

    triplets['anchor'] = []
    triplets['positive'] = []
    triplets['negative'] = []

    NNlen = 20
    translator = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    print_num = 0
    for key in entity2same:
        nearest = []
        for word in key.translate(translator).lower().split():
            if not word:
                continue
            nearest = nearest + bucket_dict[word]
        if len(nearest) > NNlen:
            nearest = get_closest(nearest, key, NNlen)
        nearest_text = set(nearest)
        expected_text = set(entity2same[key])
        print_num += 1
        if print_num % 100 == 0:
            print(key.encode('utf-8'))
            print([item.encode('utf-8') for item in nearest])

        # annoy has this annoying habit of returning the queried item back as a nearest neighbor.  Remove it.
        if key in nearest_text:
            nearest_text.remove(key)
        # print("query={} names = {} true_match = {}".format(unique_text[index], nearest_text, expected_text))
        overlap = expected_text.intersection(nearest_text)
        # collect up some statistics on how well we did on the match
        m = len(overlap)
        match += m
        # since we asked for only x nearest neighbors, and we get at most x-1 neighbors that are not the same as key (!)
        # make sure we adjust our estimate of no match appropriately
        no_match += min(len(expected_text), len(nearest_text)) - m

        # sample only the negatives that are true negatives
        # that is, they are not in the expected set - sampling only 'semi-hard negatives is not defined here'
        # positives = expected_text - nearest_text
        positives = overlap
        negatives = nearest_text - expected_text

        # print(key + str(expected_text) + str(nearest_text))
        for i in negatives:
            for j in positives:
                dist_pos = Levenshtein.distance(key.lower(), j.lower())
                pos_distances.append(dist_pos)
                dist_neg = Levenshtein.distance(key.lower(), i.lower())
                neg_distances.append(dist_neg)
                if dist_pos < dist_neg:
                    lev_accuracy += 1
                total += 1
                # print(key + "|" +  j + "|" + i)
                # print(dist_pos)
                # print(dist_neg)               

        min_neg_distance = 1000000        
        for i in negatives:
            dist_neg = Levenshtein.distance(key.lower(), i.lower())
            all_neg_distances.append(dist_neg)
            if dist_neg < min_neg_distance:
                    min_neg_distance = dist_neg

        for j in expected_text:
            dist_pos =  Levenshtein.distance(key.lower(), j.lower())
            all_pos_distances.append(dist_pos)

        closest_pos_count = 0
        for p in overlap:
            dist_pos =  Levenshtein.distance(key.lower(), p.lower())
            if dist_pos < min_neg_distance:
                closest_pos_count+=1

        if closest_pos_count > 0:
            precise+=1

        closest_positive_counts.append(closest_pos_count / min(len(expected_text),len(nearest_text)))


            
        # for i in negatives:
        #     for j in expected_text:
        #         triplets['anchor'].append(key)
        #         triplets['positive'].append(j)
        #         triplets['negative'].append(i)

    print("mean closest positive count:" + str(statistics.mean(closest_positive_counts)))
    print("mean positive distance:" + str(statistics.mean(pos_distances)))
    print("stdev positive distance:" + str(statistics.stdev(pos_distances)))
    print("max positive distance:" + str(max(pos_distances)))
    print("mean neg distance:" + str(statistics.mean(neg_distances)))
    print("stdev neg distance:" + str(statistics.stdev(neg_distances)))
    print("max neg distance:" + str(max(neg_distances)))
    print("mean all positive distance:" + str(statistics.mean(all_pos_distances)))
    print("stdev all positive distance:" + str(statistics.stdev(all_pos_distances)))
    print("max all positive distance:" + str(max(all_pos_distances)))
    print("mean all neg distance:" + str(statistics.mean(all_neg_distances)))
    print("stdev all neg distance:" + str(statistics.stdev(all_neg_distances)))
    print("max all neg distance:" + str(max(all_neg_distances)))
    print("Accuracy in the ANN for triplets that obey the distance func:" + str(lev_accuracy / total))
    print("Precision at 1: " +  str(precise / len(entity2same)))
    
    # obj = {}
    # obj['accuracy'] = lev_accuracy / total
    # obj['steps'] = 1
    # with open(output_file_name_for_hpo, 'w', encoding='utf8') as out:
    #     json.dump(obj, out)

    # if test:
    return match/(match + no_match)
    # else:
    #     return triplets, match/(match + no_match)

input_file = argv[1]
people = True if argv[2][0] is 'p' else False
print(argv)
print(people)

entities = read_entities(input_file)
entity2same = generate_names(entities, people)
bucket_dict = load_buckets(entity2same)
# for key in bucket_dict:
# 	print('key {} value {}'.format(key, bucket_dict[key]).encode('utf-8'))
print(get_stats(entity2same, bucket_dict))