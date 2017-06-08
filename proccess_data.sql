--used http://www.postgresqltutorial.com/import-csv-file-into-posgresql-table/
--used https://stackoverflow.com/questions/8584119/how-to-apply-a-function-to-each-element-of-an-array-column-in-postgres to help learn function synatax
--used how-to-find-duplicate-records-in-posgresql
--done to make sure that if the tables already exist they do not interfere
DROP TABLE aliases;
DROP TABLE finalTable;
DROP TABLE matches;
DROP TABLE duplicatedTable;
DROP TABLE match;
CREATE TABLE aliases(
	aliase1 text NOT NULL,
	aliase2 text NOT NULL
);
\copy aliases(aliase1, aliase2) FROM './testout.csv' DELIMITER '|' CSV;
WITH --FIX1 AS (SELECT replace(aliase1, 'John ', '') AS aliase1, replace(aliase2, 'John ', '') AS aliase2 FROM aliases),
--FIX2 AS (SELECT replace(aliase1, 'William ', '') AS aliase1, replace(aliase2, 'William ', '') AS aliase2 FROM FIX1),
--FIXED AS (SELECT replace(aliase1, 'of ', '') AS aliase1, replace(aliase2, 'of ', '') AS aliase2 FROM FIX2), 
TMP1 AS (SELECT aliase1 AS name, string_to_array(aliase1, ' ') AS words FROM aliases),
TMP2 AS (SELECT aliase2 AS name, string_to_array(aliase2, ' ') AS words FROM aliases),
TMP3 AS (SELECT generate_subscripts(words, 1) AS s, words AS words, name AS name FROM TMP1),
TMP4 AS (SELECT generate_subscripts(words, 1) AS s, words, name FROM TMP2),
TMP5 AS (SELECT name, words[s] AS word FROM TMP3),
TMP6 AS (SELECT name, words[s] AS word FROM TMP4),
--SELECT word, count(*) AS frequency INTO finalTable From TMP7 Group By word ORDER BY frequency DESC;
--SELECT DISTINCT TMP5.name AS name1, TMP6.name AS name2 INTO finalTable FROM TMP5, TMP6 WHERE TMP5.word = TMP6.word AND TMP5.word <> 'John' AND TMP5.word <> 'of' AND TMP5.word <> 'William';
TMP AS (SELECT array_remove(array_remove(string_to_array(TMP5.name, ' '), 'Sir'), 'Jr.') AS array1, TMP5.name AS name1, string_to_array(TMP6.name, ' ') AS array2, TMP6.name AS name2 FROM TMP5, TMP6 
	WHERE TMP5.word = TMP6.word AND TMP5.word <> 'John' AND TMP5.word <> 'of' AND TMP5.word <> 'William')
SELECT DISTINCT TMP.name1, TMP.name2 INTO matches FROM TMP WHERE (TMP.array1 @> TMP.array2 OR TMP.array2 @> TMP.array1);

--this many possible:
--SELECT COUNT(finalTable.name1) FROM finalTable, aliases WHERE finalTable.name1 = aliases.aliase1 AND finalTable.name2 = aliases.aliase2;
--This many correct matches:
SELECT COUNT(matches.name1) FROM matches, aliases WHERE matches.name1 = aliases.aliase1 AND matches.name2 = aliases.aliase2;
--These are the false positives
--SELECT * FROM matches EXCEPT SELECT matches.name1, matches.name2 FROM matches, aliases WHERE matches.name1 = aliases.aliase1 AND matches.name2 = aliases.aliase2;

--Incorrect mathches is size of matches - correct matches, Missed matches is size of aliases - correct matches
--SELECT * FROM finalTable;