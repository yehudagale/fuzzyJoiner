--used http://www.postgresqltutorial.com/import-csv-file-into-posgresql-table/
--used https://stackoverflow.com/questions/8584119/how-to-apply-a-function-to-each-element-of-an-array-column-in-postgres to help learn function synatax
--used how-to-find-duplicate-records-in-posgresql
--done to make sure that if the tables already exist they do not interfere
DROP TABLE aliases;
DROP TABLE finalTable;
DROP TABLE matches;
DROP TABLE duplicatedTable;
DROP TABLE match;
DROP TABLE WORDTABLE1;
DROP TABLE WORDTABLE2;
CREATE TABLE aliases(
	alias1 text NOT NULL,
	alias2 text NOT NULL
);
\copy aliases(alias1, alias2) FROM './testout.csv' DELIMITER '|' CSV;
 --FIX1 AS (SELECT replace(alias1, 'John ', '') AS alias1, replace(alias2, 'John ', '') AS alias2 FROM aliases),
--FIX2 AS (SELECT replace(alias1, 'William ', '') AS alias1, replace(alias2, 'William ', '') AS alias2 FROM FIX1),
--FIXED AS (SELECT replace(alias1, 'of ', '') AS alias1, replace(alias2, 'of ', '') AS alias2 FROM FIX2),
UPDATE aliases SET (alias1, alias2) = (SELECT lower(alias1), lower(alias2) FROM aliases); 
WITH TMP1 AS (SELECT alias1 AS name, string_to_array(alias1, ' ') AS words FROM aliases),
TMP3 AS (SELECT generate_subscripts(words, 1) AS s, words AS words, name AS name FROM TMP1)
SELECT name, words[s] AS word INTO WORDTABLE1  FROM TMP3 ORDER BY word ASC;
WITH TMP2 AS (SELECT alias2 AS name, string_to_array(alias2, ' ') AS words FROM aliases),
TMP4 AS (SELECT generate_subscripts(words, 1) AS s, words, name FROM TMP2)
SELECT name, words[s] AS word INTO WORDTABLE2 FROM TMP4 ORDER BY word ASC;

--SELECT word, count(*) AS frequency INTO finalTable From TMP7 Group By word ORDER BY frequency DESC;
--SELECT DISTINCT TMP5.name AS name1, TMP6.name AS name2 INTO finalTable FROM TMP5, TMP6 WHERE TMP5.word = TMP6.word AND TMP5.word <> 'John' AND TMP5.word <> 'of' AND TMP5.word <> 'William';
--TMP AS (SELECT array_remove(array_remove(string_to_array(TMP5.name, ' '), 'Sir'), 'Jr.') AS array1, TMP5.name AS name1, string_to_array(TMP6.name, ' ') AS array2, TMP6.name AS name2 FROM TMP5, TMP6 
--	WHERE TMP5.word = TMP6.word AND TMP5.word <> 'John' AND TMP5.word <> 'of' AND TMP5.word <> 'William')
--SELECT DISTINCT TMP.name1, TMP.name2 INTO matches FROM TMP WHERE (TMP.array1 @> TMP.array2 OR TMP.array2 @> TMP.array1);

--this many possible:
--SELECT COUNT(finalTable.name1) FROM finalTable, aliases WHERE finalTable.name1 = aliases.alias1 AND finalTable.name2 = aliases.alias2;
--This many correct matches:
--SELECT COUNT(matches.name1) FROM matches, aliases WHERE matches.name1 = aliases.alias1 AND matches.name2 = aliases.alias2;
--These are the false positives
--SELECT * FROM matches EXCEPT SELECT matches.name1, matches.name2 FROM matches, aliases WHERE matches.name1 = aliases.alias1 AND matches.name2 = aliases.alias2;

--Incorrect mathches is size of matches - correct matches, Missed matches is size of aliases - correct matches
--SELECT * FROM finalTable;