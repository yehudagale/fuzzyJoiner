--used http://www.postgresqltutorial.com/import-csv-file-into-posgresql-table/
--used https://stackoverflow.com/questions/8584119/how-to-apply-a-function-to-each-element-of-an-array-column-in-postgres to help learn function synatax
--done to make sure that if the tables already exist they do not interfere
DROP TABLE aliases;
DROP TABLE finaltablea;
CREATE TABLE aliases(
	aliase1 text NOT NULL,
	aliase2 text NOT NULL
);
\copy aliases(aliase1, aliase2) FROM './testout.csv' DELIMITER '|' CSV;
WITH --FIX1 AS (SELECT replace(aliase1, 'John ', '') AS aliase1, replace(aliase2, 'John ', '') AS aliase2 FROM aliases),
--FIX2 AS (SELECT replace(aliase1, 'William ', '') AS aliase1, replace(aliase2, 'William ', '') AS aliase2 FROM FIX1),
--FIXED AS (SELECT replace(aliase1, 'of ', '') AS aliase1, replace(aliase2, 'of ', '') AS aliase2 FROM FIX2), 
FIX1 AS (SELECT replace(aliase1, 'John ', '') AS aliase1, replace(aliase2, 'John ', '') AS aliase2 FROM aliases),
FIX2 AS (SELECT replace(aliase1, 'William ', '') AS aliase1, replace(aliase2, 'William ', '') AS aliase2 FROM FIX1),
FIXED AS (SELECT replace(aliase1, 'of ', '') AS aliase1, replace(aliase2, 'of ', '') AS aliase2 FROM FIX2), 
TMP1A AS (SELECT aliase1 AS name, string_to_array(aliase1, ' ') AS words FROM FIXED),
TMP2A AS (SELECT aliase2 AS name, string_to_array(aliase2, ' ') AS words FROM FIXED),
TMP3A AS (SELECT generate_subscripts(words, 1) AS s, words AS words, name AS name FROM TMP1a),
TMP4A AS (SELECT generate_subscripts(words, 1) AS s, words, name FROM TMP2a),
TMP5A AS (SELECT name, words[s] AS word FROM TMP3a),
TMP6A AS (SELECT name, words[s] AS word FROM TMP4a)
--SELECT word, count(*) AS frequency INTO finalTable From TMP7 Group By word ORDER BY frequency DESC;
SELECT DISTINCT TMP5a.name AS name1, TMP6a.name AS name2 INTO finaltablea FROM TMP5a, TMP6a WHERE TMP5a.word = TMP6a.word;
--This many successes:
SELECT COUNT(finaltablea.name1) AS successes FROM finaltablea, aliases WHERE finaltablea.name1 = aliases.aliase1 AND finaltablea.name2 = aliases.aliase2;
--SELECT * FROM finalTable;