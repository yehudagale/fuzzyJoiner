--used http://www.postgresqltutorial.com/import-csv-file-into-posgresql-table/
--used https://stackoverflow.com/questions/8584119/how-to-apply-a-function-to-each-element-of-an-array-column-in-postgres to help learn function synatax
--done to make sure that if the tables already exist they do not interfere
DROP TABLE aliases;
DROP TABLE finalTable;
CREATE TABLE aliases(
	aliase1 text NOT NULL,
	aliase2 text NOT NULL
);
\copy aliases(aliase1, aliase2) FROM './testout.csv' DELIMITER '|' CSV;
WITH TMP1 AS (SELECT aliase1 AS name, string_to_array(aliase1, ' ') AS words FROM aliases),
TMP2 AS (SELECT aliase2 AS name, string_to_array(aliase2, ' ') AS words FROM aliases),
TMP3 AS (SELECT generate_subscripts(words, 1) AS s, words AS words, name AS name FROM TMP1),
TMP4 AS (SELECT generate_subscripts(words, 1) AS s, words, name FROM TMP2),
TMP5 AS (SELECT name, words[s] AS word FROM TMP3),
TMP6 AS (SELECT name, words[s] AS word FROM TMP4)
SELECT DISTINCT TMP5.name AS name1, TMP6.name AS name2 INTO finalTable FROM TMP5, TMP6 WHERE TMP5.word = TMP6.word;
