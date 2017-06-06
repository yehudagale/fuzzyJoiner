--used http://www.postgresqltutorial.com/import-csv-file-into-posgresql-table/
--used https://stackoverflow.com/questions/8584119/how-to-apply-a-function-to-each-element-of-an-array-column-in-postgres to help learn function synatax
#done to make sure that if the tables already exist they do not interfere
DROP TABLE aliases;
DROP TABLE final;
DROP TABLE wordTable;
DROP TABLE arrays;
CREATE TABLE aliases(
	aliase1 text NOT NULL,
	aliase2 text NOT NULL
);
\copy aliases(aliase1, aliase2) FROM './testout.csv' DELIMITER '|' CSV;
CREATE TABLE arrays(
	name text NOT NULL,
	words text[] NOT NULL
);
INSERT INTO arrays SELECT aliase1, regexp_split_to_array(aliase1, E'\\s+') FROM aliases;
INSERT INTO arrays SELECT aliase2, regexp_split_to_array(aliase2, E'\\s+') FROM aliases;
CREATE TABLE wordTable(
	name text NOT NULL,
	word text NOT NULL
);

INSERT INTO wordTable
SELECT name AS name, words[s] AS word
FROM (SELECT generate_subscripts(words, 1) AS s, words, name FROM arrays) foo;
CREATE TABLE final(
	name1 text NOT NULL,
	name2 text NOT NULL
);
INSERT INTO final(name1, name2)
SELECT DISTINCT A.name AS name1, B.name AS name2 FROM wordTable A, wordTable B WHERE A.word = B.word AND A.name <> B.name;
SELECT * FROM final;
