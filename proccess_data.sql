--used http://www.postgresqltutorial.com/import-csv-file-into-posgresql-table/
CREATE TABLE aliases(
	aliase1 VARCHAR(300) NOT NULL,
	aliase2 VARCHAR(300) NOT NULL
);
\copy aliases(aliase1, aliase2) FROM './testout.csv' DELIMITER '|' CSV;
SELECT * FROM aliases;
DROP TABLE aliases