#!/bin/bash
temp_psql=psql
number=$6
temp_python=python
input_file=$1
process_method=$2
user=$3
passsword=$4
db_name=$5
new=1
machine=0
usage=$'Usage: run_script input_file process_method user_name passsword db_name [OPTIONS]
  -n, --number          process only this many names
  -s, --psql            use this location for psql instead of the default
  -p, --python          use this location for python instead of the default
  -o, --old             use the names already in the database instead of processing new ones
  -m, --machine         use the machine learning algorithm instead of the rule based one
  -h, --help            display this help and exit'
number=
if [ "$1" = "" ]; then
	echo "$usage"
else
	shift 5
	while [ "$1" != "" ]; do
	    case $1 in
	        -n | --number )           shift
	                                number=$1
	                                ;;
	        -m | --machine )        machine=1
	                                ;;
	        -s | --psql )    		shift
									temp_psql=$1
	                                ;;
	        -o | --old    )          new=0
									;;
	        -p | --python )        	shift
									temp_python=$1
	                                ;;
	        -h | help )             echo "$usage"
	                                exit 1
	    esac
	    shift
	done
	echo $old
	if [ "$new" = "1" ]; then
		if [ "$number" = 0 ]; then
			$temp_python ./names_cleanser.py -f $input_file -o ./Machine_Learning/nerData/cleansedData.txt -t $process_method
		else
			$temp_python ./names_cleanser.py -f $input_file -o ./Machine_Learning/nerData/cleansedData.txt -t $process_method -u get -n $number
		fi
		$temp_psql $db_name -U $user -f ./process_data.sql
	fi
	if [ "$machine" = "1" ]; then
		$temp_python ./Named_Entity_Recognition_Modified.py $user $passsword $db_name
	else
		$temp_python ./matcher.py $user $passsword $db_name
	fi
fi