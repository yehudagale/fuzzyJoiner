#!/bin/bash
temp_psql=psql
number=$6
temp_python=python
input_file=$1
proccess_method=$2
user=$3
passsword=$4
db_name=$5
new=1
usage="run_script 1:input_file 2:proccess_method 3:user 4:passsword 5:db_name  [-n:number -s:psql location -p python location -o: don't clean new data]"
number=
if [ "$1" = "" ]; then
	echo $usage
else
	shift 5
	while [ "$1" != "" ]; do
	    case $1 in
	        -n | --number )           shift
	                                number=$1
	                                ;;
	        -s | --psql )    		shift
									temp_psql=$1
	                                ;;
	        -o | --old    )          new=0
									;;
	        -p | --python )        	shift
									temp_python=$1
	]                                ;;
	        -h | help )             echo $usage
	                                exit 1
	    esac
	    shift
	done
	echo $old
	if [ "$new" = "1" ]; then
		if [ "$number" = 0 ]; then
			$temp_python ./cleanser.py $input_file testout.csv $proccess_method
		else
			$temp_python ./cleanser.py $input_file testout.csv $proccess_method get $number
		fi
		$temp_psql $db_name -f ./proccess_data.sql
	fi
	$temp_python ./matcher.py $user $passsword $db_name
fi
