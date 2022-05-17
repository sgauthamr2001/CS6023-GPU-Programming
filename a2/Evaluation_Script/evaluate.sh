#!/bin/bash

CURRENT_DIR="$(pwd)"
INPUT="${CURRENT_DIR}/testcases/input"
OUTPUT="${CURRENT_DIR}/testcases/output"

touch "A2-Marks.txt"
MARKSFILE=${CURRENT_DIR}/A2-Marks.txt

echo "====================START==========================" >> "$MARKSFILE"
date >> "$MARKSFILE"

cd submit/
for FILE in *.cu
do	
	ROLLNO=$(echo $FILE | cut -d'.' -f1)
	LOGFILE=${ROLLNO}.log
	
	nvcc ${ROLLNO}.cu -o A2
	
	if [ $? -ne 0 ] 
	then
		echo ${ROLLNO}, BUILD FAILED!
    	echo ${ROLLNO}, BUILD FAILED! >> "$MARKSFILE"
		continue
	fi
		
	date >> "$LOGFILE"

	for INPUT_FILE in "$INPUT"/*
	do
		echo ${INPUT_FILE}
		FILE_NUM=$(echo $(basename "$INPUT_FILE") | cut -d'.' -f1 | cut -d't' -f2 )
		OUTPUT_FILE_NAME="output${FILE_NUM}.txt"
		
		./A2 "$INPUT_FILE" $OUTPUT_FILE_NAME >> "$LOGFILE"
		diff -w "$OUTPUT/$OUTPUT_FILE_NAME" $OUTPUT_FILE_NAME > /dev/null 2>&1
		exit_code=$?
		if (($exit_code == 0)); then
		  	echo "success" >> $LOGFILE
		else 
			echo "failure" >> $LOGFILE
		fi
	done
  
	SCORE=$(grep -ic success "$LOGFILE")
	echo ${ROLLNO}, ${SCORE} 
	echo ${ROLLNO}, ${SCORE} >> "$MARKSFILE"
done

# rm *.txt *.log A2    # uncomment this line to do the clean up
cd ..

date >> "$MARKSFILE"
echo "====================DONE!==========================" >> "$MARKSFILE"
