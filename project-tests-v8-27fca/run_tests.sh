#!/bin/bash

set -uo pipefail

SPEED=${1:-false}

SCRIPT_DIR=`dirname $0`
SRC_DIR=${SCRIPT_DIR}/*_*_project/
SYMNMF_TESTS_DIR=${SCRIPT_DIR}/tests/
KMEANS_TESTS_DIR=${SCRIPT_DIR}/kmeans_tests/

RED="\033[31m"
GREEN="\033[32m"
RESET="\033[0m"

failures=0
total_tests=0

function assertByFile() {
	command=$1
	expectedFilePath=$2
	name=$3

	FIXED_WIDTH=65

	((total_tests++))

	stderr_output=`mktemp /tmp/tmpfile.XXXXXX`
	stdout_output=`mktemp /tmp/tmpfile.XXXXXX`
	eval ${command} 2>${stderr_output} > ${stdout_output} || true
	stdout_with_line_numbers=`mktemp /tmp/tmpfile.XXXXXX`
	expected_with_line_numbers=`mktemp /tmp/tmpfile.XXXXXX`
	diff_output=`mktemp /tmp/tmpfile.XXXXXX`
	nl -v 0 ${stdout_output} > ${stdout_with_line_numbers}
	nl -v 0 ${expectedFilePath} > ${expected_with_line_numbers}
	printf "%-${FIXED_WIDTH}s %s" "${name}"
	diff ${expected_with_line_numbers} ${stdout_with_line_numbers} > ${diff_output} && echo -e "${GREEN}passed${RESET}" && return 0 || echo -e "${RED}FAILED${RESET}"
	((failures++))
	echo -e "\nFailed command: ${command}\n"
	echo "--- captured STDERR ---"
	cat ${stderr_output}
	echo "--- actual ---"
	echo original: ${stdout_output}
	echo with line numbers: ${stdout_with_line_numbers}
	echo "--- expected ---"
	echo original: ${expectedFilePath}
	echo with line numbers: ${expected_with_line_numbers}
	echo "--- diff ---"
	echo diff: ${diff_output}
	echo "________________"
	echo
	return 1
}

function testSymnmfSingleGoal() {
	goal=$1
	k=$2
	inputFilePath=${SYMNMF_TESTS_DIR}/input_${3}.txt
	expectedFilePath=${SYMNMF_TESTS_DIR}/output_${4}.txt
	mode=$5
	test_name=$6
	python_executable=${7:-symnmf.py}

	C_RUN_COMMAND="valgrind --quiet --leak-check=full ${SRC_DIR}/symnmf ${goal} ${inputFilePath}"
	PYTHON_RUN_COMMAND="python3 ${SRC_DIR}/${python_executable} ${k} ${goal} ${inputFilePath}"
	if [[ "$mode" == *"c"* ]]; then
		assertByFile "${C_RUN_COMMAND}" "${expectedFilePath}" 		"[C]        ${inputFilePath} - ${test_name}"
	fi
	if [[ "$mode" == *"python"* ]]; then
		assertByFile "${PYTHON_RUN_COMMAND}" "${expectedFilePath}" 	"[Python]   ${inputFilePath} - ${test_name}"
    fi
}

function testSymnmfGaolsWithK() {
	inputFileName=$1
	expectedFileName=$2
	k=$3
	name_ext=${4:-"k${k}"}
	if [[ "${SPEED}" != *"quick"* ]]; then
		testSymnmfSingleGoal symnmf ${k} ${inputFileName} ${expectedFileName}_symnmf "python" symnmf-${name_ext}
	fi
	testSymnmfSingleGoal "" ${k} ${inputFileName} ${expectedFileName}_analysis "python" analysis-${name_ext} "analysis.py"
}

function testSymnmfAllGoals() {
	inputFileName=$1
	expectedFileName=$2
	k=${3:-0}

	if [[ "${SPEED}" != *"quick"* ]]; then
		testSymnmfSingleGoal sym ${k} ${inputFileName} ${expectedFileName}_sym "c python" sym
		testSymnmfSingleGoal ddg ${k} ${inputFileName} ${expectedFileName}_ddg "c python" ddg
		testSymnmfSingleGoal norm ${k} ${inputFileName} ${expectedFileName}_norm "c python" norm
	fi
	testSymnmfGaolsWithK $@ 2
}

function testKmeans() {
	k_and_maxIter=$1
	inputFilePath=${KMEANS_TESTS_DIR}/input_${2}.txt
	expectedFilePath=${KMEANS_TESTS_DIR}/output_${3}.txt

	KMEANS_RUN_COMMAND="python3 ${SRC_DIR}/kmeans.py ${k_and_maxIter} ${inputFilePath}"
	assertByFile "${KMEANS_RUN_COMMAND}" ${expectedFilePath} "[Kmeans]   ${inputFilePath} - args: \"${k_and_maxIter}\""
}

function symnmfSuite() {
	echo ">>> SymNMF <<<"

	if [[ "${SPEED}" == *"edge"* ]]; then
		testSymnmfAllGoals "not-existing" general_error  # Not existing file
		testSymnmfAllGoals "empty" general_error
		testSymnmfGaolsWithK 1 1_k292 292 k-almost-too-big
		testSymnmfGaolsWithK 1 general_error 293 k-too-big
		testSymnmfGaolsWithK 1 general_error 999999 k-very-too-big
		testSymnmfGaolsWithK 1 general_error 1 k-too-small-1
		testSymnmfGaolsWithK 1 general_error 0 k-too-small-0
		testSymnmfGaolsWithK 1 general_error -1 k-negative
		testSymnmfGaolsWithK 1 general_error "abc" k-invalid
		testSymnmfSingleGoal "" "" 1 general_error "c python" "Invalid goal"
	fi

	if [[ "${SPEED}" == *"slow"* ]]; then
		testSymnmfAllGoals 1 1
		testSymnmfAllGoals 2 2
		testSymnmfAllGoals 3 3
	fi

	if [[ "${SPEED}" == *"edge"* ]]; then
		testSymnmfAllGoals 4_three_points 4_three_points
	fi

	if [[ "${SPEED}" == *"edge"* ]]; then
		testSymnmfAllGoals 5_invalid general_error
	fi

	testSymnmfAllGoals 6 6
	testSymnmfAllGoals 7 7
	testSymnmfAllGoals 8 8

	echo
}

function kmeansSuite() {
	echo ">>> Kmeans <<<"

	testKmeans "3 600" 1 1
	testKmeans "7" 2 2
	testKmeans "15 300" 3 3
	testKmeans "2 2" 4 4__2_2
	testKmeans "7 999" 4 4__7_999
	testKmeans "8 2" 4 invalid_clusters
	testKmeans "1 2" 4 invalid_clusters
	testKmeans "-2 2" 4 invalid_clusters
	testKmeans "a 2" 4 invalid_clusters
	testKmeans "2 1000" 4 invalid_maxIter
	testKmeans "2 1" 4 invalid_maxIter
	testKmeans "2 -2" 4 invalid_maxIter
	testKmeans "2 a" 4 invalid_maxIter
	testKmeans "2 2" 5_invalid general_error
	testKmeans "" 4 general_error
	testKmeans "2 2 3" 4 general_error

	echo
}

function cleanTmp() {
	if [ "$(ls -1 /tmp/tmpfile.?????? | wc -l)" -gt 600 ]; then
		echo -e "\n! Removing old temporary files !\n"
		ls -tr /tmp/tmpfile.?????? | head -n 300 | while read file; do rm "$file"; done || true
	fi
}

cleanTmp

if [[ "${SPEED}" == *"kmeans"* ]]; then
	kmeansSuite
fi
symnmfSuite

if [ $failures -eq 0 ]; then
	echo -e "\n${GREEN}=== ${total_tests}/${total_tests} tests passed successfully ===${RESET}\n"
else
	echo -e "\n${RED}=== ${failures}/${total_tests} tests failed ===${RESET}\n"
	exit 1
fi
