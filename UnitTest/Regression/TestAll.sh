#!/bin/bash
unameOut="$(uname -s)"
case "${unameOut}" in
    Linux*)     machine=Linux;;
    Darwin*)    machine=Mac;;
    CYGWIN*)    machine=Cygwin;;
    MINGW*)     machine=MinGw;;
    *)          machine="UNKNOWN:${unameOut}"
esac

if [ ${machine} = "Mac" ]; then
    ColorEcho="echo"
    GPUS=""
else
    ColorEcho="echo"
    GPUS="-gpus=0"
fi

runCommand()
{
    $ColorEcho '\033[96m>>> '${1}'\033[0m'
    $1 || exit 1
}

echoTitle()
{
    $ColorEcho '\033[93m'"****************************************************************************************************"'\033[0m'
    $ColorEcho '\033[93m'${1}'\033[0m'
}

echoCheckPoint()
{
    $ColorEcho '\033[92mCHECK >>> ^^^^ '${1}' ^^^^\033[0m'
}

echoTitle 'Experimenting with regression MNIST - A regression problem with one output'
file="Models/RegMnist"
runCommand "python3 Train.py -out=${file}.fbm -restart ${GPUS}"
echoCheckPoint "MSE: around 0.3"

runCommand "python3 Reduce.py -in=${file}.fbm -out=${file}R.fbm -layers=L2_CONV,L3_FC,L4_FC"
echoCheckPoint "New Number of parameters: around 38K"

runCommand "python3 Test.py -model=${file}R.fbm ${GPUS}"
echoCheckPoint "MSE: around 0.3"

runCommand "python3 Retrain.py -in=${file}R.fbm -out=${file}RR.fbm -restart ${GPUS}"
echoCheckPoint "MSE: around 0.25"

runCommand "python3 Quantize.py -in=${file}RR.fbm -out=${file}RRQ.fbm -qInfo=1,8,.0001,.8"
echoCheckPoint "New File Size: around 36K"

runCommand "python3 Test.py -model=${file}RRQ.fbm ${GPUS}"
echoCheckPoint "MSE: around 0.3"

runCommand "python3 Retrain.py -in=${file}RRQ.fbm -out=${file}RRQR.fbm -restart ${GPUS}"
echoCheckPoint "MSE: around 0.27"

runCommand "python3 Infer.py -model=${file}RRQR.fbm ${GPUS}"
echoCheckPoint "MSE: around 0.24 (This is test MSE)"

echoTitle 'Testing parameter search'
runCommand "python3 Retrain.py -in=${file}RRQ.fbm -paramSearch -restart ${GPUS}"
echoCheckPoint "Check the <TrainParamSearch.csv> file."

# CoreML can only be tested on Mac:
if [ ${machine} = "Mac" ]; then
    runCommand "python3 ExportCoreML.py -in=${file}RR.fbm -out=${file}RR.mlmodel"
fi
