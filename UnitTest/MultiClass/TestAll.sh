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

echoTitle 'Experimenting with LeNet-5 (with an additional batch-norm layer)'
file="Models/MultiClassTestL1"
runCommand "python3 Train.py -layers=1 -out=${file}.fbm -restart ${GPUS}"
echoCheckPoint "Accuracy: around 0.98"
runCommand "python3 Infer.py -model=${file}.fbm ${GPUS}"
echoCheckPoint "Accuracy: around 0.98 (This is test accuracy)"
runCommand "python3 Reduce.py -in=${file}.fbm -out=${file}R.fbm -layers=L2_CONV,L4_FC,L5_FC"
echoCheckPoint "New Number of parameters: around 7K"
runCommand "python3 Test.py -model=${file}R.fbm ${GPUS}"
echoCheckPoint "Accuracy: around 0.40"
runCommand "python3 Retrain.py -in=${file}R.fbm -out=${file}RR.fbm -restart ${GPUS}"
echoCheckPoint "Accuracy: around 0.90"
runCommand "python3 Quantize.py -in=${file}RR.fbm -out=${file}RRQ.fbm -qInfo=1,8,.01,.8"
echoCheckPoint "New File Size: around 7K"
runCommand "python3 Test.py -model=${file}RRQ.fbm ${GPUS}"
echoCheckPoint "Accuracy: around 0.25"
runCommand "python3 Retrain.py -in=${file}RRQ.fbm -out=${file}RRQR.fbm -restart ${GPUS}"
echoCheckPoint "Accuracy: around 0.60"

echoTitle 'Training a low-rank model based on LeNet-5 from scratch'
file="Models/MultiClassTestL2"
runCommand "python3 Train.py -layers=2 -out=${file}.fbm -restart ${GPUS}"
echoCheckPoint "Accuracy: around 0.98"
runCommand "python3 Quantize.py -in=${file}.fbm -out=${file}Q.fbm -qInfo=1,8,.01,.8"
echoCheckPoint "New File Size: around 17K"
runCommand "python3 Test.py -model=${file}Q.fbm ${GPUS}"
echoCheckPoint "Accuracy: around 0.75"
runCommand "python3 Retrain.py -in=${file}Q.fbm -out=${file}QR.fbm -restart ${GPUS}"
echoCheckPoint "Accuracy: around 0.95"

echoTitle 'Experimenting with LeNet-5 with some MobileNet-V2 blocks'
file="Models/MultiClassTestL3"
runCommand "python3 Train.py -layers=3 -out=${file}.fbm -restart ${GPUS}"
echoCheckPoint "Accuracy: around 0.98"
runCommand "python3 Reduce.py -in=${file}.fbm -out=${file}R.fbm -layers=L2_CONV,L3_MN1S,L5_FC"
echoCheckPoint "New Number of parameters: around 14K"
runCommand "python3 Test.py -model=${file}R.fbm ${GPUS}"
echoCheckPoint "Accuracy: around 0.90"
runCommand "python3 Retrain.py -in=${file}R.fbm -out=${file}RR.fbm -restart ${GPUS}"
echoCheckPoint "Accuracy: around 0.95"
runCommand "python3 Quantize.py -in=${file}RR.fbm -out=${file}RRQ.fbm -qInfo=1,8,.01,.8"
echoCheckPoint "New File Size: around 18K"
runCommand "python3 Test.py -model=${file}RRQ.fbm ${GPUS}"
echoCheckPoint "Accuracy: around 0.30"
runCommand "python3 Retrain.py -in=${file}RRQ.fbm -out=${file}RRQR.fbm -restart ${GPUS}"
echoCheckPoint "Accuracy: around 0.80"

echoTitle 'Experimenting with LeNet-5 with some ResNet50 blocks'
file="Models/MultiClassTestL4"
runCommand "python3 Train.py -layers=4 -out=${file}.fbm -restart ${GPUS}"
echoCheckPoint "Accuracy: around 0.99"
runCommand "python3 Reduce.py -in=${file}.fbm -out=${file}R.fbm -layers=L3_RES2,L4_RES1,L5_FC"
echoCheckPoint "New Number of parameters: around 23K"
runCommand "python3 Test.py -model=${file}R.fbm ${GPUS}"
echoCheckPoint "Accuracy: around 0.40"
runCommand "python3 Retrain.py -in=${file}R.fbm -out=${file}RR.fbm -restart ${GPUS}"
echoCheckPoint "Accuracy: around 0.70"
runCommand "python3 Quantize.py -in=${file}RR.fbm -out=${file}RRQ.fbm -qInfo=2,8,.01,.8"
echoCheckPoint "New File Size: around 27K"
runCommand "python3 Test.py -model=${file}RRQ.fbm ${GPUS}"
echoCheckPoint "Accuracy: around 0.50"
runCommand "python3 Retrain.py -in=${file}RRQ.fbm -out=${file}RRQR.fbm -restart ${GPUS}"
echoCheckPoint "Accuracy: around 0.94"

echoTitle 'Testing parameter search'
runCommand "python3 Train.py -paramSearch -restart ${GPUS}"
echoCheckPoint "Check the <TrainParamSearch.csv> file."

# CoreML can only be tested on Mac:
if [ ${machine} = "Mac" ]; then
    file="Models/MultiClassTestL1RR"
    runCommand "python3 ExportCoreML.py -in=${file}.fbm -out=${file}.mlmodel"
fi
