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

echoTitle 'Experimenting with Image Auto Encoder - A regression problem with multi-dimensional output'
file="Models/End2End"
runCommand "python3 Test.py -model=${file}.fbm ${GPUS}"
echoCheckPoint "PSNR: around 30.137647"

runCommand "python3 Reduce.py -in=${file}.fbm -out=${file}R.fbm -layers=S1_L2_CONV,S2_L2_CONV"
echoCheckPoint "New Number of parameters: around 63,635"

runCommand "python3 Test.py -model=${file}R.fbm ${GPUS}"
echoCheckPoint "PSNR: around 6.693979"

runCommand "python3 Retrain.py -in=${file}R.fbm -out=${file}RR.fbm -restart ${GPUS}"
echoCheckPoint "PSNR: around 29.461072"

runCommand "python3 Quantize.py -in=${file}RR.fbm -out=${file}RRQ.fbm -qInfo=1,8,.0001,.8"
echoCheckPoint "New File Size: around 70K"

runCommand "python3 Test.py -model=${file}RRQ.fbm ${GPUS}"
echoCheckPoint "PSNR: around 19.862514"

runCommand "python3 Retrain.py -in=${file}RRQ.fbm -out=${file}RRQR.fbm -restart ${GPUS}"
echoCheckPoint "PSNR: around 28.970639"

runCommand "python3 Infer.py -model=${file}RRQR.fbm ${GPUS}"
echoCheckPoint "MSE: around 99.5 (This is test MSE)"

echoTitle 'Testing parameter search'
runCommand "python3 Retrain.py -in=${file}RRQ.fbm -paramSearch -restart ${GPUS}"
echoCheckPoint "Check the <TrainParamSearch.csv> file."

# CoreML can only be tested on Mac:
if [ ${machine} = "Mac" ]; then
    runCommand "python3 ExportCoreML.py -in=${file}RR.fbm -out=${file}RR.mlmodel"
fi
