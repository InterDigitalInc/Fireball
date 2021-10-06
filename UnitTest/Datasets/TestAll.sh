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

echoTitle 'Testing Audio Dataset'
runCommand "python3 AudioDSetUnitTest.py"

echoTitle 'Testing CIFAR-100 Dataset'
runCommand "python3 CifarDSetUnitTest.py"

echoTitle 'Testing ImageNet Dataset'
runCommand "python3 ImageNetDSetUnitTest.py"

echoTitle 'Testing MNIST Dataset'
runCommand "python3 MnistDSetUnitTest.py"

echoTitle 'Testing RadioML Dataset'
runCommand "python3 RadioMlDSetUnitTest.py"

echoTitle 'Testing Coco Dataset'
runCommand "python3 CocoDSetUnitTest.py"

echoTitle 'Testing SQuAD Dataset'
runCommand "python3 SquadDSetUnitTest.py"

echoTitle 'Testing GLUE Dataset'
runCommand "python3 GlueDSetUnitTest.py"
