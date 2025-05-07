#!/bin/bash

odir=./split_wav_16k
mkdir -p $odir

for type in P C riko
do 
    dirs=`ls -d ./split_wav/$type*`
    
    for dir in $dirs
    do
        id=`basename $dir`
        mkdir -p $odir/$id

        #echo $dir
        paths=`ls $dir/*.wav`
        for fn in $paths
        do
           echo $fn
        
            bn=`basename $fn .wav`
            of=$odir/$id/$bn.wav
            echo $of
            ffmpeg -i $fn -ar 16000 $of
        done

    done
done
