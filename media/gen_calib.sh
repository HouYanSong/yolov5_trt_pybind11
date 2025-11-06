#!/bin/bash

cd "$(dirname "$0")"
rm -f *.png
ffmpeg -i sample_720p.mp4 -vf "select=not(mod(n\,10))" -vsync vfr sample%04d.png
ls *.png > filelist.txt