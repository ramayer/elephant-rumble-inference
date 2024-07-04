# Elephant Rumble Inference

Packaged the AVES/HuBERT transformer based classifier as a python CLI. 

## Usage

Using it should be as simple as

```
    pip install git+https://github.com/ramayer/elephant-rumble-inference@v0.2.1
```

Followed by running the program on an audio file like this: 

```
!elephant-rumble-inference \
            --save-raven --save-scores \
            --limit-audio-hours=1 \
            --visualizations-per-audio-file=1 \
            --visualization-duration=60 \
            --load-labels ./data/Rumble \
            --save-dir /tmp/classified_audio \
            ./data/Rumble/Training/Sounds/CEB1_20111010_000000.wav
```

Most options except the audio file(s) are optional.

* `--save-raven` - will save a raven file of the sections of the audio file it thinks are rumbles
* `--visualizations-per-audio-file=3` - will save images showing how it classified different sections of the audio.
* `--save-scores` - saves a file showing how it classified each 320ms of audio in the file.
* `--load-labels` - will overlay labels from a raven file on a visualization.


## Performance Notes

This really wants a GPU.   

* It processes 24 hours of audio in 22 seconds on a 2060
* It processes 24 hours of audio in 28 minutes on a CPU

so about a 70x speedup on a GPU.

## Windows instructions

* I was only able to make this work using conda
* conda install ffmpeg 
* torchaudio StreamReaders use ffmpeg?

# TODO

    - annoying bug --- Every hour my labels slip by a sample :(
      So by hour 19, the green things are many seconds before where they should be

    - annoying bug -- the final hour is often not a full 60 minutes; and
      my two graphs aren't lining up.
