# Elephant Rumble Inference

Packaged the AVES/HuBERT transformer based classifier as a python CLI. 

## Usage

Using it should be as simple as

```
    pip install git+https://github.com/ramayer/elephant-rumble-inference@v0.1.3-demo
```

Followed by running the program on an audio file like this: 

```
> elephant-rumble-inference ./data/Rumble/Training/Sounds/CEB1_20111010_000000.wav \
    --save-classification-scores=/tmp/scores.pt \
    --save-visualizations=/tmp/vis.png \
    --limit-audio-hours=1 \
    --load-raven-files ./data/Rumble \
    --save-raven-file /tmp/new_labels.raven \
    --discrete-colormap
```

Most options except the audio file(s) are optional.

* `--save-raven-file` - will save a raven file of the sections of the audio file it thinks are rumbles
* `--save-visualizations` - will save images showing how it classified different sections of the audio.
* `--save-classification-scores` - saves a file showing how it classified each 320ms of audio in the file.
* `--load-raven-file` - will overlay labels from a raven file on a visualization.


## Performance Notes

This really wants a GPU.   

* It processes an hour of audio in 1.5 seconds on a 2060
* It processes an hour of audio in 77 seconds on a CPU

so about a 50x speedup on a GPU.



# TODO

    - annoying bug --- Every hour my labels slip by a sample :(
      So by hour 19, the green things are many seconds before where they should be

    - annoying bug -- the final hour is often not a full 60 minutes; and
      my two graphs aren't lining up.
