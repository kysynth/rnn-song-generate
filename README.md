# rnn-song-generate
- Use [Recurrent Neural Network](https://en.wikipedia.org/wiki/Recurrent_neural_network) to create songs.
- Note that some features are still under development.

## Requirements
- Python 3
- Tensorflow (It is recommended to get Python 3 and Tensorflow using [Anaconda](https://anaconda.org/), especially if you are using Windows.)
- [mido](http://mido.readthedocs.io/en/latest/)

## Sample
The following samples are created by the model (LSTM, 1 layer, 512 neurons) that was trained by 295 classical piano songs from [Classical Piano Midi Page](http://www.piano-midi.de/) for about 1000 epochs.
- [Sample 1](https://www.dropbox.com/s/vymdt07bkls7o7n/sample_1.mp3?dl=0)
- [Sample 2](https://www.dropbox.com/s/epp41z7zkk63duk/sample_2.mp3?dl=0)

## How to Use
1. Create directories named `midi` and `model`.

2. Put Midi files (.mid) that you want the model to learn from into the `midi` folder.

3. Run `python main.py` to train the model. It can take hours or even days on ordinary computers. 

4. Then find the most recently created files in `model` folder. It should look like this:

```
20171122111012-model-1000.data-00000-of-00001
20171122111012-model-1000.index
20171122111012-model-1000.meta
```

Copy the prefix (in this case, `20171122111012-model-1000`) because you will need it soon.

5. Put a Midi file into the same directory as where `main.py` is in. It can be any Midi as long as it has more than 4 bars. 
For example, I can put `alb_esp1.mid` into the directory.

6. Then run `python main.py --compose --compose_model_name 20171122111012-model-1000 --compose_start_with alb_eps1.mid --compose_song_name awesome_my_new_song`.
You have to use the model from Step 4 and the Midi file from Step 5. You can give your new song any name, of course.

7. After about half a minute, you should be able to see your new song in the directory, such as `awesome_my_new_song.mid`.


There are many other command line options available that you may find useful. You can find them in `main.py`.

## Reference
There are many great projects that use RNN to create songs. Some of them can produce fascinating results by using complicated models.
I have referred to, and particularly recommend the following projects if you are interested:
- [Composing Music With Recurrent Neural Network](http://www.hexahedria.com/2015/08/03/composing-music-with-recurrent-neural-networks/)
- [MusicGenerator](https://github.com/Conchylicultor/MusicGenerator)


