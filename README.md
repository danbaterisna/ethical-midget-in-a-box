# Ethical Midget in a Box

What each file does:

* `deep_learning_hax.py` contains the details of the model, including all layers.
* `train_hax.py` contains the training script. pass it the `-h` file for parameters.
* `main.py` is a prototype that uses the model trained by `train_hax.py` to detet fake faces.

Run `train_hax` on the dataset, followed by `main.py`.

Run `evaluator.py` on a test dataset to get a confusion matrix.

*NOTE:* `train_hax.py` is currently configured so that only the first 4000 samples are taken in the training set.
The next 1000 is taken into the validation set. (Test sets are handled by `evaluator.py`.) If these parameters need
to be adjusted, look at line 61 and 62 of `train_hax.py`.

