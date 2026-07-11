"""Fixed model/audio constants shared across the package.

These mirror upstream yxlllc/RMVPE exactly and must not drift: N_CLASS/CONST
define the 20-cent pitch bins the model was trained to output (see
utils.py's cents<->Hz conversion), and N_MELS/MEL_FMIN/MEL_FMAX/WINDOW_LENGTH
define the mel filterbank the checkpoint's weights expect (see spec.py).
Changing any of these without retraining silently breaks accuracy.

Reads: nothing (leaf module).
"""

SAMPLE_RATE = 16000
N_CLASS = 360
N_MELS = 128
MEL_FMIN = 30
MEL_FMAX = SAMPLE_RATE // 2
WINDOW_LENGTH = 1024
CONST = 1997.3794084376191
