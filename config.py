# Note: The path names shown in the template can be ignored. No two of the paths have to be the same, even when they are the same in the template.

# link to a local copy of the Acqdiv Github repo
ACQDIV_HOME = "acqdiv-database/"

# for storing model checkpoints (existing models also go here)
CHECKPOINT_HOME = "checkpoint/"

#################################
# The following are paths for model output.

# for storing a word-level vocabulary
VOCAB_HOME = "acqdiv/"

# for storing a character-level vocabulary
CHAR_VOCAB_HOME = "checkpoint/"

# for storing numerical results after different numbers of epochs
TRAJECTORIES_HOME = "checkpoint/"

# for storing visualizations
VISUALIZATIONS_HOME = "checkpoint/figures/"


# For safety, add a slash.
ACQDIV_HOME += "/"
CHECKPOINT_HOME += "/"
VOCAB_HOME += "/"
CHAR_VOCAB_HOME += "/"
TRAJECTORIES_HOME += "/"
VISUALIZATIONS_HOME += "/"



