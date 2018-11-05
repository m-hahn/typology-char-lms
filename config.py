# Note: The path names shown in the template can be ignored. No two of the paths have to be the same, even when they are the same in the template.

# link to a local copy of the Acqdiv Github repo
#ACQDIV_HOME = "/private/home/mbaroni/acqdiv/acqdiv-database/"
ACQDIV_HOME = "/private/home/mbaroni/acqdiv/georgia_data/acqdiv-database-master"

# for storing model checkpoints (existing models also go here)
#CHECKPOINT_HOME = "/private/home/mbaroni/acqdiv/"
CHECKPOINT_HOME = "/private/home/mbaroni/acqdiv/georgia_data/checkpoint"

#################################
# The following are paths for model output.

# for storing a word-level vocabulary
VOCAB_HOME = "/private/home/mbaroni/acqdiv/georgia_data/data"

# for storing a character-level vocabulary
CHAR_VOCAB_HOME = "/private/home/mbaroni/acqdiv/georgia_data/checkpoint/"

# for storing numerical results after different numbers of epochs
TRAJECTORIES_HOME = "/private/home/mbaroni/acqdiv"
 
# for storing visualizations
VISUALIZATIONS_HOME = "/private/home/mbaroni/acqdiv/figures/"


# For safety, add a slash.
ACQDIV_HOME += "/"
CHECKPOINT_HOME += "/"
VOCAB_HOME += "/"
CHAR_VOCAB_HOME += "/"
TRAJECTORIES_HOME += "/"
VISUALIZATIONS_HOME += "/"



