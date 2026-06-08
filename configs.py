resample_signals = True
mode = 'train'
vmode = 'val'
batch_size = 32
num_pats = 28

actveeg = actvsea = actaux = actm = 'silu'
mdim = 48
mdimp = 16
emote_dim = 32

do_uda = False
ortho_wt = 0.1

scheduler_type = 'step'
lr = 1e-3
epochs = 40
wdc = 1e-3

max_lr = 3e-3
p_lr = 0.25

gamma = 0.5
step_size = 10