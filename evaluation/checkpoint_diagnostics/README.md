## Checkpoint selection

Checkpoint selection followed a two-stage rule. The loss curve was first used to identify the plateau-qualified checkpoint interval, after which the checkpoint with the lowest mean KID within that interval was selected. Under this rule, epoch 16 was selected for the low-rise LoRA and epoch 20 for the high-rise LoRA. Although epoch 12 had the lowest nominal KID for the low-rise model across all checkpoints, it was excluded because it occurred before the detected plateau.
