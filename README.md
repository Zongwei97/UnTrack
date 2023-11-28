# UnTrack

Official implementation of "Single-Model and Any-Modality for Video Object Tracking" ([arxiv](https://arxiv.org/abs/2311.15851))

We propose Un-Track, a Unified Tracker of a single set of parameters for any modality, which learns their common latent space with only the RGB-X pairs. This unique shared representation seamlessly binds all modalities together, enabling effective unification and accommodating any missing modality, all within a single transformer-based architecture and without the need for modality-specific fine-tuning. 

# Visual Comparison

We compare with [ViPT](https://github.com/jiawen-zhu/ViPT) (SOTA finetuning method) and [SeqTrack](https://github.com/microsoft/VideoX/tree/master/SeqTrack) (SOTA Tracker):

### Depth Domain

![depth](https://github.com/Zongwei97/UnTrack/assets/56023848/6a6404a3-04dd-42e4-bab4-597b80dbbb28)

### Thermal Domain

![thermal](https://github.com/Zongwei97/UnTrack/assets/56023848/30c49f81-54c3-455e-8b29-de3b3cbe412e)

### Event Domain

![event](https://github.com/Zongwei97/UnTrack/assets/56023848/4b5ba910-d3d8-45e5-9404-96726e416ea0)


# Acknowledgments
This repository is heavily based on [ViPT](https://github.com/jiawen-zhu/ViPT) and [OSTrack](https://github.com/botaoye/OSTrack). Thanks to their great work!

