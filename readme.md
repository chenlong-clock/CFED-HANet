# **H**ierarchical **A**ugmentation **Net**works(HANet)

Code and dataset for lrec-coling 2024 paper: "Continual Few-shot Event Detection via Hierarchical Augmentation Networks"

The preprint version is now available in [arxiv](https://arxiv.org/abs/2403.17733)!

## Requirements

Dependencies can be installed through running:

```bash
pip install -r requirements.txt
```

## Quick Start

By the following codes, you can run the default setting of HANet on the CFED-MAVEN dataset:

```bash
bash MAVEN_all_fwUCL+TCL.sh
```

Detailed configurations can be seen in *configs.py*

## Citation

Please cite our paper if you use HANet in your work:

```bibtex
@misc{zhang2024continual,
      title={Continual Few-shot Event Detection via Hierarchical Augmentation Networks}, 
      author={Chenlong Zhang and Pengfei Cao and Yubo Chen and Kang Liu and Zhiqiang Zhang and Mengshu Sun and Jun Zhao},
      year={2024},
      eprint={2403.17733},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
