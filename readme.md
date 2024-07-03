# **H**ierarchical **A**ugmentation **Net**works(HANet)

Code and dataset for lrec-coling 2024 paper: "Continual Few-shot Event Detection via Hierarchical Augmentation Networks"

The paper is now available on [(https://aclanthology.org/2024.lrec-main.342/)](https://aclanthology.org/2024.lrec-main.342/)

## Requirements

Dependencies can be installed by running the following code:

```bash
pip install -r requirements.txt
```

## Quick Start for implementation

By the following codes, you can run the default setting of HANet on the CFED-MAVEN dataset:

```bash
bash MAVEN_all_fwUCL+TCL.sh
```

Detailed configurations can be seen in [configs.py](./configs.py)

## Hyperparameters used in the experiments

Some hyperparameters used in the experiments are not fully stated in the `MAVEN_all_fwUCL+TCL.sh`. We show these parameters as follows: 

### Random Seed 
We randomly evaluated each method with random seed ``1, 2, 3, 4, 42''

### Dataset Permutation
The permutation used in the dataset can be found in [data_incremental/{dataset}/perm{i}](./data_incremental/MAVEN)

### Other parameters:

The `--aug-repeat-times` is `5` and the `--joint-da-loss` is 'none'.

## Citation

Please cite our paper if you use HANet in your work:

```bibtex
@inproceedings{zhang-etal-2024-continual-shot,
    title = "Continual Few-shot Event Detection via Hierarchical Augmentation Networks",
    author = "Zhang, Chenlong  and
      Cao, Pengfei  and
      Chen, Yubo  and
      Liu, Kang  and
      Zhang, Zhiqiang  and
      Sun, Mengshu  and
      Zhao, Jun",
    editor = "Calzolari, Nicoletta  and
      Kan, Min-Yen  and
      Hoste, Veronique  and
      Lenci, Alessandro  and
      Sakti, Sakriani  and
      Xue, Nianwen",
    booktitle = "Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)",
    month = may,
    year = "2024",
    address = "Torino, Italia",
    publisher = "ELRA and ICCL",
    url = "https://aclanthology.org/2024.lrec-main.342",
    pages = "3868--3880",
    abstract = "Traditional continual event detection relies on abundant labeled data for training, which is often impractical to obtain in real-world applications. In this paper, we introduce continual few-shot event detection (CFED), a more commonly encountered scenario when a substantial number of labeled samples are not accessible. The CFED task is challenging as it involves memorizing previous event types and learning new event types with few-shot samples. To mitigate these challenges, we propose a memory-based framework: Hierarchical Augmentation Network (HANet). To memorize previous event types with limited memory, we incorporate prototypical augmentation into the memory set. For the issue of learning new event types in few-shot scenarios, we propose a contrastive augmentation module for token representations. Despite comparing with previous state-of-the-art methods, we also conduct comparisons with ChatGPT. Experiment results demonstrate that our method significantly outperforms all of these methods in multiple continual few-shot event detection tasks.",
}

```
