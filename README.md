# fault-injection
This code simulates the impact of emerging memory's stuck-at faults (SAFs) on deep neural networks. It also provides
different SAF-tolerance methods to increase the robustness of DNNs. Details of the proposed methods are
described in:  <em>[Thai-Hoang Nguyen](https://thaihoang.org), Muhammad Imran, Jaehyuk Choi, Joon-Sung Yang (2021). [Low-Cost and Effective Fault-Tolerance Enhancement Techniques for Emerging Memories-Based Deep Neural Networks.](https://doi.org/10.1109%2Fdac18074.2021.9586112) 2021 58th ACM/IEEE Design Automation Conference (DAC).</em>

### Requirements

<!-- Requirements: --> 
1. Python3.6 
2. Pytorch 1.5.1 
3. Cupy 7.8.0: (similar to numpy but runs on gpus)
4. torchsummary (sumarize model)
5. tensorboard (to display the results)
6. Pyinstrument (to profile the code)

### USAGE EXAMPLES
<!-- USAGE EXAMPLES --> 
  
1. Clone the repo ```git clone https://github.com/thnguyen996/fault-injection.git ```
2. Download pretrained weights and put it in checkpoint/ folder 
3. Run the code  ```python main.py --method method0 ``` to simulate stuck-at faults injection

### Noise injection simulation

```
$ python main.py --method method0
```

### Weight mapping and encoding (Proposed method)

```
$ python main.py --method method2
```

### Error correction pointer ECP 

```
$ python main.py --method ECP 
```

### Research papers citing

If you use this code for your research paper, please use the following citation:

```
@inproceedings{nguyen_imran2021,
 author = {Thai-Hoang Nguyen and Muhammad Imran and Jaehyuk Choi and Joon-Sung Yang},
 booktitle = {2021 58th ACM/IEEE Design Automation Conference (DAC)},
 doi = {10.1109/dac18074.2021.9586112},
 month = {dec},
 publisher = {IEEE},
 title = {Low-Cost and Effective Fault-Tolerance Enhancement Techniques for Emerging Memories-Based Deep Neural Networks},
 url = {https://doi.org/10.1109%2Fdac18074.2021.9586112},
 year = {2021}
}
```

