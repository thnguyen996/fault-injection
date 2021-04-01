# fault-injection
This code simulates the impact of stuck-at faults on deep neural networks


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
2. Download pretrained weight and put it in checkpoint/ folder 
3. Run the code  ```python main.py --method method0 ``` to simulate noise injection

### Noise injection simulation

```python main.py --method method0 ```

### Weight mapping and encoding (Proposed method)

```python main.py --method method2 ```

### Error correction pointer ECP 

```python main.py --method ECP ```
