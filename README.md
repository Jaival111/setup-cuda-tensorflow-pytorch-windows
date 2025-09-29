# GPU Setup Guide (CUDA, cuDNN, TensorFlow, PyTorch)

This guide helps you set up a working GPU environment with **CUDA 11.2**, **cuDNN 8.1**, **TensorFlow 2.10**, and **PyTorch (CUDA 11.3 build)**.

---

## 1. Prerequisites

* NVIDIA GPU with driver (check with 'nvidia-smi' in the terminal) supporting CUDA 11.x (i.e. >450)
* [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) installed (recommended)
* Python **3.10** (works best with both TF and Torch here)

---

## 2. Create a Conda Environment

```bash
conda create -n gpu_set python=3.10 -y
conda activate gpu_set
```

---

## 3. Install CUDA 11.2 & cuDNN 8.1

1. Download **CUDA Toolkit 11.2** from [NVIDIA CUDA archive](https://developer.nvidia.com/cuda-11.2.0-download-archive)
2. Download **cuDNN 8.1 for CUDA 11.2** from [NVIDIA cuDNN archive](https://developer.nvidia.com/rdp/cudnn-archive)
3. Extract cuDNN and copy `bin/`, `include/`, and `lib/` files into CUDA install directory.

   * Linux: `/usr/local/cuda-11.2/`
   * Windows: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\`

4. Add CUDA to path in environment variables.
    * C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin
    * C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\libnvvp

---

## 4. Install TensorFlow 2.10 (GPU)

```bash
pip install tensorflow[and-cuda]==2.10
```

TensorFlow 2.10 is the **last version** with GPU support via pip. Later versions need WSL.

Check installation:

```python
import tensorflow as tf
print(tf.__version__)
print("Built with CUDA:", tf.test.is_built_with_cuda())
print("GPU devices:", tf.config.list_physical_devices('GPU'))
```

---

## 5. Install PyTorch (CUDA 11.3)

PyTorch provides separate builds for each CUDA version. For CUDA 11.3:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu113
```

Check installation:

```python
import torch
print(torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0))
```

---

If tensorflow is working and pytorch isn't (or the other way around) then try building separate envinronment for both tensorflow and pytorch. It should work.

---

## 6. Verify Setup

Run this script to test both TF and Torch:

```python
import tensorflow as tf
import torch

print("TensorFlow version:", tf.__version__)
print("GPUs detected (TF):", tf.config.list_physical_devices('GPU'))

print("PyTorch version:", torch.__version__)
print("CUDA available (Torch):", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU detected (Torch):", torch.cuda.get_device_name(0))
```

---

## 7. Notes

* **TensorFlow requires CUDA 11.2 + cuDNN 8.1**, so stick to those versions.
* **PyTorch is more flexible** and works with CUDA 11.3 build.
* If multiple CUDA versions are installed, ensure your PATH and LD_LIBRARY_PATH point to the correct one.

---

✅ You now have TensorFlow 2.10 (CUDA 11.2, cuDNN 8.1) and PyTorch (CUDA 11.3) in the same environment.

---

## If you run into issues or have questions, please open an issue or reach out.