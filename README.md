# PyTorch desde Cero — Implementación manual de capas, optimizadores y funciones
 
Proyecto final de la asignatura **Deep Learning / PyTorch** (IMAT — Comillas ICAI).  
Implementación manual en PyTorch de más de 100 componentes de redes neuronales: funciones de activación, capas convolucionales, recurrentes, de normalización, dropout, padding, pooling, funciones de pérdida, optimizadores y schedulers de learning rate — todos implementados desde cero usando `torch.autograd.Function` con forward y backward personalizados.
 
---
 
## Descripción
 
El objetivo del proyecto es comprender los mecanismos internos de PyTorch reimplementando sus componentes principales sin usar las funciones de alto nivel de `torch.nn`. Cada módulo define su propio grafo de computación con forward pass y backward pass manual para el cálculo de gradientes mediante backpropagation.
 
---
 
## Estructura del proyecto
 
```
PYTORCH_FINAL/
├── Funciones_Activacion/       # 24 funciones de activación
├── Funciones Perdida/          # 15 funciones de pérdida
├── Optimizadores/              # 17 optimizadores
├── Scheduler/                  # 16 schedulers de learning rate
├── Conv-Layers/                # Convoluciones 1D, transpuestas, Fold/UnFold, ResNet
├── Recurrent_Layers/           # RNN, LSTM, GRU, variantes BiDir y TriDim
├── Pooling_Layers/             # MaxPooling y AvgPooling 1D/2D (eficiente y no eficiente)
├── Normalizations-Layers/      # BatchNorm, LayerNorm, GroupNorm, InstanceNorm, RMSNorm
├── Dropout-Layers/             # Dropout, Dropout1D, Dropout2D, AlphaDropout
├── Padding-Layers/             # Zero, Constant, Reflection, Replication, Circular (1D y 2D)
├── Linear-Layers/              # Linear, BiLinear, Identity
├── Sparse-Layers/              # Embedding + tests con pytest
├── Repaso_*/                   # Carpetas de repaso por bloque temático
└── Cargar datos y ficheros/    # Utilidades de carga de datos
```
 
---
 
## Cómo ejecutar
 
**Requisitos:** Python 3.10+
 
```bash
pip install torch numpy pytest
```
 
Cada archivo es independiente y ejecutable directamente:
 
```bash
python Funciones_Activacion/ReLU.py
python Recurrent_Layers/LSTM.py
python Optimizadores/Adam.py
```
 
Tests con pytest:
```bash
pytest Sparse-Layers/
```
 
---
 
## Implementaciones destacadas
 
### Funciones de activación (`Funciones_Activacion/`)
 
Todas implementadas con `torch.autograd.Function` con forward y backward propios. Incluye:
 
ReLU, ReLU6, LeakyReLU, pReLU, RReLU, ELU, CELU, SELU, GELU, SiLU, Mish, HardTanh, HardSigmoid, Hardswish, Hardshrink, SoftShrink, SoftSign, SoftPlus, Tanshrink, Threshold, GLU, MaxOut, MaxOut2Lineal, MaxOut3Lineal
 
**Ejemplo — ReLU:**
```python
class ReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs):
        mascara_back = torch.zeros(inputs.shape)
        mascara_back[inputs > 0] = 1
        ctx.save_for_backward(mascara_back)
        return inputs * mascara_back
 
    @staticmethod
    def backward(ctx, grad_outputs):
        derivada_relu, = ctx.saved_tensors
        return grad_outputs * derivada_relu
```
 
### Capas recurrentes (`Recurrent_Layers/`)
 
RNN, RNNCell, LSTM, LSTMCell, LSTMTriDim, GRU, GRUCell, GRUSinConcat, GRUTriDim, BiRNN, BiLSTM, BiGRU
 
**LSTM implementado desde cero** con las 4 puertas (forget, input, output, cell) calculadas manualmente en cada time step:
 
```python
forget_gate   = sigmoid(concat @ W_f + b_f)
input_gate    = sigmoid(concat @ W_i + b_i)
output_gate   = sigmoid(concat @ W_o + b_o)
cell_gate     = tanh(concat @ W_c + b_c)
new_cell      = forget_gate * c0 + input_gate * cell_gate
new_h         = output_gate * tanh(new_cell)
```
 
Inicialización Xavier uniform de todos los pesos.
 
### Optimizadores (`Optimizadores/`)
 
SGD, SGDMomentum, SGDNesterov, Adam, AdamW, Adamax, NAdam, RAdam, SparseAdam, Adagrad, AdaDelta, AdaFactor, RMSProp, RProp, ASGD, LBFGS
 
**Adam implementado** con bias correction y weight decay:
```python
m = β₁·m + (1−β₁)·g
v = β₂·v + (1−β₂)·g²
m̂ = m / (1−β₁ᵗ)
v̂ = v / (1−β₂ᵗ)
θ = θ − lr · m̂ / (√v̂ + ε)
```
 
### Schedulers de learning rate (`Scheduler/`)
 
StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau, CyclicLR, OneCycleLR, LinearLR, ConstantLR, PolynomialLR, LambdaLR, MultiplicativeLR, SequentialLR, ChainedLR
 
### Capas convolucionales (`Conv-Layers/`)
 
Conv1D, Conv1DF, Conv1DTransposed, Fold1D, Fold2D, UnFold1D, UnFold2D, ResNetConv1D
 
Implementación eficiente de pooling usando `unfold` de PyTorch para vectorizar la operación:
```python
inputs_unfolded = inputs.unfold(2, kernel_h, stride_h).unfold(3, kernel_w, stride_w)
output, max_idx = inputs_unfolded.reshape(N, C, Hout, Wout, -1).max(dim=-1)
```
 
### Normalización (`Normalizations-Layers/`)
 
BatchNorm (con forward y backward del gradiente completo), LayerNorm, GroupNorm, InstanceNorm, RMSNorm, LocalResponseNorm
 
**BatchNorm backward manual** con cálculo correcto de los gradientes de γ, β y la entrada:
```python
grad_gamma  = Σ(grad_out · x̂)
grad_beta   = Σ(grad_out)
grad_inputs = (1/√(σ²+ε)) · (grad_x̂ − mean(grad_x̂) − x̂·mean(grad_x̂·x̂))
```
 
### Funciones de pérdida (`Funciones Perdida/`)
 
MSELoss, L1Loss, HuberLoss, SmoothL1Loss, BCELoss, BCEWithLogitsLoss, CrossEntropyLoss, NLLLoss, KLDivLoss, PoissonNLLoss, SoftMarginLoss, MultiMarginLoss, MultiLabelSoftMarginLoss, MarginRankingLoss, TripletMarginLoss, CTCLoss, CosineEmbeddingLoss
 
---
 
## Tecnologías
 
- Python 3.10
- `torch` — framework base; se usa solo `torch.autograd.Function`, `torch.nn.Module` y operaciones tensoriales
- `pytest` — tests unitarios para capas recurrentes, pooling y embeddings
- `numpy` — operaciones auxiliares
---
 
## Autor
 
**Guzman Ignacio Perez Ibarz**
 
Proyecto final — Deep Learning con PyTorch, Grado en Ingeniería Matemática e Inteligencia Artificial (IMAT), Comillas ICAI.
