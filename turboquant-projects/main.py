from turboquant import TurboQuant
import numpy as np

vectors = np.random.random(1000)

tq = TurboQuant(dim = 1000, bit_width = 4)

y = tq.quantize(vectors)
indices = y.indices

print(indices)
memory_loss = vectors.nbytes - indices.nbytes 

print('memory loss -', memory_loss)
y = tq.dequantize(y)
print('information loss in recovery -', np.mean((vectors - y) ** 2))