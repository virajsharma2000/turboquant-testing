from turboquant import TurboQuant
import time
import numpy as np
import pandas as pd

num_dims = 100

vector_serial_nums = []
memory_losses = []
reconstruction_losses = []
performances = []
dims = []

for i in range(100):
 vectors = np.random.random(num_dims)
 
 start = time.time()
 tq = TurboQuant(dim = num_dims, bit_width = 4)

 y = tq.quantize(vectors)

 indices = y.indices

 y = tq.dequantize(y)

 end = time.time()

 memory_loss = vectors.nbytes - indices.nbytes 
 reconstruction_loss = float(np.mean((vectors - y) ** 2))
 performance = end - start

 vector_serial_nums.append(i + 1)
 memory_losses.append(str(memory_loss) + ' bytes')
 reconstruction_losses.append(reconstruction_loss)
 performances.append(str(performance) + ' secs')
 dims.append(num_dims)

 num_dims += 10

benchmark_dataframe = pd.DataFrame({'vector serial no.':vector_serial_nums, 'dims': dims, 'memory loss':memory_losses, 'reconstruction loss':reconstruction_losses, 'performance':performances})

benchmark_dataframe.to_csv('benchmark_result_metrics.csv')