import ray
import time

# Start Ray.
ray.init()

t = time.time()

def a(x):
    time.sleep(5)
    print(x)

@ray.remote(num_cpus=0.1)
def f(x):
    a(x)

result_ids = []
for i in range(20):
    result_ids.append(f.remote(i))

# Wait for the tasks to complete and retrieve the results.
# With at least 4 cores, this will take 1 second.
results = ray.get(result_ids)  # [0, 1, 2, 3]
print(time.time() - t)