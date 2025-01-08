import ray
import time
import rich

# Define a Ray remote function
@ray.remote
def slow_function(x):
    time.sleep(1)
    return x

def test_ray_integration():
    # Create Ray tasks
    results = [slow_function.remote(i) for i in range(1_000_000)]

    # Get the results
    outputs = ray.get(results)
    assert outputs == [0, 1, 2, 3, 4], "Test failed!"
    print("Test passed!")

if __name__ == "__main__":
    # Connect to the Ray cluster
    ray.init(address='auto')
    rich.print(ray.cluster_resources())
    test_ray_integration()
