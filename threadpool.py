import concurrent.futures

def worker1():
	print("Worker thread running")

def worker2():
    print(" worker threads slacking!")
# create a thread pool with 2 threads
pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)

# submit tasks to the pool
count = 0 
while count < 5:
    pool.submit(worker1)
    pool.submit(worker2)
    count+=1

# wait for all tasks to complete
pool.shutdown(wait=True)

print("Main thread continuing to run")
