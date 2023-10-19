'''File to demonstrate/test use of concurrency in Python.timer
Python is naturally a single-threaded language, and so in order to run 
multiple processes simultaneously, I must use the multiprocessing module.
'''

from multiprocessing import Process

def fizz(a):
    print('fizz ' + a)


def buzz():
    print('buzz')

def run():
    #Demonstrate that we can add arguments to the process
    proc1 = Process(target=fizz, args=('oof',))  # instantiating without any argument
    proc2 = Process(target=buzz)

    proc1.start()
    proc2.start()

    proc1.join()
    proc2.join()
timer = 0
while timer < 5:
    run()
    timer += 1
    