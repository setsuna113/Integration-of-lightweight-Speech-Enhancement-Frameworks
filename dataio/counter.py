import multiprocessing

clean_counter = multiprocessing.Value('i', 0)
noise_counter = multiprocessing.Value('i', 0)

def init(c_counter, n_counter):
    """Makes shared counters available to all workers"""
    global clean_counter, noise_counter
    clean_counter, noise_counter = c_counter, n_counter