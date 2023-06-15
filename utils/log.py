import os
def create_logger(log_filename, display=True):
    f = open(log_filename, 'a')
    counter = [0]
    def logger(text):
        if display:
            print(text)
        f.write(text + '\n')
        counter[0] += 1
        if counter[0] % 10 == 0:
            f.flush()
            os.fsync(f.fileno())
    return logger, f.close
