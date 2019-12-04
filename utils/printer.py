import time

def printProgressBar(iteration, start_time, total, prefix='', suffix='', length=100, fill='█', done=False):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """

    percent = "{0:.1f}".format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r{} |{}| {}% {}'.format(prefix, bar, percent, suffix), end='')
    # Print New Line on Complete
    if iteration == total-1:
        ending = " || Total runtime: {0:.1f} minutes.".format((time.time() - start_time) / 60) + "\n"
        print('\r{} |{}| {}% {}'.format(prefix, bar, percent, suffix + ending), end='')
        print()

