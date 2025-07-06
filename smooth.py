with open("13smooth.txt") as f:
    smooth_values = [int(line.strip()) for line in f if line.strip()]


def nearest_smooth(x):
    # if x > 216000:
    # hardcoded limit: if we're beyond the precomputed list of 13-smooth numbers, than don't
    # bother with the mixed-radix optimization and just accept a slower fft, unlikely to ever
    # happen in practice as we've precomputed upto 1hr of 60fps footage
    # return x
    last = None
    for val in smooth_values:
        if val > x:
            break
        last = val
    return last
