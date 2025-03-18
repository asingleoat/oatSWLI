# Use full colour
currently converting to greyscale but colour has more information.
trivial mechanical change: carry through three colour channels, perform three independent cross-correlations, vote for best fit (i.e. we take the max of the cross-correlation as position in the 1 channel case, probably take something like the max of the product of the three in the RGB case).

# CPU multithreading
the job is embarrasingly parallel. we should allow multicore CPU utilization when GPU acceleration isn't available.

# Flats calibration
the ring gauge data look suspiciously parabolic (just by eyeballing). not sure what we're seeing, cosine error? we should allow calibrating against the lowpassed measurement of an optical flat.

# Chirp calibration
we currently extract chirps in a pretty ad-hoc manner from *somewhere* in the data that looks okay, generally handpicked, plus some pretty unprincipled averaging

we should allow loading a saved reference chirp from file

we should produce reference chirps by iterative optimization: select an arbitrary chirp, compute cross-correlations on some sample, align sequences, average aligned sequences, use as new chirp, repeat until convergence

# Meaningful units
currently working in native pixels and frames with no connection to physical units.

image export rescales data to [0,1*2^{bit_depth}], should at least report what the scaling was so that external tools can properly interpret the data
