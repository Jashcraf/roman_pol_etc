def rough_wall_clock(n_frames,
                     exp_t,
                     mode='continuous',
                     readout_t=0.264,
                     transfer_time=3.):
    """
    Compute the rough wall clock time for a given observation sequence.
    
    Computes the total exposure time including readout/transfer overheads
    for a given observation sequence.

    Parameters
    ----------
    n_frames : int or float
        Number of exposures.
    exp_t : float
        Individual frame exposure time in seconds.
    mode : str, optional
        Observation mode - if set to 'continuous' (default), the
        total wall clock time is computed assuming continuous reads 
        with frame transfer at the end. This is the standard for 'science'
        observations. If not set to continuous, it is assumed that each
        frame takes a frame transfer time hit. This is generally the 
        assumed mode for unocculted calibration observations.
    readout_t : float, optional
        Single EXCAM exposure readout time in seconds. Default is 0.264.
    transfer_time : float, optional
        EXCAM Science frame transfer time in seconds. Default is 3.0.

    Returns
    -------
    float
        Total wall clock time in seconds.

    Notes
    -----
    EXCAM has a 0.264 second read time for SCI-sized frames.
    In continuous exposure mode, EXCAM frames can only be transferred once
    every 3 seconds due to data rate limitations. Very short exposures
    will have large overheads!
    
    EETC has a more accurate version built in for fixed-integration-time
    calculation for using EXCAM in "burst" mode.
    
    For SCI observations, transfer occurs while exposing (no n_frames * transfer_time penalty). 
    
    For CAL observations, frames are read out after each exposure and transferred in bulk at the end
    (includes n_frames * readout_time and n_frames * transfer_time penalty).
    """
    # SCI observations are reading out while exposing
    # (no n_frames * readout_time penalty)
    if mode == 'continuous':
        if exp_t < transfer_time:
            frame_t = transfer_time + readout_t 
        else:
            frame_t = exp_t + readout_t
        return frame_t * n_frames + transfer_time
    # CALIBRATION observations take the n_frames * (readout_time + transfer_time) penalty
    else:
        frame_t = exp_t + readout_t
        return n_frames * (frame_t + transfer_time)