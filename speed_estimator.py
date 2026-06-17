import time

positions = {}
times = {}

def estimate_speed(track_id,x,y):

    current_time = time.time()

    if track_id in positions:

        prev_x,prev_y = positions[track_id]
        prev_time = times[track_id]

        distance = ((x-prev_x)**2 + (y-prev_y)**2)**0.5
        dt = current_time-prev_time

        if dt == 0:
            speed = 0
        else:
            # Speed in pixels per second (approximation).
            speed = int(distance / dt)

    else:
        speed = 0

    positions[track_id] = (x,y)
    times[track_id] = current_time

    return speed