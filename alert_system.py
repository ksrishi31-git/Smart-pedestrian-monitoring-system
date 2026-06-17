def send_alert(plate,speed,time):

    message = f"""
==============================
TRAFFIC VIOLATION ALERT
==============================
Plate : {plate}
Speed : {speed} km/h
Time  : {time}

Vehicle did not stop at
pedestrian crossing
==============================
"""

    print(message)