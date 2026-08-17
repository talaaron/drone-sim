"""Package shared by the Drone (drone/) and the Ground Control Station (gcs/).

Both processes import the same protocol module from here, so
"what the Drone sends" and "what the GCS decodes" can't drift apart.
"""
