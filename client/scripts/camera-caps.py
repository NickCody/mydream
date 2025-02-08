import cv2

# List of camera properties to check
properties = {
    "CAP_PROP_FRAME_WIDTH": cv2.CAP_PROP_FRAME_WIDTH,
    "CAP_PROP_FRAME_HEIGHT": cv2.CAP_PROP_FRAME_HEIGHT,
    "CAP_PROP_FPS": cv2.CAP_PROP_FPS,
    "CAP_PROP_FOURCC": cv2.CAP_PROP_FOURCC,
    "CAP_PROP_BRIGHTNESS": cv2.CAP_PROP_BRIGHTNESS,
    "CAP_PROP_CONTRAST": cv2.CAP_PROP_CONTRAST,
    "CAP_PROP_SATURATION": cv2.CAP_PROP_SATURATION,
    "CAP_PROP_HUE": cv2.CAP_PROP_HUE,
    "CAP_PROP_GAIN": cv2.CAP_PROP_GAIN,
    "CAP_PROP_EXPOSURE": cv2.CAP_PROP_EXPOSURE,
    "CAP_PROP_AUTO_EXPOSURE": cv2.CAP_PROP_AUTO_EXPOSURE,
    "CAP_PROP_FOCUS": cv2.CAP_PROP_FOCUS,
    "CAP_PROP_AUTOFOCUS": cv2.CAP_PROP_AUTOFOCUS,
}

# Common resolutions to test (add more if needed)
COMMON_RESOLUTIONS = [
    (320, 240), (640, 480), (800, 600), (1024, 768), (1280, 720),
    (1280, 800), (1440, 900), (1600, 900), (1920, 1080), (2560, 1440),
    (3840, 2160)
]

def get_supported_resolutions(camera_index):
    """Returns a list of resolutions supported by the camera."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return None  # Camera is not available

    supported_resolutions = []
    
    for width, height in COMMON_RESOLUTIONS:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # Read back the values to check if the camera accepted the resolution
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if actual_width == width and actual_height == height:
            supported_resolutions.append((width, height))
    
    cap.release()
    return supported_resolutions

def get_camera_name(cap):
    """Attempt to get camera name using CAP_PROP_BACKEND"""
    backend = cap.get(cv2.CAP_PROP_BACKEND)
    return f"Backend {int(backend)}" if backend else "Unknown Camera"

def get_camera_count(max_cameras=10):
    """Returns the number of available cameras by probing indices."""
    available_cameras = []

    for i in range(max_cameras):  # Check indices from 0 to max_cameras-1
        cap = cv2.VideoCapture(i)
        if cap.isOpened():  # If the camera opens successfully, it's available
            available_cameras.append(i)
            cap.release()
        else:
            break  # Stop when no more cameras are found

    return len(available_cameras)

def print_camera_properties(camera_index):
    """Print all available properties for a given camera index"""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return False  # Camera not available

    print(f"\nCamera {camera_index}: {get_camera_name(cap)}")
    
    for prop_name, prop_id in properties.items():
        value = cap.get(prop_id)
        if value == -1 or value == 0:  # Some properties might return -1 or 0 if unsupported
            continue
        # Special case for FOURCC property (convert it to a readable format)
        if prop_name == "CAP_PROP_FOURCC":
            value = "".join([chr(int(value) >> (8 * i) & 0xFF) for i in range(4)])
        print(f"  {prop_name} = {value}")
    
    for res in get_supported_resolutions(camera_index):
        print (f"  Supported Resolution: {res[0]}x{res[1]}")
        
    cap.release()
    return True

if __name__ == "__main__":
    print("Detecting cameras and their properties...\n")
    for i in range(get_camera_count()):  # Iterate over possible camera indices (0-9)
        print_camera_properties(i)