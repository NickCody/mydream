from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import requests
import cv2

def get_retry_session(
    retries=5,
    backoff_factor=1,
    status_forcelist=(500, 502, 503, 504),
    session=None
):
    session = session or requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def resize_and_crop(frame, target_width=640, target_height=512):
    """
    Resizes an image to fit the given target height while keeping the aspect ratio,
    then crops the width to exactly `target_width` (centered).

    Parameters:
    - frame (numpy.ndarray): Input image.
    - target_width (int): Desired output width after cropping (default: 640).
    - target_height (int): Desired output height after resizing (default: 512).

    Returns:
    - Cropped and resized image (numpy.ndarray).
    """

    # Step 1: Get original dimensions
    orig_height, orig_width = frame.shape[:2]

    # Step 2: Compute new width while keeping aspect ratio
    new_width = int((target_height / orig_height) * orig_width)

    # Step 3: Resize the image while keeping aspect ratio
    frame_resized = cv2.resize(frame, (new_width, target_height), interpolation=cv2.INTER_AREA)

    # Step 4: Crop left/right to get exactly target_width
    crop_x_start = max(0, (new_width - target_width) // 2)  # Center cropping
    crop_x_end = crop_x_start + target_width

    # Ensure we don't crop beyond image bounds
    if crop_x_end > new_width:
        crop_x_end = new_width
        crop_x_start = crop_x_end - target_width

    frame_cropped = frame_resized[:, crop_x_start:crop_x_end]

    return frame_cropped    
