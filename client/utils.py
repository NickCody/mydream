from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import requests

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