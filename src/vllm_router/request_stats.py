from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict

from vllm_router.log import init_logger

logger = init_logger(__name__)


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


@dataclass
class RequestStats:
    # Number of queries per second
    qps: float

    # Average time-to-first-token in seconds
    ttft: float

    # Total number of requests during prefilling
    in_prefill_requests: int

    # Total number of requests during decoding
    in_decoding_requests: int

    # Total number of requests finished
    finished_requests: int

    # How long does this url serves requests
    # NOTE (ApostaC): consider moving this to engine stats
    uptime: int


class MovingAverageMonitor:
    """
    Monitors the average of the value of in a sliding window
    """

    def __init__(self, sliding_window_size: float):
        self.sliding_window_size = sliding_window_size
        self.timestamps = deque()
        self.values = deque()

    def update(self, timestamp: float, value: float):
        """
        Update the throughput monitor with a new timestamp
        """
        self.timestamps.append(timestamp)
        self.values.append(value)
        while (
            len(self.timestamps) > 0
            and self.timestamps[0] < timestamp - self.sliding_window_size
        ):
            self.timestamps.popleft()
            self.values.popleft()

    def get_average(self) -> float:
        """
        Get the throughput in the sliding window
        """
        return sum(self.values) / len(self.values)

    def get_sum(self) -> float:
        """
        Get the sum of the values in the sliding window
        """
        return sum(self.values)


class RequestStatsMonitor(metaclass=SingletonMeta):
    """
    Monitors the request statistics of all serving engines
    """

    # NOTE (ApostaC): Currently, QPS is calculated based on the number of
    # arrived requests in the sliding window, but the inter_token_latency and
    # ttft are calculated based on the number of completed requests in the
    # sliding window.
    def __init__(self, sliding_window_size: float = None):
        if hasattr(self, "_initialized"):
            return
        if sliding_window_size is None:
            raise ValueError(
                "RequestStatsMonitor must be initialized with sliding_window_size"
            )
        self.sliding_window_size = sliding_window_size
        self.qps_monitors: Dict[str, MovingAverageMonitor] = {}
        self.ttft_monitors: Dict[str, MovingAverageMonitor] = {}
        self.request_coming_time: Dict[(str, str), float] = {}
        self.in_prefill_requests: Dict[str, int] = {}
        self.in_decoding_requests: Dict[str, int] = {}
        self.finished_requests: Dict[str, int] = {}
        self.first_query_time = None
        self._initialized = True

    def on_new_request(self, engine_url: str, request_id: str, timestamp: float):
        """
        Tell the monitor that a new request has been created.

        Args:
            engine_url: The URL of the serving engine
            request_id: The global request ID
            timestamp: the timestamp when the request was created
        """
        self.request_coming_time[(engine_url, request_id)] = timestamp

        if engine_url not in self.in_prefill_requests:
            self.in_prefill_requests[engine_url] = 0
        self.in_prefill_requests[engine_url] += 1

        if engine_url not in self.qps_monitors:
            self.qps_monitors[engine_url] = MovingAverageMonitor(
                self.sliding_window_size
            )

        self.qps_monitors[engine_url].update(timestamp, 1)

        if self.first_query_time is None:
            self.first_query_time = timestamp

    def on_request_response(self, engine_url: str, request_id: str, timestamp: float):
        """
        Tell the monitor that a response token has been received for a request.

        Args:
            engine_url: The URL of the serving engine
            request_id: The global request ID
            timestamp: The timestamp when the response token was received
        """
        if (engine_url, request_id) not in self.request_coming_time:
            return
        coming_time = self.request_coming_time.pop((engine_url, request_id))

        if engine_url not in self.in_decoding_requests:
            self.in_decoding_requests[engine_url] = 0
        self.in_prefill_requests[engine_url] -= 1
        self.in_decoding_requests[engine_url] += 1

        if engine_url not in self.ttft_monitors:
            self.ttft_monitors[engine_url] = MovingAverageMonitor(
                self.sliding_window_size
            )
        self.ttft_monitors[engine_url].update(timestamp, timestamp - coming_time)

    def on_request_complete(self, engine_url: str, request_id: str, timestamp: float):
        """
        Tell the monitor that a request has been completed.

        Args:
            engine_url: The URL of the serving engine
            request_id: The global request ID
            timestamp: The timestamp when the request was completed
        """
        if engine_url not in self.finished_requests:
            self.finished_requests[engine_url] = 0
        self.in_decoding_requests[engine_url] -= 1
        self.finished_requests[engine_url] += 1

    def get_request_stats(
        self,
        current_time: float,
    ) -> Dict[str, RequestStats]:
        """
        Get the request statistics for each serving engine

        Args:
            current_time: The current timestamp in seconds

        Returns:
            A dictionary where the key is the serving engine URL and the value
            is the request statistics for that engine.
            The TTFT and inter token latency will be -1 if there is no requests
            finished in the sliding window.
        """
        # Calculate the request statistics
        ret = {}

        # Get all urls:
        urls = set(self.in_prefill_requests.keys()).union(
            set(self.in_decoding_requests.keys())
        )

        for engine_url in urls:
            if engine_url not in self.qps_monitors:
                qps = -1
            else:
                qps = self.qps_monitors[engine_url].get_sum() / self.sliding_window_size

            if engine_url not in self.ttft_monitors:
                ttft = -1
            else:
                ttft = self.ttft_monitors[engine_url].get_average()

            in_prefill_requests = self.in_prefill_requests.get(engine_url, 0)
            in_decoding_requests = self.in_decoding_requests.get(engine_url, 0)
            finished_requests = self.finished_requests.get(engine_url, 0)

            ret[engine_url] = RequestStats(
                qps=qps,
                ttft=ttft,
                in_prefill_requests=in_prefill_requests,
                in_decoding_requests=in_decoding_requests,
                finished_requests=finished_requests,
                uptime=current_time - self.first_query_time,
            )

        return ret


def InitializeRequestStatsMonitor(sliding_window_size: float):
    return RequestStatsMonitor(sliding_window_size)


def GetRequestStatsMonitor():
    return RequestStatsMonitor()
