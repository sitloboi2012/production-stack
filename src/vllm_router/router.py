import argparse
import logging
import threading
import time
import uuid
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, Gauge, generate_latest

from vllm_router.engine_stats import GetEngineStatsScraper, InitializeEngineStatsScraper
from vllm_router.files import initialize_storage
from vllm_router.httpx_client import HTTPXClientWrapper
from vllm_router.protocols import ModelCard, ModelList
from vllm_router.request_stats import (
    GetRequestStatsMonitor,
    InitializeRequestStatsMonitor,
)
from vllm_router.routing_logic import GetRoutingLogic, InitializeRoutingLogic
from vllm_router.service_discovery import (
    GetServiceDiscovery,
    InitializeServiceDiscovery,
    ServiceDiscoveryType,
)
from vllm_router.utils import validate_url

httpx_client_wrapper = HTTPXClientWrapper()
logger = logging.getLogger("uvicorn")

STACK_VERSION = "0.0.1"


@asynccontextmanager
async def lifespan(app: FastAPI):
    httpx_client_wrapper.start()
    yield
    await httpx_client_wrapper.stop()


app = FastAPI(lifespan=lifespan)

# --- Prometheus Gauges ---
# Existing metrics
num_requests_running = Gauge(
    "vllm:num_requests_running", "Number of running requests", ["server"]
)
num_requests_waiting = Gauge(
    "vllm:num_requests_waiting", "Number of waiting requests", ["server"]
)
current_qps = Gauge("vllm:current_qps", "Current Queries Per Second", ["server"])
avg_decoding_length = Gauge(
    "vllm:avg_decoding_length", "Average Decoding Length", ["server"]
)
num_prefill_requests = Gauge(
    "vllm:num_prefill_requests", "Number of Prefill Requests", ["server"]
)
num_decoding_requests = Gauge(
    "vllm:num_decoding_requests", "Number of Decoding Requests", ["server"]
)

# New metrics per dashboard update
healthy_pods_total = Gauge(
    "vllm:healthy_pods_total", "Number of healthy vLLM pods", ["server"]
)
avg_latency = Gauge(
    "vllm:avg_latency", "Average end-to-end request latency", ["server"]
)
avg_itl = Gauge("vllm:avg_itl", "Average Inter-Token Latency", ["server"])
num_requests_swapped = Gauge(
    "vllm:num_requests_swapped", "Number of swapped requests", ["server"]
)


# --- Request Processing & Routing ---
async def process_request(
    method, header, body, backend_url, request_id, endpoint, debug_request=None
):
    """
    Async generator to stream data from the backend server to the client.
    """
    first_token = False
    total_len = 0
    start_time = time.time()
    GetRequestStatsMonitor().on_new_request(backend_url, request_id, start_time)
    logger.info(f"Started request {request_id} for backend {backend_url}")

    client = httpx_client_wrapper()
    async with client.stream(
        method=method,
        url=backend_url + endpoint,
        headers=dict(header),
        content=body,
        timeout=None,
    ) as backend_response:
        # Yield headers and status code first.
        yield backend_response.headers, backend_response.status_code
        # Stream response content.
        async for chunk in backend_response.aiter_bytes():
            total_len += len(chunk)
            if not first_token:
                first_token = True
                GetRequestStatsMonitor().on_request_response(
                    backend_url, request_id, time.time()
                )
            yield chunk

    GetRequestStatsMonitor().on_request_complete(backend_url, request_id, time.time())
    logger.info(f"Completed request {request_id} for backend {backend_url}")
    # Optional debug logging can be enabled here.
    # logger.debug(f"Finished the request with id: {debug_request.headers.get('x-request-id', None)} at {time.time()}")


async def route_general_request(request: Request, endpoint: str):
    in_router_time = time.time()
    request_id = str(uuid.uuid4())
    request_body = await request.body()
    request_json = await request.json()
    requested_model = request_json.get("model", None)
    if requested_model is None:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid request: missing 'model' in request body."},
        )

    endpoints = GetServiceDiscovery().get_endpoint_info()
    engine_stats = GetEngineStatsScraper().get_engine_stats()
    request_stats = GetRequestStatsMonitor().get_request_stats(time.time())
    endpoints = list(filter(lambda x: x.model_name == requested_model, endpoints))
    if not endpoints:
        return JSONResponse(
            status_code=400, content={"error": f"Model {requested_model} not found."}
        )

    logger.debug(f"Routing request {request_id} for model: {requested_model}")
    server_url = GetRoutingLogic().route_request(
        endpoints, engine_stats, request_stats, request
    )
    logger.info(f"Request {request_id} routed to {server_url}")
    curr_time = time.time()
    logger.info(
        f"Routing request {request_id} to {server_url} at {curr_time}, process time = {curr_time - in_router_time:.4f}"
    )
    stream_generator = process_request(
        request.method,
        request.headers,
        request_body,
        server_url,
        request_id,
        endpoint=endpoint,
    )
    headers, status_code = await anext(stream_generator)
    return StreamingResponse(
        stream_generator,
        status_code=status_code,
        headers={key: value for key, value in headers.items()},
    )


# --- File Endpoints ---
@app.post("/v1/files")
async def route_files(request: Request):
    form = await request.form()
    purpose = form.get("purpose", "unknown")
    if "file" not in form:
        return JSONResponse(
            status_code=400, content={"error": "Missing required parameter 'file'"}
        )
    file_obj: UploadFile = form["file"]
    file_content = await file_obj.read()
    try:
        file_info = await FILE_STORAGE.save_file(
            file_name=file_obj.filename, content=file_content, purpose=purpose
        )
        return JSONResponse(content=file_info.metadata())
    except Exception as e:
        return JSONResponse(
            status_code=500, content={"error": f"Failed to save file: {str(e)}"}
        )


@app.get("/v1/files/{file_id}")
async def route_get_file(file_id: str):
    try:
        file = await FILE_STORAGE.get_file(file_id)
        return JSONResponse(content=file.metadata())
    except FileNotFoundError:
        return JSONResponse(
            status_code=404, content={"error": f"File {file_id} not found"}
        )


@app.get("/v1/files/{file_id}/content")
async def route_get_file_content(file_id: str):
    try:
        # TODO: Stream file content in chunks to support large files.
        file_content = await FILE_STORAGE.get_file_content(file_id)
        return Response(content=file_content)
    except FileNotFoundError:
        return JSONResponse(
            status_code=404, content={"error": f"File {file_id} not found"}
        )


# --- API Endpoints ---
@app.post("/v1/chat/completions")
async def route_chat_completition(request: Request):
    return await route_general_request(request, "/v1/chat/completions")


@app.post("/v1/completions")
async def route_completition(request: Request):
    return await route_general_request(request, "/v1/completions")


@app.get("/version")
async def show_version():
    return JSONResponse(content={"version": STACK_VERSION})


@app.get("/v1/models")
async def show_models():
    endpoints = GetServiceDiscovery().get_endpoint_info()
    existing_models = set()
    model_cards = []
    for endpoint in endpoints:
        if endpoint.model_name in existing_models:
            continue
        model_card = ModelCard(
            id=endpoint.model_name,
            object="model",
            created=endpoint.added_timestamp,
            owned_by="vllm",
        )
        model_cards.append(model_card)
        existing_models.add(endpoint.model_name)
    model_list = ModelList(data=model_cards)
    return JSONResponse(content=model_list.model_dump())


@app.get("/health")
async def health() -> Response:
    if not GetServiceDiscovery().get_health():
        return JSONResponse(
            content={"status": "Service discovery module is down."}, status_code=503
        )
    if not GetEngineStatsScraper().get_health():
        return JSONResponse(
            content={"status": "Engine stats scraper is down."}, status_code=503
        )
    return Response(status_code=200)


# --- Prometheus Metrics Endpoint ---
@app.get("/metrics")
async def metrics():
    # Retrieve request stats from the monitor.
    stats = GetRequestStatsMonitor().get_request_stats(time.time())
    for server, stat in stats.items():
        current_qps.labels(server=server).set(stat.qps)
        # Assuming stat contains the following attributes:
        avg_decoding_length.labels(server=server).set(stat.avg_decoding_length)
        num_prefill_requests.labels(server=server).set(stat.in_prefill_requests)
        num_decoding_requests.labels(server=server).set(stat.in_decoding_requests)
        num_requests_running.labels(server=server).set(
            stat.in_prefill_requests + stat.in_decoding_requests
        )
        avg_latency.labels(server=server).set(stat.avg_latency)
        avg_itl.labels(server=server).set(stat.avg_itl)
        num_requests_swapped.labels(server=server).set(stat.num_swapped_requests)
    # For healthy pods, we use a hypothetical function from service discovery.
    healthy = {}
    endpoints = GetServiceDiscovery().get_endpoint_info()
    for ep in endpoints:
        # Assume each endpoint object has an attribute 'healthy' (1 if healthy, 0 otherwise).
        healthy[ep.url] = 1 if getattr(ep, "healthy", True) else 0
    for server, value in healthy.items():
        healthy_pods_total.labels(server=server).set(value)
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# --- Argument Parsing and Initialization ---
def validate_args(args):
    if args.service_discovery == "static":
        if args.static_backends is None:
            raise ValueError(
                "Static backends must be provided when using static service discovery."
            )
        if args.static_models is None:
            raise ValueError(
                "Static models must be provided when using static service discovery."
            )
    if args.service_discovery == "k8s" and args.k8s_port is None:
        raise ValueError("K8s port must be provided when using K8s service discovery.")
    if args.routing_logic == "session" and args.session_key is None:
        raise ValueError(
            "Session key must be provided when using session routing logic."
        )
    if args.log_stats and args.log_stats_interval <= 0:
        raise ValueError("Log stats interval must be greater than 0.")
    if args.engine_stats_interval <= 0:
        raise ValueError("Engine stats interval must be greater than 0.")
    if args.request_stats_window <= 0:
        raise ValueError("Request stats window must be greater than 0.")


def parse_args():
    parser = argparse.ArgumentParser(description="Run the FastAPI app.")
    parser.add_argument(
        "--host", default="0.0.0.0", help="The host to run the server on."
    )
    parser.add_argument(
        "--port", type=int, default=8001, help="The port to run the server on."
    )
    parser.add_argument(
        "--service-discovery",
        required=True,
        choices=["static", "k8s"],
        help="The service discovery type.",
    )
    parser.add_argument(
        "--static-backends",
        type=str,
        default=None,
        help="The URLs of static backends, separated by commas. E.g., http://localhost:8000,http://localhost:8001",
    )
    parser.add_argument(
        "--static-models",
        type=str,
        default=None,
        help="The models of static backends, separated by commas. E.g., model1,model2",
    )
    parser.add_argument(
        "--k8s-port",
        type=int,
        default=8000,
        help="The port of vLLM processes when using K8s service discovery.",
    )
    parser.add_argument(
        "--k8s-namespace",
        type=str,
        default="default",
        help="The namespace of vLLM pods when using K8s service discovery.",
    )
    parser.add_argument(
        "--k8s-label-selector",
        type=str,
        default="",
        help="The label selector to filter vLLM pods when using K8s service discovery.",
    )
    parser.add_argument(
        "--routing-logic",
        type=str,
        required=True,
        choices=["roundrobin", "session"],
        help="The routing logic to use",
    )
    parser.add_argument(
        "--session-key",
        type=str,
        default=None,
        help="The key (in the header) to identify a session.",
    )
    parser.add_argument(
        "--file-storage-class",
        type=str,
        default="local_file",
        choices=["local_file"],
        help="The file storage class to use.",
    )
    parser.add_argument(
        "--file-storage-path",
        type=str,
        default="/tmp/vllm_files",
        help="The path to store files.",
    )
    parser.add_argument(
        "--engine-stats-interval",
        type=int,
        default=30,
        help="The interval in seconds to scrape engine statistics.",
    )
    parser.add_argument(
        "--request-stats-window",
        type=int,
        default=60,
        help="The sliding window in seconds to compute request statistics.",
    )
    parser.add_argument(
        "--log-stats", action="store_true", help="Log statistics periodically."
    )
    parser.add_argument(
        "--log-stats-interval",
        type=int,
        default=10,
        help="The interval in seconds to log statistics.",
    )
    args = parser.parse_args()
    validate_args(args)
    return args


def parse_static_urls(args):
    urls = args.static_backends.split(",")
    backend_urls = []
    for url in urls:
        if validate_url(url):
            backend_urls.append(url)
        else:
            logger.warning(f"Skipping invalid URL: {url}")
    return backend_urls


def parse_static_model_names(args):
    models = args.static_models.split(",")
    return models


def InitializeAll(args):
    if args.service_discovery == "static":
        InitializeServiceDiscovery(
            ServiceDiscoveryType.STATIC,
            urls=parse_static_urls(args),
            models=parse_static_model_names(args),
        )
    elif args.service_discovery == "k8s":
        InitializeServiceDiscovery(
            ServiceDiscoveryType.K8S,
            namespace=args.k8s_namespace,
            port=args.k8s_port,
            label_selector=args.k8s_label_selector,
        )
    else:
        raise ValueError(f"Invalid service discovery type: {args.service_discovery}")
    InitializeEngineStatsScraper(args.engine_stats_interval)
    InitializeRequestStatsMonitor(args.request_stats_window)
    global FILE_STORAGE
    FILE_STORAGE = initialize_storage(args.file_storage_class, args.file_storage_path)
    InitializeRoutingLogic(args.routing_logic, session_key=args.session_key)


def log_stats(interval: int = 10):
    while True:
        time.sleep(interval)
        logstr = "\n" + "=" * 50 + "\n"
        endpoints = GetServiceDiscovery().get_endpoint_info()
        engine_stats = GetEngineStatsScraper().get_engine_stats()
        request_stats = GetRequestStatsMonitor().get_request_stats(time.time())
        for endpoint in endpoints:
            url = endpoint.url
            logstr += f"Model: {endpoint.model_name}\n"
            logstr += f"Server: {url}\n"
            if url in engine_stats:
                es = engine_stats[url]
                logstr += (
                    f" Engine Stats: Running Requests: {es.num_running_requests}, "
                    f"Queued Requests: {es.num_queuing_requests}, "
                    f"GPU Cache Hit Rate: {es.gpu_prefix_cache_hit_rate:.2f}\n"
                )
            else:
                logstr += " Engine Stats: No stats available\n"
            if url in request_stats:
                rs = request_stats[url]
                logstr += (
                    f" Request Stats: QPS: {rs.qps:.2f}, "
                    f"Avg Latency: {rs.avg_latency}, "
                    f"Avg ITL: {rs.avg_itl}, "
                    f"Prefill Requests: {rs.in_prefill_requests}, "
                    f"Decoding Requests: {rs.in_decoding_requests}, "
                    f"Swapped Requests: {rs.num_swapped_requests}, "
                    f"Finished: {rs.finished_requests}, "
                    f"Uptime: {rs.uptime:.2f} sec\n"
                )
                current_qps.labels(server=url).set(rs.qps)
                avg_decoding_length.labels(server=url).set(rs.avg_decoding_length)
                num_prefill_requests.labels(server=url).set(rs.in_prefill_requests)
                num_decoding_requests.labels(server=url).set(rs.in_decoding_requests)
                num_requests_running.labels(server=url).set(
                    rs.in_prefill_requests + rs.in_decoding_requests
                )
                avg_latency.labels(server=url).set(rs.avg_latency)
                avg_itl.labels(server=url).set(rs.avg_itl)
                num_requests_swapped.labels(server=url).set(rs.num_swapped_requests)
            else:
                logstr += " Request Stats: No stats available\n"
            logstr += "-" * 50 + "\n"
        logstr += "=" * 50 + "\n"
        logger.info(logstr)


def main():
    args = parse_args()
    InitializeAll(args)
    if args.log_stats:
        threading.Thread(
            target=log_stats, args=(args.log_stats_interval,), daemon=True
        ).start()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
