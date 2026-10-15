"""
Request and Response Logging Middleware for debugging API traffic.
Logs incoming requests (method, path, IP, body) and outgoing responses (status code, duration).
"""

import logging
import time

logger = logging.getLogger("rag.traffic")


class RequestResponseLoggingMiddleware:
    """
    Middleware that outputs debug lines to the console for every incoming
    HTTP request and outgoing response, including execution duration.
    """

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        start_time = time.perf_counter()

        # Extract client IP
        x_forwarded_for = request.META.get("HTTP_X_FORWARDED_FOR")
        if x_forwarded_for:
            ip = x_forwarded_for.split(",")[0].strip()
        else:
            ip = request.META.get("REMOTE_ADDR", "127.0.0.1")

        method = request.method
        full_path = request.get_full_path()
        content_type = request.content_type or ""

        # Safely preview request payload without breaking multipart file streams
        body_summary = ""
        if content_type.startswith("multipart/"):
            body_summary = "[Multipart File Upload]"
        else:
            try:
                raw_body = request.body
                if raw_body:
                    decoded = raw_body.decode("utf-8", errors="replace").strip()
                    if decoded:
                        if len(decoded) > 250:
                            body_summary = f"[Payload: {decoded[:250]}...]"
                        else:
                            body_summary = f"[Payload: {decoded}]"
            except Exception:
                body_summary = ""

        # Log incoming request
        if body_summary:
            logger.info(">>> [REQUEST]  %s %s | Client: %s | %s", method, full_path, ip, body_summary)
        else:
            logger.info(">>> [REQUEST]  %s %s | Client: %s", method, full_path, ip)

        # Process the request
        try:
            response = self.get_response(request)
            duration_ms = (time.perf_counter() - start_time) * 1000

            status_code = response.status_code
            status_text = response.reason_phrase if hasattr(response, "reason_phrase") else ""

            # Log outgoing response
            logger.info(
                "<<< [RESPONSE] %s %s | Status: %s %s | Duration: %.2fms",
                method,
                full_path,
                status_code,
                status_text,
                duration_ms,
            )
            return response

        except Exception as exc:
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                "<<< [RESPONSE EXCEPTION] %s %s | Error: %s | Duration: %.2fms",
                method,
                full_path,
                exc,
                duration_ms,
            )
            raise
