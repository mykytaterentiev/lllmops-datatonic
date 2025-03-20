import os
import logging

if os.getenv("LOCAL_MODE", "false").lower() in ("true", "1"):
    def setup_logger(logger_name="llm_evaluation_pipeline"):
        logger = logging.getLogger(logger_name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def setup_tracing():
        from contextlib import contextmanager
        class DummyTracer:
            @contextmanager
            def start_as_current_span(self, name, attributes=None):
                yield
        return DummyTracer()
else:
    from google.cloud import logging as cloud_logging
    from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry import trace

    def setup_logger(logger_name="llm_evaluation_pipeline"):
        logging_client = cloud_logging.Client()
        logger = logging_client.logger(logger_name)
        return logger

    def setup_tracing():
        exporter = CloudTraceSpanExporter()
        tracer_provider = TracerProvider()
        tracer_provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(tracer_provider)
        return trace.get_tracer(__name__)
