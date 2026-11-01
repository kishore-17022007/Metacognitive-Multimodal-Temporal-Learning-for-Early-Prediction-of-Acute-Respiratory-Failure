"""Structured JSON logging. Never log patient-level values; log ids hashed/omitted."""
import json, logging, sys, time


class JsonFormatter(logging.Formatter):
    def format(self, record):
        d = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "level": record.levelname, "logger": record.name, "msg": record.getMessage()}
        extra = getattr(record, "fields", None)
        if extra:
            d.update(extra)
        return json.dumps(d)


def get_logger(name: str = "tbi_mlops") -> logging.Logger:
    lg = logging.getLogger(name)
    if not lg.handlers:
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(JsonFormatter())
        lg.addHandler(h)
        lg.setLevel(logging.INFO)
        lg.propagate = False
    return lg
