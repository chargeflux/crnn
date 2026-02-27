import logging
from os import environ

TRACE = logging.DEBUG - 1


def configure_logging(log_level="INFO") -> None:
    logging.addLevelName(TRACE, "TRACE")

    log_level = environ.get("LOGGING_LEVEL", log_level).upper()
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)-5s:%(name)s] %(message)s",
    )

    logging.getLogger("PIL.PngImagePlugin").propagate = False
