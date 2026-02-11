import logging

def configure_logging(level=logging.INFO) -> None:
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level)
        return
    logging.basicConfig(
        level=level,
        format="[%(levelname)s] %(message)s",
    )
