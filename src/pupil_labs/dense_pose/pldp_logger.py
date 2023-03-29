import logging

from rich.logging import RichHandler

logging.basicConfig(
    format="%(message)s",
    datefmt="[%X]",
    level=logging.INFO,
    handlers=[RichHandler()],
)
logger = logging.getLogger("pl-densepose")
logger.addHandler(logging.StreamHandler())


if __name__ == "__main__":
    logging.info("Hello, World!")
