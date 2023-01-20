import logging

from rich.logging import RichHandler

logger = logging.getLogger("pl-densepose")
logging.basicConfig(
    format="%(message)s",
    datefmt="[%X]",
    level=logging.INFO,
    handlers=[RichHandler()],
)
logger.addHandler(logging.StreamHandler())


if __name__ == "__main__":
    logging.info("Hello, World!")
