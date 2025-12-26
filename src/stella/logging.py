import os
import logging


def _configure_logger(
    logger: logging.Logger,
    log_file: str = None,
    log_file_mode: str = "w",
    fmt: str = "%(asctime)s | %(levelname)s | %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
    main_process_level: int = logging.INFO,
    other_process_level: int = logging.WARN,
    local_rank: int = -1
) -> None:
    """Shared logic for configuring loggers."""
    logger.setLevel(logging.DEBUG)

    # ---- Clear existing handlers ----
    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    # ---- Console handler ----
    console_handler = logging.StreamHandler()
    console_handler.setLevel(
        main_process_level if local_rank in [-1, 0] else other_process_level
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # ---- File handler ----
    if log_file is not None and local_rank in [-1, 0]:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(
            log_file, mode=log_file_mode, encoding="utf-8"
        )
        file_handler.setLevel(main_process_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


def setup_root_logger(
    log_file: str = None,
    log_file_mode: str = "w",
    fmt: str = "%(asctime)s | %(levelname)s | %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
    main_process_level: int = logging.INFO,
    other_process_level: int = logging.WARN,
    local_rank: int = -1,
) -> None:
    """Configure root logger."""
    logger = logging.getLogger()  # root logger
    _configure_logger(
        logger,
        log_file,
        log_file_mode,
        fmt,
        datefmt,
        main_process_level,
        other_process_level,
        local_rank
    )


def get_logger(
    name: str,
    log_file: str = None,
    log_file_mode: str = "w",
    fmt: str = "%(asctime)s | %(levelname)s | %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
    main_process_level: int = logging.INFO,
    other_process_level: int = logging.WARN,
    local_rank: int = -1,
) -> logging.Logger:
    """Create a logger independent from root."""
    logger = logging.getLogger(name)
    logger.propagate = False
    _configure_logger(
        logger,
        log_file,
        log_file_mode,
        fmt,
        datefmt,
        main_process_level,
        other_process_level,
        local_rank,
    )
    return logger


def _get_library_name() -> str:
    return __name__.split(".")[0]


def _get_library_root_logger() -> logging.Logger:
    return logging.getLogger(_get_library_name())


def _configure_library_root_logger() -> logging.Logger:
    """Configure the library's root logger using the default logging settings."""
    logger = _get_library_root_logger()
    logger.propagate = False  # By default, prevent log records from propagating to the root logger.
    _configure_logger(logger)
    return logger


def set_verbosity(level: int) -> None:
    """
    Set verbosity level for the library's root logger and all its handlers.
    ⚠️ Calling this function always applies the library's default logging configuration, 
    even if the user has already customized logging settings.
    """
    logger = _configure_library_root_logger()
    logger.setLevel(level)

    # also update handler levels
    for handler in logger.handlers:
        handler.setLevel(level)


def set_verbosity_debug() -> None:
    """Shortcut: set library verbosity to DEBUG."""
    set_verbosity(logging.DEBUG)


def set_verbosity_info() -> None:
    """Shortcut: set library verbosity to INFO."""
    set_verbosity(logging.INFO)


def set_verbosity_warning() -> None:
    """Shortcut: set library verbosity to WARNING."""
    set_verbosity(logging.WARNING)


def set_verbosity_error() -> None:
    """Shortcut: set library verbosity to ERROR."""
    set_verbosity(logging.ERROR)


def set_verbosity_critical() -> None:
    """Shortcut: set library verbosity to CRITICAL."""
    set_verbosity(logging.CRITICAL)
