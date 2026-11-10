from .adapter import PaluAdapter
from .common import dry_run_cli


if __name__ == "__main__":
    dry_run_cli(PaluAdapter, "palu")
