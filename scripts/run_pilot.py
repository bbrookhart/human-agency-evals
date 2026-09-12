import sys

from human_agency_evals.cli import main

if __name__ == "__main__":
    sys.argv = (
        [sys.argv[0], "run"]
        + (["--config", "configs/experiment_pilot.yaml"] if len(sys.argv) == 1 else [])
        + sys.argv[1:]
    )
    main()
