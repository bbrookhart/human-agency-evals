import sys

from human_agency_evals.cli import main

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "validate"] + ([]) + sys.argv[1:]
    main()
