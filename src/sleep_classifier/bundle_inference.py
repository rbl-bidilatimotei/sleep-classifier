import argparse
import json
import logging
import sys
from pathlib import Path

from sleep_classifier.bundle_loader import EvaluationBundle
from sleep_classifier.bundle_validation import strict_json_loads

LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Load a verified bundle and classify one label-free completed-epoch JSON request."""
    parser = argparse.ArgumentParser(description="Run verified evaluation-bundle inference")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    bundle = EvaluationBundle(args.bundle)
    request = strict_json_loads(args.request.read_bytes())
    response = bundle.predictor().predict(request)
    LOGGER.info("Evaluation bundle=%s fold=%s", bundle.manifest["bundle_id"],
                bundle.manifest["provenance"]["fold_id"])
    sys.stdout.write(json.dumps(response, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
