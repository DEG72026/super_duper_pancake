"""Display the Iris form and validate flower measurements."""

import math
from collections.abc import Mapping

from flask import Flask, render_template, request

app = Flask(__name__)

# Limits use the full sklearn Iris dataset:
# observed minimum * 0.8 and observed maximum * 1.2, in centimetres.
FEATURE_LIMITS = {
    "sepal_length": (3.44, 9.48),
    "sepal_width": (1.60, 5.28),
    "petal_length": (0.80, 8.28),
    "petal_width": (0.08, 3.00),
}

# Preserve the feature order agreed with the API and training model.
FEATURE_NAMES = tuple(FEATURE_LIMITS)


def parse_measurements(form: Mapping[str, str]) -> dict[str, float]:
    """Convert form values to floats and check their allowed ranges.

    Args:
        form: Submitted flower measurements in centimetres.

    Returns:
        Valid measurements as floats in the agreed feature order.

    Raises:
        ValueError: If a field is missing, non-numeric, non-finite,
            or outside its allowed range.

    Examples:
        Accept valid measurements and surrounding whitespace:

        >>> sample = dict(zip(FEATURE_NAMES, (" 5.1 ", "3.5", "1.4", "0.2")))
        >>> parse_measurements(sample)
        {'sepal_length': 5.1, 'sepal_width': 3.5, 'petal_length': 1.4, 'petal_width': 0.2}

        Reject missing or blank measurements:

        >>> parse_measurements({})
        Traceback (most recent call last):
            ...
        ValueError: Sepal length is required.

        >>> parse_measurements({**sample, "sepal_width": " "})
        Traceback (most recent call last):
            ...
        ValueError: Sepal width is required.

        Reject non-numeric input:

        >>> parse_measurements({**sample, "petal_length": "large"})
        Traceback (most recent call last):
            ...
        ValueError: Petal length must be a number.

        Reject infinity and NaN:

        >>> for invalid in ("inf", "nan"):
        ...     try:
        ...         parse_measurements({**sample, "petal_width": invalid})
        ...     except ValueError as exc:
        ...         print(exc)
        Petal width must be a finite number.
        Petal width must be a finite number.

        Catch an example of measurements entered in millimetres:

        >>> parse_measurements({**sample, "sepal_length": "51"})
        Traceback (most recent call last):
            ...
        ValueError: Sepal length must be between 3.44 and 9.48 cm. Check the units.

        Reject values just outside every allowed range:

        >>> for name, (lower, upper) in FEATURE_LIMITS.items():
        ...     for invalid in (lower - 0.01, upper + 0.01):
        ...         try:
        ...             parse_measurements({**sample, name: str(invalid)})
        ...         except ValueError:
        ...             continue
        ...         raise AssertionError(f"Accepted invalid value for {name}")

        Accept both boundaries for every measurement:

        >>> for name, (lower, upper) in FEATURE_LIMITS.items():
        ...     for boundary in (lower, upper):
        ...         result = parse_measurements({**sample, name: str(boundary)})
        ...         assert result[name] == boundary
    """
    measurements = {}

    # check the limits per feature
    for name, (lower, upper) in FEATURE_LIMITS.items():
        # make names readable (from sepal-lenght to Sepal lenghth)
        label = name.replace("_", " ").capitalize()
        # remove surrounding blanks
        raw_value = form.get(name, "").strip()

        # discard empty value
        if not raw_value:
            raise ValueError(f"{label} is required.")

        try:
            value = float(raw_value)
        except ValueError:
            raise ValueError(f"{label} must be a number.") from None
        # when someone tries to bypass the form directly into the flask
        if not math.isfinite(value):
            raise ValueError(f"{label} must be a finite number.")

        # Compare the units to boundaries
        if not lower <= value <= upper:
            raise ValueError(
                f"{label} must be between {lower:g} and {upper:g} cm. "
                "Check the units."
            )
        #store accepted value
        measurements[name] = value

    return measurements


@app.route("/", methods=["GET", "POST"])
def home() -> tuple[str, int]:
    """Display the form and validate submitted measurements.

    Returns:
        Rendered HTML and HTTP status 200, or 400 for invalid input.

    Examples:
        >>> client = app.test_client()
        >>> client.get("/").status_code
        200

        >>> sample = dict(zip(FEATURE_NAMES, ("5.1", "3.5", "1.4", "0.2")))
        >>> client.post("/", data=sample).status_code
        200

        >>> response = client.post("/", data={})
        >>> response.status_code
        400
        >>> "Sepal length is required." in response.get_data(as_text=True)
        True
    """
    measurements = None
    error = None

    if request.method == "POST":
        try:
            measurements = parse_measurements(request.form)
        except ValueError as exc:
            error = str(exc)

    # Pass the same limits to HTML IOT have browser and server rules agree.
    html = render_template(
        "index.html",
        feature_limits=FEATURE_LIMITS,
        measurements=measurements,
        error=error,
    )
    status_code = 400 if error else 200

    return html, status_code

@app.get("/health")
def health() -> tuple[dict[str, str], int]:
    """Report whether the UI server can handle a request.

    This check does not contact the prediction API or load a model.

    Returns:
        A JSON response identifying the service and HTTP status 200.

    Examples:
        >>> client = app.test_client()
        >>> response = client.get("/health")
        >>> response.status_code
        200
        >>> response.get_json()
        {'service': 'iris-ui', 'status': 'ok'}
    """
    return {"service": "iris-ui", "status": "ok"}, 200


if __name__ == "__main__":
    # This server is for local development.
    app.run(host="127.0.0.1", port=5001)