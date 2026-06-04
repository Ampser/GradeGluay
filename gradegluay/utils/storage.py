from pathlib import Path

import pandas as pd


STOCK_STATUS = "In_Stock"
SOLD_STATUS = "Sold"
VALID_STATUSES = {STOCK_STATUS, SOLD_STATUS}
GRADE_ORDER = ("S", "M", "L")
CSV_HEADERS = [
    "Timestamp",
    "Total_Width_CM",
    "Grade_Result",
    "Recommended_Price",
    "Status",
    "Sold_Timestamp",
]


def resolve_path(path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def empty_sales_history():
    return pd.DataFrame(columns=CSV_HEADERS)


def initialize_sales_history(sales_history_path) -> None:
    sales_history_path = resolve_path(sales_history_path)

    try:
        sales_history_path.parent.mkdir(parents=True, exist_ok=True)
        if not sales_history_path.exists():
            empty_sales_history().to_csv(sales_history_path, index=False)
    except OSError as exc:
        raise RuntimeError("Unable to initialize sales history.") from exc


def read_sales_history(sales_history_path):
    sales_history_path = resolve_path(sales_history_path)
    initialize_sales_history(sales_history_path)

    try:
        sales_history = pd.read_csv(sales_history_path)
    except pd.errors.EmptyDataError:
        sales_history = empty_sales_history()
    except OSError as exc:
        raise RuntimeError("Unable to read sales history.") from exc

    for column in CSV_HEADERS:
        if column not in sales_history.columns:
            sales_history[column] = None

    sales_history = sales_history[CSV_HEADERS]
    sales_history["Status"] = sales_history["Status"].fillna(STOCK_STATUS)
    sales_history["Status"] = sales_history["Status"].astype("object")
    sales_history["Sold_Timestamp"] = sales_history["Sold_Timestamp"].astype("object")
    sales_history.loc[
        ~sales_history["Status"].isin(VALID_STATUSES),
        "Status",
    ] = STOCK_STATUS
    sales_history["Total_Width_CM"] = pd.to_numeric(
        sales_history["Total_Width_CM"],
        errors="coerce",
    )
    sales_history["Recommended_Price"] = pd.to_numeric(
        sales_history["Recommended_Price"],
        errors="coerce",
    )

    return sales_history


def write_sales_history(sales_history_path, sales_history) -> None:
    sales_history_path = resolve_path(sales_history_path)

    try:
        sales_history_path.parent.mkdir(parents=True, exist_ok=True)
        sales_history[CSV_HEADERS].to_csv(sales_history_path, index=False)
    except OSError as exc:
        raise RuntimeError("Unable to write sales history.") from exc


def append_sales_record(sales_history_path, record):
    sales_history = read_sales_history(sales_history_path)
    sales_history = pd.concat(
        [sales_history, pd.DataFrame([record])],
        ignore_index=True,
    )
    write_sales_history(sales_history_path, sales_history)
    return sales_history


def update_sales_history(sales_history_path, updater):
    sales_history = read_sales_history(sales_history_path)
    updated_sales_history = updater(sales_history)

    if updated_sales_history is None:
        updated_sales_history = sales_history

    write_sales_history(sales_history_path, updated_sales_history)
    return updated_sales_history


def build_stock_summary(sales_history):
    grade_summary = {}
    in_stock_rows = sales_history[sales_history["Status"] == STOCK_STATUS]
    sold_rows = sales_history[sales_history["Status"] == SOLD_STATUS]

    for grade in GRADE_ORDER:
        grade_rows = in_stock_rows[in_stock_rows["Grade_Result"] == grade]
        average_price = grade_rows["Recommended_Price"].mean()
        grade_summary[grade] = {
            "count": int(len(grade_rows)),
            "average_price": (
                None if pd.isna(average_price) else round(float(average_price), 2)
            ),
        }

    return {
        "grades": grade_summary,
        "total_value": round(float(in_stock_rows["Recommended_Price"].sum()), 2),
        "total_revenue": round(float(sold_rows["Recommended_Price"].sum()), 2),
    }
