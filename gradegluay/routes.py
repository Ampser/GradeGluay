from datetime import datetime
from pathlib import Path
from uuid import uuid4

from flask import Blueprint, current_app, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

from gradegluay import limiter
from gradegluay.utils.image_processing import process_banana_image
from gradegluay.utils.storage import (
    SOLD_STATUS,
    STOCK_STATUS,
    append_sales_record,
    build_stock_summary,
    empty_sales_history,
    initialize_sales_history,
    read_sales_history,
    update_sales_history,
)


main_bp = Blueprint("main", __name__)
IMAGE_PROCESSING_ERROR = "เกิดข้อผิดพลาดระหว่างประมวลผลรูปภาพ กรุณาลองใหม่อีกครั้ง"
SALES_HISTORY_ERROR = "ไม่สามารถบันทึกหรืออ่านประวัติการขายได้ กรุณาลองใหม่อีกครั้ง"


def allowed_image(filename: str) -> bool:
    return Path(filename).suffix.lower() in current_app.config["ALLOWED_EXTENSIONS"]


def cleanup_file(path) -> None:
    if path is None:
        return

    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        current_app.logger.warning("Failed to clean up upload: %s", path)


def save_uploaded_image(image, upload_dir):
    upload_dir = Path(upload_dir)
    original_name = secure_filename(image.filename)
    filename = f"{uuid4().hex}_{original_name}"
    image_path = upload_dir / filename

    try:
        upload_dir.mkdir(parents=True, exist_ok=True)
        image.save(image_path)
    except OSError as exc:
        raise RuntimeError("Unable to save uploaded image.") from exc

    return image_path


@main_bp.route("/", methods=["GET"])
@limiter.limit("60 per hour")
def guide():
    return render_template("guide.html")


@main_bp.route("/upload", methods=["GET", "POST"])
@limiter.limit("30 per hour")
def index():
    try:
        initialize_sales_history(current_app.config["SALES_HISTORY_PATH"])
    except RuntimeError:
        current_app.logger.exception("Failed to initialize sales history.")
        return render_template("index.html", error=SALES_HISTORY_ERROR)

    if request.method == "POST":
        image = request.files.get("banana_image")

        if not image or image.filename == "":
            return render_template(
                "index.html",
                error="Please upload a banana comb photo.",
            )

        if not allowed_image(image.filename):
            return render_template(
                "index.html",
                error="Only JPG, JPEG, and PNG images are supported.",
            )

        image_path = None

        try:
            image_path = save_uploaded_image(
                image,
                current_app.config["UPLOAD_FOLDER"],
            )
            result = process_banana_image(
                image_path,
                request.form.get("click_x_ratio"),
                request.form.get("click_y_ratio"),
                current_app.config["ANNOTATED_UPLOAD_FOLDER"],
            )
        except ValueError as error:
            cleanup_file(image_path)
            return render_template("index.html", error=str(error))
        except Exception:
            cleanup_file(image_path)
            current_app.logger.exception("Image processing failed.")
            return render_template("index.html", error=IMAGE_PROCESSING_ERROR)

        return render_template("index.html", result=result)

    return render_template("index.html")


@main_bp.route("/save_record", methods=["POST"])
@limiter.limit("30 per hour")
def save_record():
    sales_history_path = current_app.config["SALES_HISTORY_PATH"]
    record_row = {
        "Timestamp": datetime.now().isoformat(timespec="seconds"),
        "Total_Width_CM": request.form.get("width", type=float),
        "Grade_Result": request.form.get("grade", ""),
        "Recommended_Price": request.form.get("price", type=float),
        "Status": STOCK_STATUS,
        "Sold_Timestamp": None,
    }

    try:
        append_sales_record(sales_history_path, record_row)
    except RuntimeError:
        current_app.logger.exception("Failed to save sales record.")
        return render_template("index.html", error=SALES_HISTORY_ERROR)

    return redirect(url_for("main.index", saved="1"))


@main_bp.route("/history", methods=["GET"])
@limiter.limit("60 per hour")
def history():
    try:
        sales_history = read_sales_history(current_app.config["SALES_HISTORY_PATH"])
    except RuntimeError:
        current_app.logger.exception("Failed to read sales history.")
        sales_history = empty_sales_history()

    records = sales_history.fillna("").to_dict("records")
    summary = build_stock_summary(sales_history)

    return render_template("history.html", records=records, summary=summary)


@main_bp.route("/history/delete/<int:row_index>", methods=["POST"])
@limiter.limit("30 per hour")
def delete_history_record(row_index):
    sales_history_path = current_app.config["SALES_HISTORY_PATH"]

    def delete_row(sales_history):
        if 0 <= row_index < len(sales_history):
            return sales_history.drop(sales_history.index[row_index]).reset_index(
                drop=True,
            )
        return sales_history

    try:
        update_sales_history(sales_history_path, delete_row)
    except RuntimeError:
        current_app.logger.exception("Failed to delete sales history record.")

    return redirect(url_for("main.history"))


@main_bp.route("/history/sell/<int:row_index>", methods=["POST"])
@limiter.limit("30 per hour")
def sell_history_record(row_index):
    sales_history_path = current_app.config["SALES_HISTORY_PATH"]

    def sell_row(sales_history):
        if 0 <= row_index < len(sales_history):
            sales_history.loc[row_index, "Status"] = SOLD_STATUS
            sales_history.loc[row_index, "Sold_Timestamp"] = datetime.now().isoformat(
                timespec="seconds",
            )
        return sales_history

    try:
        update_sales_history(sales_history_path, sell_row)
    except RuntimeError:
        current_app.logger.exception("Failed to sell sales history record.")

    return redirect(url_for("main.history"))


@main_bp.route("/history/cancel_sell/<int:row_index>", methods=["POST"])
@limiter.limit("30 per hour")
def cancel_sell_history_record(row_index):
    sales_history_path = current_app.config["SALES_HISTORY_PATH"]

    def cancel_sell_row(sales_history):
        if 0 <= row_index < len(sales_history):
            sales_history.loc[row_index, "Status"] = STOCK_STATUS
            sales_history.loc[row_index, "Sold_Timestamp"] = None
        return sales_history

    try:
        update_sales_history(sales_history_path, cancel_sell_row)
    except RuntimeError:
        current_app.logger.exception("Failed to cancel sales history record.")

    return redirect(url_for("main.history"))
