import csv
import io
from zipfile import ZipFile

import pandas as pd
from fastapi.responses import StreamingResponse

from .pred import predict


def process_batch(images_zip_bytes: bytes, texts_csv_bytes: bytes):
    texts_df = pd.read_csv(io.BytesIO(texts_csv_bytes))
    with ZipFile(io.BytesIO(images_zip_bytes)) as archive:
        images_dict = {name: archive.read(name) for name in archive.namelist()}
    results = []

    for _, row in texts_df.iterrows():
        image_path = row.get("image_path")
        text = row.get("text")
        if not image_path or image_path not in images_dict:
            results.append(
                {"text": text, "informative": None, "humanitarian": None, "damage": None, "error": "Image not found"})
            continue
        try:
            prediction = predict(text, images_dict[image_path])
            results.append({"text": text, "informative": prediction.get("informative"),
                            "humanitarian": prediction.get("humanitarian"), "damage": prediction.get("damage"),
                            "error": None})
        except Exception as e:
            results.append({"text": text, "informative": None, "humanitarian": None, "damage": None, "error": str(e)})

    output_csv = io.StringIO()
    writer = csv.DictWriter(output_csv, fieldnames=["text", "informative", "humanitarian", "damage", "error"])
    writer.writeheader()
    writer.writerows(results)
    output_csv.seek(0)

    return StreamingResponse(iter([output_csv.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=predictions.csv"})
