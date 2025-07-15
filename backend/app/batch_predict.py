from fastapi.responses import StreamingResponse
from zipfile import ZipFile
import pandas as pd
import io
import csv
from .predict import predict

def process_batch(images_zip_bytes: bytes, texts_csv_bytes: bytes):
    texts_df = pd.read_csv(io.BytesIO(texts_csv_bytes))

    with ZipFile(io.BytesIO(images_zip_bytes)) as archive:
        images_dict = {name: archive.read(name) for name in archive.namelist()}

    results = []
    for _, row in texts_df.iterrows():
        image_path = row['image_path']
        text = row['text']

        if image_path not in images_dict:
            results.append({
                "image_path": image_path,
                "informative": None,
                "humanitarian": None,
                "error": "Image not found"
            })
            continue

        try:
            pred = predict(text, images_dict[image_path])
            results.append({
                "image_path": image_path,
                **pred,
                "error": None
            })
        except Exception as e:
            results.append({
                "image_path": image_path,
                "informative": None,
                "humanitarian": None,
                "error": str(e)
            })

    output_csv = io.StringIO()
    writer = csv.DictWriter(output_csv, fieldnames=["image_path", "informative", "humanitarian", "error"])
    writer.writeheader()
    writer.writerows(results)
    output_csv.seek(0)

    return StreamingResponse(iter([output_csv.getvalue()]),
                             media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=predictions.csv"})
